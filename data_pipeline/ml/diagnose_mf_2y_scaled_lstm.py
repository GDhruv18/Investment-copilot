from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "sequences"
    / "mutual_funds"
    / "2y"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "lstm_results"
    / "mf_2y_scaled_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOCKED TRAINING SETTINGS
# =============================================================================

LOOKBACK = 252
N_FEATURES = 27

BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.20
LEARNING_RATE = 0.001


# =============================================================================
# FEATURE ORDER
# =============================================================================

FEATURE_NAMES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",
    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",
    "nav_vs_sma_20",
    "nav_vs_sma_50",
    "nav_vs_sma_200",
    "nav_range_20d",
    "nav_range_60d",
    "downside_volatility_60d",
    "upside_volatility_60d",
    "rolling_sharpe_60d",
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "nav_range_60d_pct",
]


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_paths(split):

    X_path = (
        SEQUENCE_DIR
        / f"mutual_funds_2y_{split}_X.npy"
    )

    y_path = (
        SEQUENCE_DIR
        / f"mutual_funds_2y_{split}_y.npy"
    )

    return X_path, y_path


# =============================================================================
# TRAIN-ONLY STANDARDIZATION
# =============================================================================

def fit_scaler(X):

    """
    Calculate feature-wise mean and standard deviation
    using TRAINING data only.

    NaN values are ignored when calculating statistics.
    """

    print()
    print("FITTING TRAIN-ONLY STANDARDIZER")
    print("-" * 80)

    n_features = X.shape[2]

    means = np.zeros(
        n_features,
        dtype=np.float64,
    )

    stds = np.zeros(
        n_features,
        dtype=np.float64,
    )

    for i in range(n_features):

        values = np.asarray(
            X[:, :, i],
            dtype=np.float64,
        ).reshape(-1)

        valid = values[
            np.isfinite(values)
        ]

        if len(valid) == 0:
            raise ValueError(
                f"Feature {FEATURE_NAMES[i]} "
                f"contains no finite values."
            )

        means[i] = np.mean(valid)
        stds[i] = np.std(valid)

        if not np.isfinite(means[i]):
            raise ValueError(
                f"Non-finite mean for "
                f"{FEATURE_NAMES[i]}"
            )

        if (
            not np.isfinite(stds[i])
            or stds[i] <= 0
        ):
            raise ValueError(
                f"Invalid standard deviation "
                f"for {FEATURE_NAMES[i]}: "
                f"{stds[i]}"
            )

        print(
            f"{i:2d} "
            f"{FEATURE_NAMES[i]:<28} "
            f"mean={means[i]: .8f} "
            f"std={stds[i]: .8f}"
        )

    return means, stds


# =============================================================================
# DATA GENERATOR
# =============================================================================

class ScaledSequenceGenerator(
    tf.keras.utils.Sequence
):
    """
    Memory-mapped sequence generator.

    Scaling parameters are supplied externally and MUST come
    from the training split only.
    """

    def __init__(
        self,
        X_path,
        y_path,
        batch_size,
        shuffle,
        means,
        stds,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.X = np.load(
            X_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        self.y = np.load(
            y_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X/y mismatch: "
                f"{len(self.X)} vs {len(self.y)}"
            )

        if self.X.ndim != 3:
            raise ValueError(
                "X must be 3-dimensional."
            )

        if self.X.shape[1] != LOOKBACK:
            raise ValueError(
                f"Expected lookback {LOOKBACK}, "
                f"got {self.X.shape[1]}"
            )

        if self.X.shape[2] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, "
                f"got {self.X.shape[2]}"
            )

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.means = np.asarray(
            means,
            dtype=np.float32,
        )

        self.stds = np.asarray(
            stds,
            dtype=np.float32,
        )

        if self.means.shape != (
            N_FEATURES,
        ):
            raise ValueError(
                "Invalid scaler mean shape."
            )

        if self.stds.shape != (
            N_FEATURES,
        ):
            raise ValueError(
                "Invalid scaler std shape."
            )

        self.indices = np.arange(
            len(self.X)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.X)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        start = (
            index
            * self.batch_size
        )

        end = min(
            start
            + self.batch_size,
            len(self.X),
        )

        batch_indices = (
            self.indices[start:end]
        )

        X_batch = np.asarray(
            self.X[batch_indices],
            dtype=np.float32,
        )

        y_batch = np.asarray(
            self.y[batch_indices],
            dtype=np.int8,
        )

        # -------------------------------------------------------------
        # Standardize using TRAIN-ONLY statistics.
        # -------------------------------------------------------------

        X_batch = (
            X_batch - self.means
        ) / self.stds

        # -------------------------------------------------------------
        # Any remaining NaN/infinity indicates a data problem.
        # The original sequence audit established that complete
        # sequences used here should already be finite.
        # -------------------------------------------------------------

        if not np.isfinite(
            X_batch
        ).all():

            raise ValueError(
                "Non-finite value encountered "
                "after scaling."
            )

        return X_batch, y_batch

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# =============================================================================
# MODEL
# =============================================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            LOOKBACK,
            N_FEATURES,
        ),
        name="sequence_input",
    )

    x = tf.keras.layers.LSTM(
        LSTM_UNITS,
        name="lstm_64",
    )(inputs)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        DENSE_UNITS,
        activation="relu",
        name="dense_32",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mf_2y_lstm_scaled_diagnostic",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "INVESTMENT COPILOT - MF 2Y "
        "TRAIN-ONLY SCALING DIAGNOSTIC"
    )
    print("=" * 80)

    print()
    print("CONTROLLED EXPERIMENT")
    print("---------------------")
    print("Features          : 27")
    print("Lookback          : 252")
    print("LSTM              : 64")
    print("Dense             : 32")
    print("Dropout           : 0.20")
    print("Learning rate     : 0.001")
    print("Batch size        : 64")
    print("Max epochs        : 50")
    print("Early stopping    : 7")
    print("Scaling           : Standardization")
    print("Scaler fit on     : TRAIN ONLY")
    print("Validation use    : Transform only")
    print("Test use          : Transform only")
    print("Seed              : 42")

    # -------------------------------------------------------------------------
    # Load train X only for scaler fitting
    # -------------------------------------------------------------------------

    train_X_path, train_y_path = get_paths(
        "train"
    )

    val_X_path, val_y_path = get_paths(
        "validation"
    )

    test_X_path, test_y_path = get_paths(
        "test"
    )

    for path in [
        train_X_path,
        train_y_path,
        val_X_path,
        val_y_path,
        test_X_path,
        test_y_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    train_X = np.load(
        train_X_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    train_y = np.load(
        train_y_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    print()
    print(
        f"Training X shape: "
        f"{train_X.shape}"
    )

    print(
        f"Training y shape: "
        f"{train_y.shape}"
    )

    # -------------------------------------------------------------------------
    # Fit scaler ONLY on training data
    # -------------------------------------------------------------------------

    means, stds = fit_scaler(
        train_X
    )

    # Save scaler parameters for reproducibility.
    scaler_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mean": means,
            "std": stds,
        }
    )

    scaler_df.to_csv(
        OUTPUT_DIR
        / "train_only_scaler.csv",
        index=False,
    )

    del train_X
    del train_y

    gc.collect()

    # -------------------------------------------------------------------------
    # Create generators
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING SCALED GENERATORS")
    print("=" * 80)

    train_generator = ScaledSequenceGenerator(
        train_X_path,
        train_y_path,
        batch_size=BATCH_SIZE,
        shuffle=True,
        means=means,
        stds=stds,
    )

    validation_generator = ScaledSequenceGenerator(
        val_X_path,
        val_y_path,
        batch_size=BATCH_SIZE,
        shuffle=False,
        means=means,
        stds=stds,
    )

    test_generator = ScaledSequenceGenerator(
        test_X_path,
        test_y_path,
        batch_size=BATCH_SIZE,
        shuffle=False,
        means=means,
        stds=stds,
    )

    print(
        f"Train batches      : "
        f"{len(train_generator):,}"
    )

    print(
        f"Validation batches : "
        f"{len(validation_generator):,}"
    )

    print(
        f"Test batches       : "
        f"{len(test_generator):,}"
    )

    # -------------------------------------------------------------------------
    # Verify scaled batch
    # -------------------------------------------------------------------------

    sample_X, sample_y = (
        train_generator[0]
    )

    print()
    print(
        f"Scaled batch shape: "
        f"{sample_X.shape}"
    )

    print(
        f"Scaled batch dtype: "
        f"{sample_X.dtype}"
    )

    print(
        f"Scaled batch min: "
        f"{np.min(sample_X):.6f}"
    )

    print(
        f"Scaled batch max: "
        f"{np.max(sample_X):.6f}"
    )

    print(
        f"Scaled batch mean: "
        f"{np.mean(sample_X):.6f}"
    )

    print(
        f"Scaled batch std: "
        f"{np.std(sample_X):.6f}"
    )

    if not np.isfinite(
        sample_X
    ).all():

        raise RuntimeError(
            "Scaled batch contains "
            "non-finite values."
        )

    del sample_X
    del sample_y

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    model = build_model()

    model.summary()

    parameter_count = (
        model.count_params()
    )

    print()
    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    best_model_path = (
        OUTPUT_DIR
        / "best_model.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = (
        time.time() - start_time
    )

    print()
    print(
        f"Training time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_df.to_csv(
        OUTPUT_DIR
        / "training_history.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION EVALUATION")
    print("=" * 80)

    val_y_true = np.asarray(
        validation_generator.y,
        dtype=np.int8,
    )

    val_prob = model.predict(
        validation_generator,
        verbose=1,
    )

    val_y_pred = np.argmax(
        val_prob,
        axis=1,
    ).astype(np.int8)

    val_metrics = calculate_metrics(
        val_y_true,
        val_y_pred,
    )

    print()

    for name, value in val_metrics.items():

        print(
            f"{name:<25}: "
            f"{value:.6f}"
        )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)

    test_y_true = np.asarray(
        test_generator.y,
        dtype=np.int8,
    )

    test_prob = model.predict(
        test_generator,
        verbose=1,
    )

    test_y_pred = np.argmax(
        test_prob,
        axis=1,
    ).astype(np.int8)

    test_metrics = calculate_metrics(
        test_y_true,
        test_y_pred,
    )

    print()

    for name, value in test_metrics.items():

        print(
            f"{name:<25}: "
            f"{value:.6f}"
        )

    # -------------------------------------------------------------------------
    # Best epoch
    # -------------------------------------------------------------------------

    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_index = (
        best_epoch - 1
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    results = {

        "experiment": (
            "MF 2Y train-only standardization"
        ),

        "asset_class": "mutual_funds",

        "horizon": "2y",

        "lookback": LOOKBACK,

        "n_features": N_FEATURES,

        "scaling": "standardization",

        "scaler_fit_split": "train_only",

        "lstm_units": LSTM_UNITS,

        "dense_units": DENSE_UNITS,

        "dropout": DROPOUT_RATE,

        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,

        "max_epochs": MAX_EPOCHS,

        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),

        "parameter_count": int(
            parameter_count
        ),

        "epochs_trained": len(
            history.history["loss"]
        ),

        "best_epoch": best_epoch,

        "best_val_loss": float(
            history.history["val_loss"][
                best_index
            ]
        ),

        "best_val_accuracy": float(
            history.history["val_accuracy"][
                best_index
            ]
        ),

        "train_accuracy_at_best_epoch": float(
            history.history["accuracy"][
                best_index
            ]
        ),

        "train_loss_at_best_epoch": float(
            history.history["loss"][
                best_index
            ]
        ),

        "training_seconds": float(
            training_seconds
        ),

        "validation": val_metrics,

        "test": test_metrics,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    model.save(
        OUTPUT_DIR
        / "final_model.keras"
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "MF 2Y SCALED LSTM DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        f"Best epoch    : {best_epoch}"
    )

    print(
        f"Best val loss : "
        f"{results['best_val_loss']:.6f}"
    )

    print(
        f"Best val acc  : "
        f"{results['best_val_accuracy']:.6f}"
    )

    print()
    print("VALIDATION")

    for name, value in val_metrics.items():

        print(
            f"  {name:<22}: "
            f"{value:.6f}"
        )

    print()
    print("TEST")

    for name, value in test_metrics.items():

        print(
            f"  {name:<22}: "
            f"{value:.6f}"
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )

    del train_generator
    del validation_generator
    del test_generator
    del model

    gc.collect()

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()