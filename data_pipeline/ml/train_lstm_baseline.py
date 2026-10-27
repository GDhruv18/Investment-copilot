from pathlib import Path
import json
import gc
import time
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

#python data_pipeline\ml\train_lstm_baseline.py
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
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "lstm_results02"
)


# =============================================================================
# LOCKED TRAINING POLICY
# =============================================================================

LOOKBACKS = {
    "6m": 126,
    "1y": 252,
    "2y": 252,
}

ASSET_CONFIG = {
    "stocks": {
        "prefix": "stock",
    },

    "etfs": {
        "prefix": "etf",
    },

    "mutual_funds": {
        "prefix": "mutual_fund",
    },
}

HORIZONS = ["2y"]


# =============================================================================
# LSTM HYPERPARAMETERS
# =============================================================================

LSTM_UNITS = 64

DENSE_UNITS = 32

DROPOUT_RATE = 0.20

LEARNING_RATE = 0.001

BATCH_SIZE = 64

MAX_EPOCHS = 50

EARLY_STOPPING_PATIENCE = 7


# =============================================================================
# MODEL
# =============================================================================

def build_lstm_model(
    lookback,
    n_features=27,
):
    """
    Initial locked LSTM baseline.

    Input:
        (lookback, 27)

    Architecture:
        LSTM(64)
        Dropout(0.20)
        Dense(32, ReLU)
        Dropout(0.20)
        Dense(3, Softmax)
    """

    inputs = tf.keras.Input(
        shape=(
            lookback,
            n_features,
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
        name="investment_copilot_lstm_baseline",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    return model


# =============================================================================
# MEMORY-MAPPED DATA GENERATOR
# =============================================================================

class NPYSequenceGenerator(
    tf.keras.utils.Sequence
):
    """
    Batch generator for memory-mapped .npy sequence data.

    X is never loaded completely into RAM.

    Each __getitem__ loads only the requested batch.
    """

    def __init__(
        self,
        X_path,
        y_path,
        batch_size,
        shuffle,
    ):

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
                f"X/y length mismatch: "
                f"{len(self.X)} vs {len(self.y)}"
            )

        self.batch_size = batch_size

        self.shuffle = shuffle

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

        batch_indices = self.indices[
            start:end
        ]

        # -------------------------------------------------------------
        # Advanced indexing on a memory-mapped array creates only this
        # batch in RAM.
        # -------------------------------------------------------------

        X_batch = np.asarray(
            self.X[batch_indices],
            dtype=np.float32,
        )

        y_batch = np.asarray(
            self.y[batch_indices],
            dtype=np.int8,
        )

        return (
            X_batch,
            y_batch,
        )

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_paths(
    asset_class,
    horizon,
    split,
):
    prefix = ASSET_CONFIG[
        asset_class
    ]["prefix"]

    directory = (
        SEQUENCE_DIR
        / asset_class
        / horizon
    )

    X_path = (
        directory
        / f"{asset_class}_{horizon}_{split}_X.npy"
    )

    y_path = (
        directory
        / f"{asset_class}_{horizon}_{split}_y.npy"
    )

    metadata_path = (
        directory
        / f"{asset_class}_{horizon}_{split}_metadata.parquet"
    )

    return (
        X_path,
        y_path,
        metadata_path,
    )


# =============================================================================
# VALIDATE SEQUENCE FILES
# =============================================================================

def validate_sequence_files(
    asset_class,
    horizon,
):
    """
    Verify the sequence files before training.
    """

    paths = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        X_path, y_path, metadata_path = (
            get_paths(
                asset_class,
                horizon,
                split,
            )
        )

        for path in [
            X_path,
            y_path,
            metadata_path,
        ]:

            if not path.exists():
                raise FileNotFoundError(
                    f"Required sequence file not found:\n"
                    f"{path}"
                )

        paths[split] = {
            "X": X_path,
            "y": y_path,
            "metadata": metadata_path,
        }

    # -------------------------------------------------------------------------
    # Inspect X shapes without loading the complete arrays.
    # -------------------------------------------------------------------------

    for split in paths:

        X = np.load(
            paths[split]["X"],
            mmap_mode="r",
            allow_pickle=False,
        )

        y = np.load(
            paths[split]["y"],
            mmap_mode="r",
            allow_pickle=False,
        )

        metadata = pd.read_parquet(
            paths[split]["metadata"]
        )

        expected_lookback = LOOKBACKS[
            horizon
        ]

        if X.ndim != 3:
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"X must be 3D."
            )

        if X.shape[1] != expected_lookback:
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"lookback mismatch."
            )

        if X.shape[2] != 27:
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"expected 27 features."
            )

        if len(X) != len(y):
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"X/y mismatch."
            )

        if len(X) != len(metadata):
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"X/metadata mismatch."
            )

        if X.dtype != np.float32:
            raise ValueError(
                f"{asset_class}/{horizon}/{split}: "
                f"X must be float32."
            )

        print(
            f"    {split:<12} "
            f"X={X.shape} "
            f"y={y.shape}"
        )

        del X
        del y
        del metadata

    gc.collect()

    print(
        "    Sequence files: PASS"
    )

    return paths


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
    }


# =============================================================================
# PREDICTIONS
# =============================================================================

def generate_predictions(
    model,
    generator,
):
    """
    Generate predictions for one complete split.
    """

    y_true = np.asarray(
        generator.y,
        dtype=np.int8,
    )

    probabilities = model.predict(
        generator,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int8)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction length mismatch: "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    return (
        y_true,
        y_pred,
        probabilities,
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_training_history(
    history,
    output_path,
):

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1
        ),
    )

    history_df.to_csv(
        output_path,
        index=False,
    )


def save_confusion_matrix(
    matrix,
    output_path,
):

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Weak",
            "Neutral",
            "Attractive",
        ],
        columns=[
            "Weak",
            "Neutral",
            "Attractive",
        ],
    )

    matrix_df.to_csv(
        output_path
    )


def save_classification_report(
    y_true,
    y_pred,
    output_path,
):

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
        target_names=[
            "Weak",
            "Neutral",
            "Attractive",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        output_path
    )


# =============================================================================
# TRAIN ONE MODEL
# =============================================================================

def train_one_model(
    asset_class,
    horizon,
):
    print()
    print()
    print("#" * 80)
    print(
        f"LSTM BASELINE: "
        f"{asset_class.upper()} | "
        f"{horizon.upper()}"
    )
    print("#" * 80)

    lookback = LOOKBACKS[
        horizon
    ]

    output_dir = (
        RESULTS_DIR
        / asset_class
        / horizon
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------------------

    print()
    print("SEQUENCE FILES")
    print("-" * 80)

    paths = validate_sequence_files(
        asset_class,
        horizon,
    )

    # -------------------------------------------------------------------------
    # Generators
    # -------------------------------------------------------------------------

    print()
    print("CREATING DATA GENERATORS")
    print("-" * 80)

    train_generator = NPYSequenceGenerator(
        paths["train"]["X"],
        paths["train"]["y"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_generator = NPYSequenceGenerator(
        paths["validation"]["X"],
        paths["validation"]["y"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_generator = NPYSequenceGenerator(
        paths["test"]["X"],
        paths["test"]["y"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"    Train batches      : "
        f"{len(train_generator):,}"
    )

    print(
        f"    Validation batches : "
        f"{len(validation_generator):,}"
    )

    print(
        f"    Test batches       : "
        f"{len(test_generator):,}"
    )

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    print()
    print("MODEL")
    print("-" * 80)

    model = build_lstm_model(
        lookback=lookback,
        n_features=27,
    )

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
        output_dir
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
    print("TRAINING")
    print("-" * 80)

    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Training time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    save_training_history(
        history,
        output_dir
        / "training_history.csv",
    )

    # -------------------------------------------------------------------------
    # Validation evaluation
    #
    # This is the evaluation used for model-development decisions.
    # -------------------------------------------------------------------------

    print()
    print("VALIDATION EVALUATION")
    print("-" * 80)

    val_y_true, val_y_pred, val_prob = (
        generate_predictions(
            model,
            validation_generator,
        )
    )

    val_metrics = calculate_metrics(
        val_y_true,
        val_y_pred,
    )

    for name, value in val_metrics.items():

        print(
            f"    {name:<20}: "
            f"{value:.6f}"
        )

    val_matrix = confusion_matrix(
        val_y_true,
        val_y_pred,
        labels=[
            0,
            1,
            2,
        ],
    )

    print()
    print("Validation confusion matrix:")
    print(val_matrix)

    save_confusion_matrix(
        val_matrix,
        output_dir
        / "validation_confusion_matrix.csv",
    )

    save_classification_report(
        val_y_true,
        val_y_pred,
        output_dir
        / "validation_classification_report.csv",
    )

    # -------------------------------------------------------------------------
    # Test evaluation
    #
    # IMPORTANT:
    #
    # This is recorded as the baseline test result.
    # It must NOT be used for changing architecture/hyperparameters.
    # -------------------------------------------------------------------------

    print()
    print("TEST EVALUATION")
    print("-" * 80)

    test_y_true, test_y_pred, test_prob = (
        generate_predictions(
            model,
            test_generator,
        )
    )

    test_metrics = calculate_metrics(
        test_y_true,
        test_y_pred,
    )

    for name, value in test_metrics.items():

        print(
            f"    {name:<20}: "
            f"{value:.6f}"
        )

    test_matrix = confusion_matrix(
        test_y_true,
        test_y_pred,
        labels=[
            0,
            1,
            2,
        ],
    )

    print()
    print("Test confusion matrix:")
    print(test_matrix)

    save_confusion_matrix(
        test_matrix,
        output_dir
        / "test_confusion_matrix.csv",
    )

    save_classification_report(
        test_y_true,
        test_y_pred,
        output_dir
        / "test_classification_report.csv",
    )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------

    best_epoch = int(
        np.argmin(
            history.history[
                "val_loss"
            ]
        )
        + 1
    )

    results = {
        "asset_class": asset_class,
        "horizon": horizon,
        "lookback": lookback,
        "n_features": 27,

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

        "training_seconds": float(
            training_seconds
        ),

        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    final_model_path = (
        output_dir
        / "final_model.keras"
    )

    model.save(
        final_model_path
    )

    print()
    print(
        f"Model saved: "
        f"{final_model_path}"
    )

    print(
        f"Best model saved: "
        f"{best_model_path}"
    )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    del train_generator
    del validation_generator
    del test_generator
    del model

    del val_y_true
    del val_y_pred
    del val_prob

    del test_y_true
    del test_y_pred
    del test_prob

    gc.collect()

    tf.keras.backend.clear_session()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("INVESTMENT COPILOT - LSTM BASELINE TRAINING")
    print("=" * 80)

    print()
    print("LOCKED ARCHITECTURE")
    print("-------------------")
    print("LSTM          : 64 units")
    print("Dropout       : 0.20")
    print("Dense         : 32 units, ReLU")
    print("Dropout       : 0.20")
    print("Output        : 3 classes, Softmax")

    print()
    print("LOCKED TRAINING POLICY")
    print("----------------------")
    print("Optimizer     : Adam")
    print("Learning rate : 0.001")
    print("Loss          : Sparse Categorical Crossentropy")
    print("Batch size    : 64")
    print("Max epochs    : 50")
    print("Early stopping: 7 epochs")

    print()
    print("DATA POLICY")
    print("-----------")
    print("Features      : 27")
    print("Scaling       : NOT performed")
    print("Target        : Existing target")
    print("Split         : Existing temporal split")
    print("Test usage    : Evaluation only")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    total_models = (
        len(ASSET_CONFIG)
        * len(HORIZONS)
    )

    completed = 0

    for asset_class in [
        "stocks",
        "etfs",
        "mutual_funds",
    ]:

        for horizon in HORIZONS:

            result = train_one_model(
                asset_class,
                horizon,
            )

            all_results.append(
                result
            )

            completed += 1

            print()
            print(
                f"Models completed: "
                f"{completed}/{total_models}"
            )

    # =========================================================================
    # Final summary
    # =========================================================================

    summary_rows = []

    for result in all_results:

        summary_rows.append(
            {
                "asset_class": result[
                    "asset_class"
                ],

                "horizon": result[
                    "horizon"
                ],

                "lookback": result[
                    "lookback"
                ],

                "parameters": result[
                    "parameter_count"
                ],

                "best_epoch": result[
                    "best_epoch"
                ],

                "val_accuracy": result[
                    "validation"
                ]["accuracy"],

                "val_balanced_accuracy": result[
                    "validation"
                ]["balanced_accuracy"],

                "val_macro_f1": result[
                    "validation"
                ]["macro_f1"],

                "test_accuracy": result[
                    "test"
                ]["accuracy"],

                "test_balanced_accuracy": result[
                    "test"
                ]["balanced_accuracy"],

                "test_macro_f1": result[
                    "test"
                ]["macro_f1"],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "lstm_baseline_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print()
    print("=" * 80)
    print("LSTM BASELINE TRAINING COMPLETE")
    print("=" * 80)

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary saved to:\n"
        f"{summary_path}"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Compare LSTM validation performance "
        "against the Logistic Regression baseline."
    )


if __name__ == "__main__":
    main()