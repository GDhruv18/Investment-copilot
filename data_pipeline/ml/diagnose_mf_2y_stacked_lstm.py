from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQ_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "sequences"
    / "mutual_funds"
    / "2y"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOCKED DATA SETTINGS
# ============================================================

LOOKBACK = 252
N_FEATURES = 27
N_CLASSES = 3

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 7

LEARNING_RATE = 0.001

DROPOUT = 0.20


# ============================================================
# MODEL SETTINGS
# ============================================================

FIRST_LSTM_UNITS = 64
SECOND_LSTM_UNITS = 32
DENSE_UNITS = 32


# ============================================================
# FILES
# ============================================================

X_TRAIN_PATH = SEQ_DIR / "mutual_funds_2y_train_X.npy"
Y_TRAIN_PATH = SEQ_DIR / "mutual_funds_2y_train_y.npy"

X_VAL_PATH = SEQ_DIR / "mutual_funds_2y_validation_X.npy"
Y_VAL_PATH = SEQ_DIR / "mutual_funds_2y_validation_y.npy"

X_TEST_PATH = SEQ_DIR / "mutual_funds_2y_test_X.npy"
Y_TEST_PATH = SEQ_DIR / "mutual_funds_2y_test_y.npy"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 100)
    print("MUTUAL FUNDS 2Y - STACKED LSTM DIAGNOSTIC")
    print("=" * 100)

    X_train = np.load(
        X_TRAIN_PATH,
        mmap_mode="r"
    )

    y_train = np.load(
        Y_TRAIN_PATH,
        mmap_mode="r"
    )

    X_val = np.load(
        X_VAL_PATH,
        mmap_mode="r"
    )

    y_val = np.load(
        Y_VAL_PATH,
        mmap_mode="r"
    )

    X_test = np.load(
        X_TEST_PATH,
        mmap_mode="r"
    )

    y_test = np.load(
        Y_TEST_PATH,
        mmap_mode="r"
    )

    print(f"\nTrain X:      {X_train.shape}")
    print(f"Train y:      {y_train.shape}")

    print(f"Validation X: {X_val.shape}")
    print(f"Validation y: {y_val.shape}")

    print(f"Test X:       {X_test.shape}")
    print(f"Test y:       {y_test.shape}")

    assert X_train.shape[1:] == (
        LOOKBACK,
        N_FEATURES,
    )

    assert X_val.shape[1:] == (
        LOOKBACK,
        N_FEATURES,
    )

    assert X_test.shape[1:] == (
        LOOKBACK,
        N_FEATURES,
    )

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# GENERATOR
# ============================================================

class NPYSequenceGenerator(
    keras.utils.Sequence
):

    def __init__(
        self,
        X,
        y,
        batch_size=64,
        shuffle=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.y)
        )

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(
                len(self.indices)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        start = (
            index
            * self.batch_size
        )

        end = min(
            (index + 1)
            * self.batch_size,
            len(self.indices)
        )

        batch_indices = self.indices[
            start:end
        ]

        X_batch = np.asarray(
            self.X[batch_indices],
            dtype=np.float32,
        )

        y_batch = np.asarray(
            self.y[batch_indices],
            dtype=np.int32,
        )

        return X_batch, y_batch

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(
                self.indices
            )


# ============================================================
# BUILD STACKED LSTM
# ============================================================

def build_model():

    model = keras.Sequential(
        [

            layers.Input(
                shape=(
                    LOOKBACK,
                    N_FEATURES,
                )
            ),

            # First recurrent layer
            layers.LSTM(
                FIRST_LSTM_UNITS,
                return_sequences=True,
            ),

            layers.Dropout(
                DROPOUT
            ),

            # Second recurrent layer
            layers.LSTM(
                SECOND_LSTM_UNITS,
                return_sequences=False,
            ),

            layers.Dropout(
                DROPOUT
            ),

            # Dense representation
            layers.Dense(
                DENSE_UNITS,
                activation="relu",
            ),

            layers.Dropout(
                DROPOUT
            ),

            # 3-class output
            layers.Dense(
                N_CLASSES,
                activation="softmax",
            ),
        ]
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    model,
    generator,
):

    y_true = []
    y_pred = []

    for i in range(
        len(generator)
    ):

        X_batch, y_batch = generator[i]

        probabilities = model.predict(
            X_batch,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        y_true.extend(
            y_batch
        )

        y_pred.extend(
            predictions
        )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

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
    )

    return (
        accuracy,
        balanced_accuracy,
        macro_f1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    train_generator = (
        NPYSequenceGenerator(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    validation_generator = (
        NPYSequenceGenerator(
            X_val,
            y_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    test_generator = (
        NPYSequenceGenerator(
            X_test,
            y_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model()

    print("\n" + "=" * 100)
    print("STACKED LSTM MODEL")
    print("=" * 100)

    model.summary()

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 100)
    print("TRAINING")
    print("=" * 100)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[
            early_stopping
        ],
        verbose=1,
    )

    history_df = pd.DataFrame(
        history.history
    )

    best_epoch = (
        int(
            history_df[
                "val_loss"
            ].idxmin()
        )
        + 1
    )

    best_row = history_df.iloc[
        best_epoch - 1
    ]

    # ========================================================
    # TRAINING DIAGNOSTICS
    # ========================================================

    print("\n" + "=" * 100)
    print("BEST EPOCH")
    print("=" * 100)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Train loss: "
        f"{best_row['loss']:.6f}"
    )

    print(
        f"Validation loss: "
        f"{best_row['val_loss']:.6f}"
    )

    print(
        f"Train accuracy: "
        f"{best_row['accuracy']:.6f}"
    )

    print(
        f"Validation accuracy: "
        f"{best_row['val_accuracy']:.6f}"
    )

    print(
        f"Loss gap: "
        f"{best_row['val_loss'] - best_row['loss']:.6f}"
    )

    print(
        f"Accuracy gap: "
        f"{best_row['accuracy'] - best_row['val_accuracy']:.6f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)

    (
        val_accuracy,
        val_balanced_accuracy,
        val_macro_f1,
    ) = calculate_metrics(
        model,
        validation_generator,
    )

    print(
        f"Accuracy: "
        f"{val_accuracy:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{val_balanced_accuracy:.6f}"
    )

    print(
        f"Macro F1: "
        f"{val_macro_f1:.6f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 100)
    print("TEST RESULTS")
    print("=" * 100)

    (
        test_accuracy,
        test_balanced_accuracy,
        test_macro_f1,
    ) = calculate_metrics(
        model,
        test_generator,
    )

    print(
        f"Accuracy: "
        f"{test_accuracy:.6f}"
    )

    print(
        f"Balanced accuracy: "
        f"{test_balanced_accuracy:.6f}"
    )

    print(
        f"Macro F1: "
        f"{test_macro_f1:.6f}"
    )

    # ========================================================
    # ORIGINAL MF 2Y LSTM COMPARISON
    # ========================================================

    ORIGINAL_VAL_BALANCED = 0.512147
    ORIGINAL_VAL_F1 = 0.464452

    ORIGINAL_TEST_BALANCED = 0.493740
    ORIGINAL_TEST_F1 = 0.453214

    print("\n" + "=" * 100)
    print("COMPARISON WITH ORIGINAL MF 2Y LSTM")
    print("=" * 100)

    print("\nOriginal single LSTM")
    print("Architecture: LSTM(64)")
    print("Validation balanced accuracy: 51.21%")
    print("Validation Macro F1:          46.45%")
    print("Test balanced accuracy:       49.37%")
    print("Test Macro F1:                45.32%")

    print("\nStacked LSTM")
    print(
        f"Validation balanced accuracy: "
        f"{val_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro F1:          "
        f"{val_macro_f1 * 100:.2f}%"
    )

    print(
        f"Test balanced accuracy:       "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro F1:                "
        f"{test_macro_f1 * 100:.2f}%"
    )

    # ========================================================
    # DELTAS
    # ========================================================

    print("\n" + "=" * 100)
    print("DELTA FROM ORIGINAL LSTM")
    print("=" * 100)

    print(
        f"Validation balanced accuracy: "
        f"{(val_balanced_accuracy - ORIGINAL_VAL_BALANCED) * 100:+.2f} pp"
    )

    print(
        f"Validation Macro F1: "
        f"{(val_macro_f1 - ORIGINAL_VAL_F1) * 100:+.2f} pp"
    )

    print(
        f"Test balanced accuracy: "
        f"{(test_balanced_accuracy - ORIGINAL_TEST_BALANCED) * 100:+.2f} pp"
    )

    print(
        f"Test Macro F1: "
        f"{(test_macro_f1 - ORIGINAL_TEST_F1) * 100:+.2f} pp"
    )

    # ========================================================
    # DECISION
    # ========================================================

    print("\n" + "=" * 100)
    print("DECISION")
    print("=" * 100)

    if (
        val_balanced_accuracy > ORIGINAL_VAL_BALANCED
        and
        val_macro_f1 > ORIGINAL_VAL_F1
        and
        test_balanced_accuracy > ORIGINAL_TEST_BALANCED
        and
        test_macro_f1 > ORIGINAL_TEST_F1
    ):

        print(
            "STACKED LSTM IMPROVES BOTH "
            "VALIDATION AND TEST PERFORMANCE."
        )

    elif (
        val_balanced_accuracy > ORIGINAL_VAL_BALANCED
        and
        val_macro_f1 > ORIGINAL_VAL_F1
    ):

        print(
            "STACKED LSTM IMPROVES VALIDATION "
            "BUT DOES NOT IMPROVE ALL TEST METRICS."
        )

    else:

        print(
            "STACKED LSTM DOES NOT IMPROVE "
            "THE ORIGINAL LSTM."
        )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()