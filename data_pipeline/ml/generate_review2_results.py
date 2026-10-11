import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

ASSET_CLASS = "mutual_funds"
HORIZON = "2y"

BASE_DIR = os.path.join("data", "ml")

RESULT_DIR = os.path.join(
    BASE_DIR,
    "lstm_results",
    ASSET_CLASS,
    HORIZON
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "sequences",
    ASSET_CLASS,
    HORIZON
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "review2",
    ASSET_CLASS,
    HORIZON
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Actual filenames used by the project
TRAIN_X_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_train_X.npy"
)

TRAIN_Y_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_train_y.npy"
)

VAL_X_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_validation_X.npy"
)

VAL_Y_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_validation_y.npy"
)

TEST_X_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_test_X.npy"
)

TEST_Y_PATH = os.path.join(
    SEQUENCE_DIR,
    "mutual_funds_2y_test_y.npy"
)

MODEL_PATH = os.path.join(
    RESULT_DIR,
    "best_model.keras"
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

METRICS_PATH = os.path.join(
    RESULT_DIR,
    "metrics.json"
)

# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    VAL_X_PATH,
    VAL_Y_PATH,
    TEST_X_PATH,
    TEST_Y_PATH,
]

print("=" * 70)
print("REVIEW 2 RESULT GENERATION")
print("=" * 70)

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )

print("\nAll required files found.")

# ============================================================
# LOAD TENSORFLOW MODEL
# ============================================================

print("\nLoading trained LSTM model...")

import tensorflow as tf

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("\nMODEL SUMMARY")
print("-" * 70)

model.summary()

# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

print("\nLoading training history...")

history = pd.read_csv(HISTORY_PATH)

print("\nTraining history:")
print(history.to_string(index=False))

# ============================================================
# DETERMINE COLUMN NAMES
# ============================================================

# The training history should normally contain:
# loss
# accuracy
# val_loss
# val_accuracy

required_history_columns = [
    "loss",
    "accuracy",
    "val_loss",
    "val_accuracy",
]

missing_history_columns = [
    col for col in required_history_columns
    if col not in history.columns
]

if missing_history_columns:
    raise ValueError(
        f"Missing history columns: {missing_history_columns}\n"
        f"Available columns: {list(history.columns)}"
    )

# Add epoch number if not already present
if "epoch" not in history.columns:
    history.insert(
        0,
        "epoch",
        np.arange(1, len(history) + 1)
    )

# ============================================================
# SAVE CLEAN HISTORY TABLE
# ============================================================

history_output = os.path.join(
    OUTPUT_DIR,
    "training_history_for_review.csv"
)

history.to_csv(
    history_output,
    index=False
)

print(
    f"\nTraining history saved to:\n{history_output}"
)

# ============================================================
# FIND BEST EPOCH
# ============================================================

best_epoch_index = history["val_loss"].idxmin()

best_epoch_row = history.loc[best_epoch_index]

best_epoch = int(best_epoch_row["epoch"])

print("\nBEST EPOCH")
print("-" * 70)
print(f"Best epoch: {best_epoch}")
print(f"Training loss: {best_epoch_row['loss']:.6f}")
print(f"Validation loss: {best_epoch_row['val_loss']:.6f}")
print(f"Training accuracy: {best_epoch_row['accuracy']:.6f}")
print(f"Validation accuracy: {best_epoch_row['val_accuracy']:.6f}")

# ============================================================
# GRAPH 1 — LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Sparse Categorical Crossentropy Loss")
plt.title("Mutual Fund 2Y LSTM - Training and Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nLoss graph saved to:\n{loss_path}"
)

# ============================================================
# GRAPH 2 — ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mutual Fund 2Y LSTM - Training and Validation Accuracy")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nAccuracy graph saved to:\n{accuracy_path}"
)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation sequences...")

X_val = np.load(
    VAL_X_PATH,
    mmap_mode="r"
)

y_val = np.load(
    VAL_Y_PATH,
    mmap_mode="r"
)

print(f"Validation X shape: {X_val.shape}")
print(f"Validation y shape: {y_val.shape}")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test sequences...")

X_test = np.load(
    TEST_X_PATH,
    mmap_mode="r"
)

y_test = np.load(
    TEST_Y_PATH,
    mmap_mode="r"
)

print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {y_test.shape}")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

val_probabilities = model.predict(
    X_val,
    batch_size=64,
    verbose=1
)

val_predictions = np.argmax(
    val_probabilities,
    axis=1
)

print("\nGenerating test predictions...")

test_probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Weak",
    "Neutral",
    "Attractive",
]

# ============================================================
# VALIDATION METRICS
# ============================================================

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    val_predictions
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

# ============================================================
# PRINT METRICS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print("\nVALIDATION")
print("-" * 70)

print(
    f"Accuracy:           {val_accuracy:.4f} "
    f"({val_accuracy * 100:.2f}%)"
)

print(
    f"Balanced Accuracy:  {val_balanced_accuracy:.4f} "
    f"({val_balanced_accuracy * 100:.2f}%)"
)

print(
    f"Macro F1:            {val_macro_f1:.4f} "
    f"({val_macro_f1 * 100:.2f}%)"
)

print("\nTEST")
print("-" * 70)

print(
    f"Accuracy:           {test_accuracy:.4f} "
    f"({test_accuracy * 100:.2f}%)"
)

print(
    f"Balanced Accuracy:  {test_balanced_accuracy:.4f} "
    f"({test_balanced_accuracy * 100:.2f}%)"
)

print(
    f"Macro F1:            {test_macro_f1:.4f} "
    f"({test_macro_f1 * 100:.2f}%)"
)

# ============================================================
# CONFUSION MATRIX — VALIDATION
# ============================================================

val_cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=[0, 1, 2]
)

print("\n")
print("=" * 70)
print("VALIDATION CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        val_cm,
        index=[
            "Actual Weak",
            "Actual Neutral",
            "Actual Attractive",
        ],
        columns=[
            "Pred Weak",
            "Pred Neutral",
            "Pred Attractive",
        ],
    )
)

# ============================================================
# VALIDATION CONFUSION MATRIX GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=val_cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Mutual Fund 2Y LSTM - Validation Confusion Matrix"
)

plt.tight_layout()

val_cm_path = os.path.join(
    OUTPUT_DIR,
    "validation_confusion_matrix.png"
)

plt.savefig(
    val_cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nValidation confusion matrix saved to:\n{val_cm_path}"
)

# ============================================================
# CONFUSION MATRIX — TEST
# ============================================================

test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1, 2]
)

print("\n")
print("=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        test_cm,
        index=[
            "Actual Weak",
            "Actual Neutral",
            "Actual Attractive",
        ],
        columns=[
            "Pred Weak",
            "Pred Neutral",
            "Pred Attractive",
        ],
    )
)

# ============================================================
# TEST CONFUSION MATRIX GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Mutual Fund 2Y LSTM - Test Confusion Matrix"
)

plt.tight_layout()

test_cm_path = os.path.join(
    OUTPUT_DIR,
    "test_confusion_matrix.png"
)

plt.savefig(
    test_cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nTest confusion matrix saved to:\n{test_cm_path}"
)

# ============================================================
# CLASSIFICATION REPORT — VALIDATION
# ============================================================

val_report = classification_report(
    y_val,
    val_predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4
)

print("\n")
print("=" * 70)
print("VALIDATION CLASSIFICATION REPORT")
print("=" * 70)

print(val_report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "validation_classification_report.txt"
    ),
    "w"
) as f:
    f.write(val_report)

# ============================================================
# CLASSIFICATION REPORT — TEST
# ============================================================

test_report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4
)

print("\n")
print("=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(test_report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "test_classification_report.txt"
    ),
    "w"
) as f:
    f.write(test_report)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

actual_counts = pd.Series(
    y_test
).value_counts().sort_index()

predicted_counts = pd.Series(
    test_predictions
).value_counts().sort_index()

distribution = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Actual Count": [
        actual_counts.get(i, 0)
        for i in range(3)
    ],
    "Predicted Count": [
        predicted_counts.get(i, 0)
        for i in range(3)
    ],
})

distribution["Actual %"] = (
    distribution["Actual Count"]
    / len(y_test)
    * 100
)

distribution["Predicted %"] = (
    distribution["Predicted Count"]
    / len(y_test)
    * 100
)

print(
    distribution.to_string(index=False)
)

distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    ),
    index=False
)

# ============================================================
# FINAL REVIEW SUMMARY
# ============================================================

summary = {
    "project": "Investment Copilot",
    "asset_class": "Mutual Funds",
    "horizon": "2Y",
    "model": "LSTM",
    "input_shape": [252, 27],
    "architecture": [
        "LSTM(64)",
        "Dropout(0.20)",
        "Dense(32, ReLU)",
        "Dropout(0.20)",
        "Dense(3, Softmax)",
    ],
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "Sparse Categorical Crossentropy",
    "batch_size": 64,
    "best_epoch": best_epoch,
    "validation": {
        "accuracy": float(val_accuracy),
        "balanced_accuracy": float(val_balanced_accuracy),
        "macro_f1": float(val_macro_f1),
    },
    "test": {
        "accuracy": float(test_accuracy),
        "balanced_accuracy": float(test_balanced_accuracy),
        "macro_f1": float(test_macro_f1),
    },
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "review2_summary.json"
)

with open(
    summary_path,
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )

# ============================================================
# FINAL OUTPUT LIST
# ============================================================

print("\n")
print("=" * 70)
print("REVIEW 2 FILES GENERATED")
print("=" * 70)

print(f"""
Output directory:

{OUTPUT_DIR}

Generated files:

1. loss_curve.png
2. accuracy_curve.png
3. validation_confusion_matrix.png
4. test_confusion_matrix.png
5. validation_classification_report.txt
6. test_classification_report.txt
7. class_distribution.csv
8. training_history_for_review.csv
9. review2_summary.json

These are the main statistical results required for Review 2.
""")

print("=" * 70)
print("DONE")
print("=" * 70)