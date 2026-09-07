import os
import random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ML_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_pipeline", "ml", "output")

SPLITS_DIR = os.path.join(
    ML_OUTPUT_DIR, "splits"
)

TRAIN_FILE = os.path.join(
    SPLITS_DIR, "stock_train_30d.csv"
)

VALIDATION_FILE = os.path.join(
    SPLITS_DIR, "stock_validation_30d.csv"
)

TEST_FILE = os.path.join(
    SPLITS_DIR, "stock_test_30d.csv"
)

SCALER_FILE = os.path.join(
    ML_OUTPUT_DIR, "lstm_30d", "lstm_30d_scaler.joblib"
)

MODEL_DIR = os.path.join(
    ML_OUTPUT_DIR, "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR, "lstm_stock_30d.pt"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR, "lstm_stock_30d_training_history.csv"
)

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR, "lstm_stock_30d_validation_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "daily_return",
    "return_30d",
    "return_90d",
    "return_180d",
    "return_1y",
    "volatility_30d",
    "volatility_90d",
    "sma50_ratio",
    "sma200_ratio",
    "drawdown"
]

TARGET_COLUMN = "target_30d"

SEQUENCE_LENGTH = 30


# ============================================================
# HYPERPARAMETERS
# ============================================================

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.20

DENSE_SIZE = 32

BATCH_SIZE = 256
LEARNING_RATE = 0.001

MAX_EPOCHS = 50
PATIENCE = 7

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LSTM 30D STOCK RETURN MODEL")
print("=" * 70)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

for df in [train_df, validation_df, test_df]:
    df["date"] = pd.to_datetime(df["date"])


print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Test rows:       {len(test_df):,}")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading training scaler...")

scaler = joblib.load(SCALER_FILE)


# ============================================================
# SCALE FEATURES
# ============================================================

print("Scaling features...")

train_df = train_df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

validation_df = validation_df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)


train_scaled = train_df.copy()
validation_scaled = validation_df.copy()
test_scaled = test_df.copy()

train_scaled[FEATURE_COLUMNS] = scaler.transform(
    train_df[FEATURE_COLUMNS]
)

validation_scaled[FEATURE_COLUMNS] = scaler.transform(
    validation_df[FEATURE_COLUMNS]
)

test_scaled[FEATURE_COLUMNS] = scaler.transform(
    test_df[FEATURE_COLUMNS]
)


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_training_sequences(df):
    """
    Create sequences entirely within the training dataset.

    For each observation:
        X = previous 30 observations
        y = target_30d corresponding to the final observation
    """

    X_list = []
    y_list = []

    for ticker, group in df.groupby("ticker", sort=False):

        group = group.sort_values("date")

        features = group[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        targets = group[TARGET_COLUMN].to_numpy(
            dtype=np.float32
        )

        for i in range(SEQUENCE_LENGTH - 1, len(group)):

            X_list.append(
                features[i - SEQUENCE_LENGTH + 1:i + 1]
            )

            y_list.append(targets[i])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


def create_context_sequences(history_df, target_df):
    """
    Create validation/test sequences using historical context.

    For each target observation:

        29 observations before it
        + current observation
        = 30 observations

    Historical context can come from the preceding dataset.

    Target values from history are NEVER used as inputs.
    """

    X_list = []
    y_list = []

    history_df = history_df.sort_values(
        ["ticker", "date"]
    )

    target_df = target_df.sort_values(
        ["ticker", "date"]
    )

    for ticker in target_df["ticker"].unique():

        history_group = history_df[
            history_df["ticker"] == ticker
        ].sort_values("date")

        target_group = target_df[
            target_df["ticker"] == ticker
        ].sort_values("date")

        history_features = history_group[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        target_features = target_group[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        target_values = target_group[
            TARGET_COLUMN
        ].to_numpy(dtype=np.float32)

        combined_features = np.concatenate(
            [
                history_features,
                target_features
            ],
            axis=0
        )

        history_length = len(history_features)

        for j in range(len(target_group)):

            current_index = history_length + j

            start_index = (
                current_index - SEQUENCE_LENGTH + 1
            )

            if start_index < 0:
                continue

            sequence = combined_features[
                start_index:current_index + 1
            ]

            if len(sequence) != SEQUENCE_LENGTH:
                continue

            X_list.append(sequence)
            y_list.append(target_values[j])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

print("\nCreating training sequences...")

X_train, y_train = create_training_sequences(
    train_scaled
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# ============================================================
# CREATE VALIDATION SEQUENCES
# ============================================================

print("\nCreating validation sequences...")

X_validation, y_validation = create_context_sequences(
    train_scaled,
    validation_scaled
)

print(f"X_validation shape: {X_validation.shape}")
print(f"y_validation shape: {y_validation.shape}")


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

print("\nCreating test sequences...")

history_for_test = pd.concat(
    [
        train_scaled,
        validation_scaled
    ],
    ignore_index=True
)

X_test, y_test = create_context_sequences(
    history_for_test,
    test_scaled
)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# VALIDATE DATA
# ============================================================

print("\nChecking data quality...")

for name, X, y in [
    ("Training", X_train, y_train),
    ("Validation", X_validation, y_validation),
    ("Test", X_test, y_test)
]:

    print(f"\n{name}:")

    print(
        f"  NaNs in X: {np.isnan(X).sum():,}"
    )

    print(
        f"  NaNs in y: {np.isnan(y).sum():,}"
    )

    print(
        f"  Infinite values in X: "
        f"{np.isinf(X).sum():,}"
    )

    print(
        f"  Infinite values in y: "
        f"{np.isinf(y).sum():,}"
    )


# ============================================================
# CONVERT TO TORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

validation_dataset = TensorDataset(
    torch.from_numpy(X_validation),
    torch.from_numpy(y_validation)
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)


# ============================================================
# DATA LOADERS
# ============================================================

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMReturnPredictor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        dense_size
    ):

        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(
            hidden_size,
            dense_size
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            dense_size,
            1
        )

    def forward(self, x):

        # x shape:
        # (batch, sequence_length, features)

        lstm_output, _ = self.lstm(x)

        # Take the final time step
        last_output = lstm_output[:, -1, :]

        x = self.dropout(last_output)

        x = self.fc1(x)

        x = self.relu(x)

        output = self.fc2(x)

        return output.squeeze(1)


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = LSTMReturnPredictor(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    dense_size=DENSE_SIZE
).to(DEVICE)


print("\nModel:")
print(model)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTrainable parameters: {parameter_count:,}"
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = []

best_validation_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss = train_one_epoch()

    validation_loss = evaluate_loss(
        validation_loader
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss
        }
    )

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {validation_loss:.6f}"
    )

    # Save best model
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": len(FEATURE_COLUMNS),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "dense_size": DENSE_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN
            },
            MODEL_FILE
        )

        print("  -> Best model saved.")

    else:

        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered after "
            f"{PATIENCE} epochs without improvement."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)

print(
    f"\nTraining history saved to:\n"
    f"{HISTORY_FILE}"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(loader):

    predictions = []
    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )

            output = model(X_batch)

            predictions.extend(
                output.cpu().numpy()
            )

            actuals.extend(
                y_batch.numpy()
            )

    return (
        np.asarray(predictions),
        np.asarray(actuals)
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions, validation_actual = predict(
    validation_loader
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions, test_actual = predict(
    test_loader
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    directional_accuracy = np.mean(
        np.sign(actual) == np.sign(predicted)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


validation_metrics = calculate_metrics(
    validation_actual,
    validation_predictions
)

test_metrics = calculate_metrics(
    test_actual,
    test_predictions
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    f"MAE:                 "
    f"{validation_metrics['MAE']:.6f} "
    f"({validation_metrics['MAE'] * 100:.4f}%)"
)

print(
    f"RMSE:                "
    f"{validation_metrics['RMSE']:.6f} "
    f"({validation_metrics['RMSE'] * 100:.4f}%)"
)

print(
    f"R²:                  "
    f"{validation_metrics['R2']:.6f}"
)

print(
    f"Directional Accuracy:"
    f" {validation_metrics['Directional Accuracy'] * 100:.2f}%"
)

print(
    f"Actual Mean Return:  "
    f"{validation_actual.mean() * 100:.4f}%"
)

print(
    f"Predicted Mean Return:"
    f" {validation_predictions.mean() * 100:.4f}%"
)

print(
    f"Actual Std:          "
    f"{validation_actual.std() * 100:.4f}%"
)

print(
    f"Prediction Std:      "
    f"{validation_predictions.std() * 100:.4f}%"
)


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"MAE:                 "
    f"{test_metrics['MAE']:.6f} "
    f"({test_metrics['MAE'] * 100:.4f}%)"
)

print(
    f"RMSE:                "
    f"{test_metrics['RMSE']:.6f} "
    f"({test_metrics['RMSE'] * 100:.4f}%)"
)

print(
    f"R²:                  "
    f"{test_metrics['R2']:.6f}"
)

print(
    f"Directional Accuracy:"
    f" {test_metrics['Directional Accuracy'] * 100:.2f}%"
)

print(
    f"Actual Mean Return:  "
    f"{test_actual.mean() * 100:.4f}%"
)

print(
    f"Predicted Mean Return:"
    f" {test_predictions.mean() * 100:.4f}%"
)

print(
    f"Actual Std:          "
    f"{test_actual.std() * 100:.4f}%"
)

print(
    f"Prediction Std:      "
    f"{test_predictions.std() * 100:.4f}%"
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_prediction_df = validation_df.copy()

# The sequence creation preserves ticker/date order,
# so predictions align with validation rows.
validation_prediction_df = validation_prediction_df[
    ["ticker", "date", TARGET_COLUMN]
].copy()

validation_prediction_df = validation_prediction_df.iloc[
    :len(validation_predictions)
].copy()

validation_prediction_df[
    "predicted_30d_return"
] = validation_predictions

validation_prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"\nValidation predictions saved to:\n"
    f"{PREDICTIONS_FILE}"
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    print("\nGPU memory:")
    print(
        f"Allocated: {allocated:.2f} MB"
    )

    print(
        f"Reserved:  {reserved:.2f} MB"
    )


print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETE")
print("=" * 70)
