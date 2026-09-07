from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FOLDER = (
    PROJECT_ROOT
    / "data_pipeline"
    / "ml"
    / "output"
    / "splits"
)

TRAIN_FILE = SPLIT_FOLDER / "stock_train_30d.csv"
VALIDATION_FILE = SPLIT_FOLDER / "stock_validation_30d.csv"


# ============================================================
# TARGET
# ============================================================

TARGET = "target_30d"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA FOR MEAN BASELINE")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")


# ============================================================
# CALCULATE TRAINING MEAN
# ============================================================

training_mean = train_df[TARGET].mean()

print("\n" + "=" * 70)
print("MEAN BASELINE")
print("=" * 70)

print(f"Training mean 30D return: {training_mean:.4%}")


# ============================================================
# CREATE PREDICTIONS
# ============================================================

y_validation = validation_df[TARGET]

predictions = np.full(
    len(validation_df),
    training_mean
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_validation,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        predictions
    )
)

r2 = r2_score(
    y_validation,
    predictions
)

directional_accuracy = (
    np.sign(y_validation)
    == np.sign(predictions)
).mean()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"MAE:                 {mae:.6f}")
print(f"RMSE:                {rmse:.6f}")
print(f"R²:                  {r2:.6f}")
print(f"Directional Accuracy:{directional_accuracy:.2%}")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("WHY THIS BASELINE MATTERS")
print("=" * 70)

print(
    "\nThe baseline predicts the same training-average return "
    "for every validation observation."
)

print(
    "XGBoost must outperform this baseline to demonstrate "
    "useful predictive value."
)

print("\nMean baseline evaluation completed successfully.")