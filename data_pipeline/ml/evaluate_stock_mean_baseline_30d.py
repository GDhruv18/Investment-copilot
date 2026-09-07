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
# COLUMNS
# ============================================================

TICKER = "ticker"
TARGET = "target_30d"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA FOR STOCK-SPECIFIC MEAN BASELINE")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Training stocks: {train_df[TICKER].nunique()}")
print(f"Validation stocks: {validation_df[TICKER].nunique()}")


# ============================================================
# CALCULATE STOCK-SPECIFIC TRAINING MEANS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING STOCK-SPECIFIC MEANS")
print("=" * 70)

stock_means = (
    train_df
    .groupby(TICKER)[TARGET]
    .mean()
)

global_mean = train_df[TARGET].mean()

print(f"Global training mean: {global_mean:.4%}")
print(f"Stock-specific means calculated: {len(stock_means)}")


# ============================================================
# CREATE VALIDATION PREDICTIONS
# ============================================================

validation_df["predicted_30d"] = (
    validation_df[TICKER]
    .map(stock_means)
)

# Safety fallback:
# If a stock somehow appears in validation but not training,
# use the global training mean.
validation_df["predicted_30d"] = (
    validation_df["predicted_30d"]
    .fillna(global_mean)
)

y_validation = validation_df[TARGET]
predictions = validation_df["predicted_30d"]


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
print("STOCK-SPECIFIC MEAN BASELINE RESULTS")
print("=" * 70)

print(f"MAE:                 {mae:.6f}")
print(f"RMSE:                {rmse:.6f}")
print(f"R²:                  {r2:.6f}")
print(f"Directional Accuracy:{directional_accuracy:.2%}")


# ============================================================
# STOCK MEAN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("STOCK MEAN STATISTICS")
print("=" * 70)

print(f"Lowest stock mean:  {stock_means.min():.4%}")
print(f"Highest stock mean: {stock_means.max():.4%}")
print(f"Average stock mean: {stock_means.mean():.4%}")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print("""
Global Mean Baseline:
    Every stock receives the same predicted return.

Stock-Specific Mean Baseline:
    Each stock receives its own historical average return.

XGBoost:
    Uses the engineered features to predict the return.
""")

print("Stock-specific mean baseline evaluation completed successfully.")