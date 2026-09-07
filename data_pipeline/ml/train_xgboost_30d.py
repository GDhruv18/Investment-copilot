from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FOLDER = PROJECT_ROOT / "data_pipeline" / "ml" / "output" / "splits"
MODEL_FOLDER = PROJECT_ROOT / "data_pipeline" / "ml" / "output" / "models"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = SPLIT_FOLDER / "stock_train_30d.csv"
VALIDATION_FILE = SPLIT_FOLDER / "stock_validation_30d.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "daily_return",
    "return_30d",
    "return_90d",
    "return_180d",
    "return_1y",
    "volatility_30d",
    "volatility_90d",
    "sma50_ratio",
    "sma200_ratio",
    "drawdown",
]

TARGET = "target_30d"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING 30D STOCK DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_validation = validation_df[FEATURES].copy()
y_validation = validation_df[TARGET].copy()


# Safety check
print("\nChecking missing values...")

print(
    f"Training missing values:   {X_train.isna().sum().sum():,}"
)
print(
    f"Validation missing values: {X_validation.isna().sum().sum():,}"
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST 30D BASELINE")
print("=" * 70)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False,
)


print("Training completed.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

predictions = model.predict(X_validation)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(y_validation, predictions)

rmse = np.sqrt(
    mean_squared_error(y_validation, predictions)
)

r2 = r2_score(y_validation, predictions)

actual_direction = np.sign(y_validation)
predicted_direction = np.sign(predictions)

directional_accuracy = (
    actual_direction == predicted_direction
).mean()


print(f"\nMAE:                 {mae:.6f}")
print(f"RMSE:                {rmse:.6f}")
print(f"R²:                  {r2:.6f}")
print(f"Directional Accuracy:{directional_accuracy:.2%}")


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)

print(f"Actual mean return:      {y_validation.mean():.4%}")
print(f"Predicted mean return:   {predictions.mean():.4%}")

print(f"Actual std deviation:    {y_validation.std():.4%}")
print(f"Prediction std deviation:{predictions.std():.4%}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }
).sort_values(
    by="importance",
    ascending=False,
)

for _, row in importance_df.iterrows():
    print(
        f"{row['feature']:<20} "
        f"{row['importance']:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = MODEL_FOLDER / "xgboost_stock_30d.joblib"

joblib.dump(model, MODEL_FILE)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"Saved to: {MODEL_FILE}")


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

results_df = validation_df[
    ["ticker", "date", TARGET]
].copy()

results_df["predicted_30d"] = predictions

PREDICTION_FILE = MODEL_FOLDER / "xgboost_stock_30d_validation_predictions.csv"

results_df.to_csv(
    PREDICTION_FILE,
    index=False,
)

print(f"Predictions saved to: {PREDICTION_FILE}")

print("\n30D XGBoost baseline completed successfully.")