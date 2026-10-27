
from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"

MODEL_DIR = ML_ROOT / "selected_models" / "logistic_regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Only models selected using validation results are included.
# Do not add models based on test-set performance.
SELECTED_MODELS = {
    ("stocks", "6m"): (
        "stocks", "stock_6m"
    ),
    ("stocks", "1y"): (
        "stocks", "stock_1y"
    ),
    ("stocks", "2y"): (
        "stocks", "stock_2y"
    ),
    ("mutual_funds", "6m"): (
        "mutual_funds", "mutual_fund_6m"
    ),
    ("mutual_funds", "1y"): (
        "mutual_funds", "mutual_fund_1y"
    ),
}

STOCK_ETF_FEATURES = [
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
    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_sma_200",
    "volume_ratio_20d",
    "volume_ratio_60d",
    "volume_change_20d",
    "range_20d",
    "range_60d",
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]

MF_FEATURES = [
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


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series, asset_class):
    if asset_class in {"stocks", "etfs"}:
        mapping = {
            "Weak": 0,
            "Neutral": 1,
            "Attractive": 2,
        }

        normalized = series.astype(str).map(mapping)

    elif asset_class == "mutual_funds":
        normalized = pd.to_numeric(
            series,
            errors="coerce",
        )

    else:
        raise ValueError(
            f"Unsupported asset class: {asset_class}"
        )

    if normalized.isna().any():
        invalid = (
            series[normalized.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid target values for {asset_class}: "
            f"{invalid}"
        )

    normalized = normalized.astype(int)

    if not set(normalized.unique()).issubset({0, 1, 2}):
        raise ValueError(
            f"Unexpected target classes for {asset_class}"
        )

    if set(normalized.unique()) != {0, 1, 2}:
        raise ValueError(
            f"Training data for {asset_class} does not contain "
            "all three classes. Refusing to save an unexpected model."
        )

    return normalized


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_pipeline():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# TRAIN AND SAVE ONE MODEL
# ============================================================

def train_and_save(asset_class, horizon, dataset_prefix):
    print()
    print("=" * 72)
    print(f"Saving selected model: {asset_class} | {horizon}")
    print("=" * 72)

    train_path = (
        ML_ROOT
        / asset_class
        / f"{dataset_prefix}_train.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    train = pd.read_parquet(train_path)

    if asset_class == "mutual_funds":
        features = MF_FEATURES
    else:
        features = STOCK_ETF_FEATURES

    missing_columns = [
        column
        for column in features + ["target"]
        if column not in train.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {train_path}: "
            f"{missing_columns}"
        )

    X_train = train[features].copy()
    y_train = normalize_target(
        train["target"],
        asset_class,
    )

    if len(X_train) == 0:
        raise ValueError(
            f"Training dataset is empty: {train_path}"
        )

    model = build_pipeline()

    # The imputer, scaler, and classifier are fitted on TRAIN only.
    model.fit(X_train, y_train)

    # Confirm all expected classes are represented.
    fitted_classes = (
        model.named_steps["model"].classes_.tolist()
    )

    if fitted_classes != [0, 1, 2]:
        raise ValueError(
            f"Unexpected fitted classes: {fitted_classes}"
        )

    model_name = f"{asset_class}_{horizon}_logistic"

    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"

    # Save the complete fitted pipeline, including preprocessing.
    joblib.dump(model, model_path)

    metadata = {
        "asset_class": asset_class,
        "horizon": horizon,
        "model_type": "LogisticRegression",
        "pipeline": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()",
            "LogisticRegression(solver='lbfgs', "
            "max_iter=2000, random_state=42)",
        ],
        "features": features,
        "class_mapping": {
            "0": "Weak",
            "1": "Neutral",
            "2": "Attractive",
        },
        "classes": fitted_classes,
        "training_rows": int(len(X_train)),
        "training_start": str(
            pd.to_datetime(train["date"]).min().date()
        ),
        "training_end": str(
            pd.to_datetime(train["date"]).max().date()
        ),
        "training_dataset": str(
            train_path.relative_to(PROJECT_ROOT)
        ),
        "selection_basis": (
            "Previously evaluated validation results; "
            "balanced accuracy primary and Macro-F1 secondary."
        ),
        "test_set_used_for_selection": False,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    # Reload the saved pipeline to verify that it is readable.
    loaded_model = joblib.load(model_path)

    if loaded_model.named_steps["model"].classes_.tolist() != [
        0, 1, 2
    ]:
        raise RuntimeError(
            f"Saved model verification failed: {model_path}"
        )

    print(f"Training rows : {len(X_train):,}")
    print(f"Feature count : {len(features)}")
    print(f"Classes       : {fitted_classes}")
    print(f"Model saved   : {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print("Verification  : PASS")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("SAVE SELECTED LOGISTIC REGRESSION MODELS")
    print("=" * 72)

    print("\nImportant:")
    print("- Existing temporal splits remain unchanged.")
    print("- Only training partitions are used for fitting.")
    print("- Validation/test data is not used in this script.")
    print("- Existing LSTM artifacts remain untouched.")
    print("- No model selection or hyperparameter tuning is performed.")

    for (asset_class, horizon), (
        dataset_asset_class,
        dataset_prefix,
    ) in SELECTED_MODELS.items():

        if asset_class != dataset_asset_class:
            raise ValueError(
                "Model configuration and dataset asset class disagree."
            )

        train_and_save(
            asset_class,
            horizon,
            dataset_prefix,
        )

    print()
    print("=" * 72)
    print("ALL FIVE SELECTED LOGISTIC MODELS SAVED")
    print("=" * 72)
    print(f"Output directory: {MODEL_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
