
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"

RESULTS_DIR = ML_ROOT / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "stocks": (
        ML_ROOT / "stocks" / "stock_2y_train.parquet",
        ML_ROOT / "stocks" / "stock_2y_validation.parquet",
    ),
    "etfs": (
        ML_ROOT / "etfs" / "etf_2y_train.parquet",
        ML_ROOT / "etfs" / "etf_2y_validation.parquet",
    ),
    "mutual_funds": (
        ML_ROOT / "mutual_funds" / "mutual_fund_2y_train.parquet",
        ML_ROOT / "mutual_funds" / "mutual_fund_2y_validation.parquet",
    ),
}

STOCK_ETF_FEATURES = [
    "return_5d", "return_20d", "return_60d",
    "return_120d", "return_252d",
    "volatility_20d", "volatility_60d",
    "volatility_120d", "volatility_252d",
    "drawdown_60d", "drawdown_120d", "drawdown_252d",
    "price_vs_sma_20", "price_vs_sma_50", "price_vs_sma_200",
    "volume_ratio_20d", "volume_ratio_60d", "volume_change_20d",
    "range_20d", "range_60d",
    "return_20d_pct", "return_60d_pct",
    "return_120d_pct", "return_252d_pct",
    "volatility_60d_pct", "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]

MF_FEATURES = [
    "return_5d", "return_20d", "return_60d",
    "return_120d", "return_252d",
    "volatility_20d", "volatility_60d",
    "volatility_120d", "volatility_252d",
    "drawdown_60d", "drawdown_120d", "drawdown_252d",
    "nav_vs_sma_20", "nav_vs_sma_50", "nav_vs_sma_200",
    "nav_range_20d", "nav_range_60d",
    "downside_volatility_60d", "upside_volatility_60d",
    "rolling_sharpe_60d",
    "return_20d_pct", "return_60d_pct",
    "return_120d_pct", "return_252d_pct",
    "volatility_60d_pct", "drawdown_120d_pct",
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
        target = series.astype(str).map(mapping)

    else:
        target = pd.to_numeric(series, errors="coerce")

    if target.isna().any():
        raise ValueError(
            f"Invalid target values in {asset_class}: "
            f"{series[target.isna()].unique().tolist()}"
        )

    target = target.astype(int)

    if not set(target.unique()).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected target classes in {asset_class}")

    return target


# ============================================================
# METRICS
# ============================================================

def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)

    metrics = {
        "accuracy": accuracy_score(y_validation, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation, predictions
        ),
        "macro_f1": f1_score(
            y_validation, predictions,
            average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_validation, predictions,
            average="weighted", zero_division=0
        ),
    }

    return metrics, predictions


def print_results(name, metrics, y_true, predictions):

    print(f"\n{name}")
    print("-" * 65)

    for metric, value in metrics.items():
        print(f"{metric:<22}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1, 2]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=["Weak", "Neutral", "Attractive"],
            zero_division=0,
        )
    )


# ============================================================
# MODEL PIPELINES
# ============================================================

def build_models():

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ============================================================
# TRAIN ONE ASSET CLASS
# ============================================================

def train_one_dataset(asset_class, paths):

    print("\n" + "=" * 80)
    print(f"{asset_class.upper()} | 2Y TREE BASELINES")
    print("=" * 80)

    train = pd.read_parquet(paths[0])
    validation = pd.read_parquet(paths[1])

    features = (
        MF_FEATURES
        if asset_class == "mutual_funds"
        else STOCK_ETF_FEATURES
    )

    X_train = train[features].copy()
    X_validation = validation[features].copy()

    y_train = normalize_target(train["target"], asset_class)
    y_validation = normalize_target(
        validation["target"], asset_class
    )

    print(f"Training rows   : {len(X_train):,}")
    print(f"Validation rows : {len(X_validation):,}")
    print(f"Feature count   : {len(features)}")

    results = []

    for model_name, estimator in build_models().items():

        print("\nTraining:", model_name)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

        # Imputer and model are fitted using training data only.
        pipeline.fit(X_train, y_train)

        metrics, predictions = evaluate_model(
            pipeline, X_validation, y_validation
        )

        print_results(
            model_name,
            metrics,
            y_validation,
            predictions,
        )

        row = {
            "asset_class": asset_class,
            "horizon": "2y",
            "model": model_name,
            **metrics,
        }

        results.append(row)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TREE-BASED NONLINEAR BASELINE TRAINING")
    print("=" * 80)

    print("\nModels:")
    print("  1. Random Forest")
    print("  2. Extra Trees")

    print("\nPreprocessing:")
    print("  Median imputation fitted on training data only")
    print("  No feature scaling")
    print("  Validation used only for evaluation")
    print("  Test set is not loaded or accessed")

    all_results = []

    for asset_class, paths in DATASETS.items():

        results = train_one_dataset(asset_class, paths)
        all_results.extend(results)

    summary = pd.DataFrame(all_results)

    output_path = RESULTS_DIR / "tree_baseline_summary.csv"
    summary.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("TREE BASELINE SUMMARY")
    print("=" * 80)

    print(summary.to_string(index=False))

    print("\nSaved:", output_path)
    print("\nTEST SET WAS NOT USED.")
    print("Tree-based baseline training completed.")


if __name__ == "__main__":
    main()
