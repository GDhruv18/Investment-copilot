from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"

RESULTS_DIR = ML_ROOT / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "stocks": {
        "6m": (
            ML_ROOT / "stocks" / "stock_6m_train.parquet",
            ML_ROOT / "stocks" / "stock_6m_validation.parquet",
            ML_ROOT / "stocks" / "stock_6m_test.parquet",
        ),
        "1y": (
            ML_ROOT / "stocks" / "stock_1y_train.parquet",
            ML_ROOT / "stocks" / "stock_1y_validation.parquet",
            ML_ROOT / "stocks" / "stock_1y_test.parquet",
        ),
        "2y": (
            ML_ROOT / "stocks" / "stock_2y_train.parquet",
            ML_ROOT / "stocks" / "stock_2y_validation.parquet",
            ML_ROOT / "stocks" / "stock_2y_test.parquet",
        ),
    },

    "etfs": {
        "6m": (
            ML_ROOT / "etfs" / "etf_6m_train.parquet",
            ML_ROOT / "etfs" / "etf_6m_validation.parquet",
            ML_ROOT / "etfs" / "etf_6m_test.parquet",
        ),
        "1y": (
            ML_ROOT / "etfs" / "etf_1y_train.parquet",
            ML_ROOT / "etfs" / "etf_1y_validation.parquet",
            ML_ROOT / "etfs" / "etf_1y_test.parquet",
        ),
        "2y": (
            ML_ROOT / "etfs" / "etf_2y_train.parquet",
            ML_ROOT / "etfs" / "etf_2y_validation.parquet",
            ML_ROOT / "etfs" / "etf_2y_test.parquet",
        ),
    },

    "mutual_funds": {
        "6m": (
            ML_ROOT / "mutual_funds" / "mutual_fund_6m_train.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_6m_validation.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_6m_test.parquet",
        ),
        "1y": (
            ML_ROOT / "mutual_funds" / "mutual_fund_1y_train.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_1y_validation.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_1y_test.parquet",
        ),
        "2y": (
            ML_ROOT / "mutual_funds" / "mutual_fund_2y_train.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_2y_validation.parquet",
            ML_ROOT / "mutual_funds" / "mutual_fund_2y_test.parquet",
        ),
    },
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
    """
    Convert the project's existing target representation into
    integer ML classes.

    Stocks / ETFs:
        Weak       -> 0
        Neutral    -> 1
        Attractive -> 2

    Mutual funds:
        "0" -> 0
        "1" -> 1
        "2" -> 2

    This DOES NOT modify the source parquet files.
    """

    if asset_class in {"stocks", "etfs"}:

        mapping = {
            "Weak": 0,
            "Neutral": 1,
            "Attractive": 2,
        }

        normalized = series.astype(str).map(mapping)

    elif asset_class == "mutual_funds":

        normalized = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
        )

    else:
        raise ValueError(
            f"Unknown asset class: {asset_class}"
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

    invalid_classes = sorted(
        set(normalized.unique())
        - {0, 1, 2}
    )

    if invalid_classes:
        raise ValueError(
            f"Invalid normalized target classes: "
            f"{invalid_classes}"
        )

    return normalized


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(
    train_path,
    validation_path,
    test_path,
    asset_class,
):
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)

    if asset_class == "mutual_funds":
        features = MF_FEATURES
    else:
        features = STOCK_ETF_FEATURES

    X_train = train[features].copy()
    X_validation = validation[features].copy()
    X_test = test[features].copy()

    y_train = normalize_target(
        train["target"],
        asset_class,
    )

    y_validation = normalize_target(
        validation["target"],
        asset_class,
    )

    y_test = normalize_target(
        test["target"],
        asset_class,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def print_metrics(name, metrics):
    print(
        f"  {name:<25s}"
        f" Accuracy={metrics['accuracy']:.4f}  "
        f"BalancedAcc={metrics['balanced_accuracy']:.4f}  "
        f"MacroF1={metrics['macro_f1']:.4f}  "
        f"WeightedF1={metrics['weighted_f1']:.4f}"
    )


# ============================================================
# MAJORITY BASELINE
# ============================================================

def train_majority_baseline(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Majority-class baseline.

    This is the minimum performance that any useful model
    should beat.
    """

    model = DummyClassifier(
        strategy="most_frequent"
    )

    # DummyClassifier does not actually need features,
    # but using X_train keeps the interface consistent.
    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    return model, metrics, predictions


# ============================================================
# LOGISTIC REGRESSION BASELINE
# ============================================================

def build_logistic_pipeline():
    """
    Pipeline:

        Missing-value imputation
                ↓
        StandardScaler
                ↓
        Logistic Regression

    IMPORTANT:
    SimpleImputer and StandardScaler are fitted only on
    training data by sklearn Pipeline.fit().
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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

    return pipeline


def train_logistic_baseline(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    model = build_logistic_pipeline()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    return model, metrics, predictions


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(y_true, y_pred):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    print()
    print("  Confusion matrix:")
    print()
    print(
        "              Pred 0   Pred 1   Pred 2"
    )

    print(
        f"Actual 0     {matrix[0, 0]:8d}"
        f" {matrix[0, 1]:8d}"
        f" {matrix[0, 2]:8d}"
    )

    print(
        f"Actual 1     {matrix[1, 0]:8d}"
        f" {matrix[1, 1]:8d}"
        f" {matrix[1, 2]:8d}"
    )

    print(
        f"Actual 2     {matrix[2, 0]:8d}"
        f" {matrix[2, 1]:8d}"
        f" {matrix[2, 2]:8d}"
    )


# ============================================================
# MAIN DATASET TRAINING
# ============================================================

def train_one_dataset(
    asset_class,
    horizon,
    paths,
):
    print()
    print("=" * 80)
    print(
        f"{asset_class.upper()} | {horizon.upper()}"
    )
    print("=" * 80)

    print()
    print("Loading datasets...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_dataset(
        paths[0],
        paths[1],
        paths[2],
        asset_class,
    )

    print(
        f"Train      : {len(X_train):,} rows"
    )

    print(
        f"Validation : {len(X_validation):,} rows"
    )

    print(
        f"Test       : {len(X_test):,} rows"
    )

    print()
    print("Feature count:", X_train.shape[1])

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("Training class distribution:")

    train_counts = y_train.value_counts(
        normalize=True
    ).sort_index()

    for class_id in [0, 1, 2]:

        count = int(
            (y_train == class_id).sum()
        )

        percentage = (
            100.0 * train_counts.get(
                class_id,
                0.0
            )
        )

        print(
            f"  Class {class_id}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    print()
    print("MODEL 1: MAJORITY-CLASS BASELINE")

    (
        majority_model,
        majority_metrics,
        majority_predictions,
    ) = train_majority_baseline(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print_metrics(
        "Majority",
        majority_metrics,
    )

    # --------------------------------------------------------
    # Logistic regression
    # --------------------------------------------------------

    print()
    print("MODEL 2: MULTINOMIAL LOGISTIC REGRESSION")

    (
        logistic_model,
        logistic_metrics,
        logistic_predictions,
    ) = train_logistic_baseline(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print_metrics(
        "Logistic Regression",
        logistic_metrics,
    )

    print_confusion_matrix(
        y_validation,
        logistic_predictions,
    )

    # --------------------------------------------------------
    # Improvement over majority
    # --------------------------------------------------------

    print()
    print("Validation improvement over majority baseline:")

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]:

        improvement = (
            logistic_metrics[metric]
            - majority_metrics[metric]
        )

        print(
            f"  {metric:<20s}: "
            f"{improvement:+.4f}"
        )

    # --------------------------------------------------------
    # Store validation results
    # --------------------------------------------------------

    result_row = {
        "asset_class": asset_class,
        "horizon": horizon,

        "majority_accuracy":
            majority_metrics["accuracy"],

        "majority_balanced_accuracy":
            majority_metrics["balanced_accuracy"],

        "majority_macro_f1":
            majority_metrics["macro_f1"],

        "majority_weighted_f1":
            majority_metrics["weighted_f1"],

        "logistic_accuracy":
            logistic_metrics["accuracy"],

        "logistic_balanced_accuracy":
            logistic_metrics["balanced_accuracy"],

        "logistic_macro_f1":
            logistic_metrics["macro_f1"],

        "logistic_weighted_f1":
            logistic_metrics["weighted_f1"],
    }

    result_path = (
        RESULTS_DIR
        / f"{asset_class}_{horizon}_baseline_results.csv"
    )

    pd.DataFrame(
        [result_row]
    ).to_csv(
        result_path,
        index=False,
    )

    print()
    print(
        f"Validation results saved to:"
    )
    print(result_path)

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

    print()
    print(
        "TEST SET WAS NOT USED FOR MODEL SELECTION."
    )

    print(
        "The test data was loaded only to verify that the "
        "dataset structure is available."
    )

    print(
        "No test predictions were generated."
    )

    return result_row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BASELINE MODEL TRAINING")
    print("=" * 80)

    print()
    print("Models:")
    print("  1. Majority-class baseline")
    print("  2. Multinomial Logistic Regression")

    print()
    print("Evaluation:")
    print("  Training -> fit")
    print("  Validation -> model comparison")
    print("  Test -> untouched")

    print()
    print("Preprocessing:")
    print("  Median imputation -> TRAIN ONLY")
    print("  StandardScaler -> TRAIN ONLY")
    print("  Validation/test -> transformed using train parameters")

    all_results = []

    for asset_class, horizons in DATASETS.items():

        for horizon, paths in horizons.items():
            if horizon != "2y":
                continue
            result = train_one_dataset(
                asset_class,
                horizon,
                paths,
            )

            all_results.append(result)

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        all_results
    )

    summary_path = (
        RESULTS_DIR
        / "baseline_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("BASELINE TRAINING SUMMARY")
    print("=" * 80)

    print()

    display_columns = [
        "asset_class",
        "horizon",
        "majority_accuracy",
        "majority_balanced_accuracy",
        "majority_macro_f1",
        "logistic_accuracy",
        "logistic_balanced_accuracy",
        "logistic_macro_f1",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Combined results saved to:"
    )
    print(summary_path)

    print()
    print("NEXT STEP:")
    print(
        "Compare Logistic Regression against the "
        "majority baseline."
    )
    print(
        "Then train the tree-based nonlinear baseline."
    )


if __name__ == "__main__":
    sys.exit(main())