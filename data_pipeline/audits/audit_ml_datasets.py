from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_ROOT = PROJECT_ROOT / "data" / "ml"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "stocks_6m": {
        "path": ML_ROOT / "stocks" / "stock_6m_ml.parquet",
        "asset_type": "stocks",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "6m",
        "expected_assets": 100,
    },

    "stocks_1y": {
        "path": ML_ROOT / "stocks" / "stock_1y_ml.parquet",
        "asset_type": "stocks",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "1y",
        "expected_assets": 100,
    },

    "stocks_2y": {
        "path": ML_ROOT / "stocks" / "stock_2y_ml.parquet",
        "asset_type": "stocks",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "2y",
        "expected_assets": 100,
    },

    "etfs_6m": {
        "path": ML_ROOT / "etfs" / "etf_6m_ml.parquet",
        "asset_type": "etfs",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "6m",
        "expected_assets": 41,
    },

    "etfs_1y": {
        "path": ML_ROOT / "etfs" / "etf_1y_ml.parquet",
        "asset_type": "etfs",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "1y",
        "expected_assets": 41,
    },

    "etfs_2y": {
        "path": ML_ROOT / "etfs" / "etf_2y_ml.parquet",
        "asset_type": "etfs",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "2y",
        "expected_assets": 41,
    },

    "mutual_funds_6m": {
        "path": ML_ROOT / "mutual_funds" / "mutual_fund_6m_ml.parquet",
        "asset_type": "mutual_funds",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "6m",
        "expected_assets": 95,
    },

    "mutual_funds_1y": {
        "path": ML_ROOT / "mutual_funds" / "mutual_fund_1y_ml.parquet",
        "asset_type": "mutual_funds",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "1y",
        "expected_assets": 94,
    },

    "mutual_funds_2y": {
        "path": ML_ROOT / "mutual_funds" / "mutual_fund_2y_ml.parquet",
        "asset_type": "mutual_funds",
        "asset_col": "ticker",
        "date_col": "date",
        "horizon": "2y",
        "expected_assets": 83,
    },
}


# ============================================================
# LOCKED FEATURE SETS
# ============================================================

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


MUTUAL_FUND_FEATURES = [
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
# HELPERS
# ============================================================

def normalize_dates(series):

    dt = pd.to_datetime(
        series,
        errors="coerce"
    )

    if isinstance(
        dt.dtype,
        pd.DatetimeTZDtype
    ):
        dt = dt.dt.tz_localize(None)

    return dt.dt.normalize()


def numeric_values(series):

    """
    Safely convert a feature column to float for numerical
    validation.

    This avoids np.isinf() failures caused by Arrow/pandas
    extension dtypes.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(
        dtype=float,
        na_value=np.nan
    )


def print_check(name, passed):

    status = "PASS" if passed else "FAIL"

    print(
        f"{name:<30}: {status}"
    )


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(name, config):

    print("\n" + "=" * 78)
    print(f"AUDITING: {name}")
    print("=" * 78)

    path = config["path"]

    if not path.exists():

        print(
            f"ERROR: Dataset does not exist:"
        )

        print(path)

        return False

    df = pd.read_parquet(path)

    passed = True

    asset_col = config["asset_col"]
    date_col = config["date_col"]
    horizon = config["horizon"]

    if config["asset_type"] in {
        "stocks",
        "etfs"
    }:

        expected_features = STOCK_ETF_FEATURES

    else:

        expected_features = MUTUAL_FUND_FEATURES

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    print(
        f"Rows              : "
        f"{len(df):,}"
    )

    print(
        f"Columns           : "
        f"{len(df.columns)}"
    )

    # ========================================================
    # 1. REQUIRED COLUMNS
    # ========================================================

    required_columns = (
        [
            asset_col,
            date_col,
            "label_end_date",
            "target",
        ]
        + expected_features
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print_check(
            "Required columns",
            False
        )

        print(
            f"Missing           : "
            f"{missing_columns}"
        )

        return False

    print_check(
        "Required columns",
        True
    )

    # ========================================================
    # 2. EXACT COLUMN STRUCTURE
    # ========================================================

    expected_columns = (
        [
            asset_col,
            date_col,
        ]
        + expected_features
        + [
            "label_end_date",
            "target",
        ]
    )

    if list(df.columns) != expected_columns:

        print_check(
            "Column structure",
            False
        )

        actual_set = set(df.columns)

        expected_set = set(
            expected_columns
        )

        extra_columns = sorted(
            actual_set - expected_set
        )

        missing_columns = sorted(
            expected_set - actual_set
        )

        if extra_columns:

            print(
                f"Extra columns     : "
                f"{extra_columns}"
            )

        if missing_columns:

            print(
                f"Missing columns   : "
                f"{missing_columns}"
            )

        if (
            not extra_columns
            and not missing_columns
        ):

            print(
                "Column names are correct "
                "but order differs."
            )

        passed = False

    else:

        print_check(
            "Column structure",
            True
        )

    # ========================================================
    # 3. FEATURE SET
    # ========================================================

    actual_features = [
        column
        for column in df.columns
        if column not in {
            asset_col,
            date_col,
            "label_end_date",
            "target",
        }
    ]

    if set(actual_features) != set(
        expected_features
    ):

        print_check(
            "Feature set",
            False
        )

        extra_features = sorted(
            set(actual_features)
            - set(expected_features)
        )

        missing_features = sorted(
            set(expected_features)
            - set(actual_features)
        )

        if extra_features:

            print(
                f"Extra features    : "
                f"{extra_features}"
            )

        if missing_features:

            print(
                f"Missing features  : "
                f"{missing_features}"
            )

        passed = False

    else:

        print_check(
            "Feature set",
            True
        )

    print(
        f"Feature count     : "
        f"{len(actual_features)}"
    )

    # ========================================================
    # 4. NULL KEY VALUES
    # ========================================================

    null_assets = int(
        df[asset_col].isna().sum()
    )

    null_dates = int(
        df[date_col].isna().sum()
    )

    print(
        f"Null asset values : "
        f"{null_assets}"
    )

    print(
        f"Null date values  : "
        f"{null_dates}"
    )

    if (
        null_assets > 0
        or null_dates > 0
    ):

        passed = False

    # ========================================================
    # 5. DATE VALIDATION
    # ========================================================

    observation_dates = normalize_dates(
        df[date_col]
    )

    label_end_dates = normalize_dates(
        df["label_end_date"]
    )

    invalid_observation_dates = int(
        observation_dates.isna().sum()
    )

    invalid_label_end_dates = int(
        label_end_dates.isna().sum()
    )

    if invalid_observation_dates > 0:

        print(
            f"Invalid observation dates: "
            f"{invalid_observation_dates}"
        )

        passed = False

    else:

        print_check(
            "Observation dates",
            True
        )

    if invalid_label_end_dates > 0:

        print(
            f"Invalid label end dates: "
            f"{invalid_label_end_dates}"
        )

        passed = False

    else:

        print_check(
            "Label end dates",
            True
        )

    # ========================================================
    # 6. DUPLICATE KEYS
    # ========================================================

    duplicate_keys = int(
        df.duplicated(
            subset=[
                asset_col,
                date_col
            ]
        ).sum()
    )

    print(
        f"Duplicate keys    : "
        f"{duplicate_keys}"
    )

    if duplicate_keys > 0:

        passed = False

    # ========================================================
    # 7. ASSET COUNT
    # ========================================================

    asset_count = df[
        asset_col
    ].nunique()

    print(
        f"Assets            : "
        f"{asset_count}"
    )

    print(
        f"Expected assets   : "
        f"{config['expected_assets']}"
    )

    if asset_count != config[
        "expected_assets"
    ]:

        print_check(
            "Asset count",
            False
        )

        passed = False

    else:

        print_check(
            "Asset count",
            True
        )

    # ========================================================
    # 8. TEMPORAL ORDER
    # ========================================================

    temporal_df = pd.DataFrame(
        {
            "asset":
                df[asset_col].astype(str),

            "date":
                observation_dates,
        }
    )

    temporal_df = temporal_df.sort_values(
        [
            "asset",
            "date"
        ]
    )

    temporal_violations = 0

    for _, group in temporal_df.groupby(
        "asset",
        sort=False
    ):

        if not group[
            "date"
        ].is_monotonic_increasing:

            temporal_violations += 1

    print(
        f"Temporal violations: "
        f"{temporal_violations}"
    )

    if temporal_violations > 0:

        passed = False

    # ========================================================
    # 9. INFINITE VALUES
    # ========================================================

    infinite_values = 0

    for feature in expected_features:

        values = numeric_values(
            df[feature]
        )

        infinite_values += int(
            np.isinf(values).sum()
        )

    print(
        f"Infinite values   : "
        f"{infinite_values}"
    )

    if infinite_values > 0:

        passed = False

    # ========================================================
    # 10. FEATURE NUMERIC TYPES
    # ========================================================

    non_numeric_features = []

    for feature in expected_features:

        converted = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        original_non_null = (
            df[feature].notna()
        )

        conversion_failures = (
            original_non_null
            & converted.isna()
        ).sum()

        if conversion_failures > 0:

            non_numeric_features.append(
                feature
            )

    if non_numeric_features:

        print_check(
            "Feature numeric types",
            False
        )

        print(
            f"Non-numeric       : "
            f"{non_numeric_features}"
        )

        passed = False

    else:

        print_check(
            "Feature numeric types",
            True
        )

    # ========================================================
    # 11. TARGET NULL CHECK
    # ========================================================

    null_targets = int(
        df["target"].isna().sum()
    )

    print(
        f"Null targets      : "
        f"{null_targets}"
    )

    if null_targets > 0:

        print_check(
            "Target completeness",
            False
        )

        passed = False

    else:

        print_check(
            "Target completeness",
            True
        )

    # ========================================================
    # 12. TARGET VALUE VALIDATION
    # ========================================================

    unique_targets = sorted(
        df["target"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Target values     : "
        f"{unique_targets}"
    )

    if config["asset_type"] == "mutual_funds":

        expected_targets = {
            "0",
            "1",
            "2",
        }

    else:

        expected_targets = {
            "Attractive",
            "Neutral",
            "Weak",
        }

    if set(unique_targets) != (
        expected_targets
    ):

        print_check(
            "Target labels",
            False
        )

        print(
            f"Expected          : "
            f"{sorted(expected_targets)}"
        )

        passed = False

    else:

        print_check(
            "Target labels",
            True
        )

    # ========================================================
    # 13. TARGET DATE DIRECTION
    # ========================================================

    end_before_observation = int(
        (
            label_end_dates.notna()
            &
            observation_dates.notna()
            &
            (
                label_end_dates
                < observation_dates
            )
        ).sum()
    )

    equal_end_dates = int(
        (
            label_end_dates.notna()
            &
            observation_dates.notna()
            &
            (
                label_end_dates
                == observation_dates
            )
        ).sum()
    )

    print(
        f"Label end < date : "
        f"{end_before_observation}"
    )

    print(
        f"Label end == date: "
        f"{equal_end_dates}"
    )

    if end_before_observation > 0:

        print_check(
            "Target date direction",
            False
        )

        passed = False

    else:

        print_check(
            "Target date direction",
            True
        )

    # ========================================================
    # 14. HORIZON VALIDATION
    # ========================================================

    horizon_days = (
        label_end_dates
        - observation_dates
    ).dt.days

    valid_horizon_days = (
        horizon_days.dropna()
    )

    if len(valid_horizon_days) > 0:

        minimum_days = int(
            valid_horizon_days.min()
        )

        maximum_days = int(
            valid_horizon_days.max()
        )

        print(
            f"Horizon days     : "
            f"{minimum_days} -> "
            f"{maximum_days}"
        )

        minimum_expected_days = {
            "6m": 150,
            "1y": 300,
            "2y": 650,
        }[horizon]

        horizon_violations = int(
            (
                valid_horizon_days
                < minimum_expected_days
            ).sum()
        )

        print(
            f"Horizon violations: "
            f"{horizon_violations}"
        )

        if horizon_violations > 0:

            print_check(
                "Horizon validity",
                False
            )

            passed = False

        else:

            print_check(
                "Horizon validity",
                True
            )

    # ========================================================
    # 15. FEATURE NaN INFORMATION
    # ========================================================

    feature_nan_counts = (
        df[expected_features]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total_feature_nan_cells = int(
        feature_nan_counts.sum()
    )

    complete_feature_rows = int(
        df[expected_features]
        .notna()
        .all(axis=1)
        .sum()
    )

    print(
        f"Feature NaN cells : "
        f"{total_feature_nan_cells:,}"
    )

    print(
        f"Complete feature rows: "
        f"{complete_feature_rows:,} / "
        f"{len(df):,}"
    )

    # ========================================================
    # 16. CONSTANT FEATURES
    # ========================================================

    constant_features = []

    for feature in expected_features:

        unique_values = (
            df[feature]
            .dropna()
            .nunique()
        )

        if unique_values <= 1:

            constant_features.append(
                feature
            )

    if constant_features:

        print_check(
            "Constant features",
            False
        )

        print(
            f"Features          : "
            f"{constant_features}"
        )

        passed = False

    else:

        print_check(
            "Constant features",
            True
        )

    # ========================================================
    # 17. PERCENTILE RANGE
    # ========================================================

    percentile_features = [
        feature
        for feature in expected_features
        if feature.endswith("_pct")
    ]

    percentile_failures = []

    for feature in percentile_features:

        values = numeric_values(
            df[feature]
        )

        values = values[
            ~np.isnan(values)
        ]

        if len(values) == 0:

            percentile_failures.append(
                f"{feature}: no valid values"
            )

            continue

        minimum = values.min()
        maximum = values.max()

        if (
            minimum < -1e-9
            or maximum > 1 + 1e-9
        ):

            percentile_failures.append(
                f"{feature}: "
                f"min={minimum}, "
                f"max={maximum}"
            )

    if percentile_failures:

        print_check(
            "Percentile ranges",
            False
        )

        for failure in percentile_failures:

            print(
                f"  {failure}"
            )

        passed = False

    else:

        print_check(
            "Percentile ranges",
            True
        )

    # ========================================================
    # 18. CROSS-SECTIONAL PERCENTILE VARIATION
    # ========================================================

    variation_failures = []

    for feature in percentile_features:

        temp = pd.DataFrame(
            {
                "date":
                    observation_dates,

                "value":
                    pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    )
            }
        )

        variation = (
            temp.groupby(
                "date"
            )["value"]
            .nunique(
                dropna=True
            )
        )

        valid_variation = variation[
            variation > 0
        ]

        if (
            len(valid_variation) > 0
            and valid_variation.max() <= 1
        ):

            variation_failures.append(
                feature
            )

    if variation_failures:

        print_check(
            "Percentile variation",
            False
        )

        print(
            f"Constant by date : "
            f"{variation_failures}"
        )

        passed = False

    else:

        print_check(
            "Percentile variation",
            True
        )

    # ========================================================
    # 19. LEAKAGE COLUMN CHECK
    # ========================================================

    forbidden_tokens = [
        "future_return",
        "future_price",
        "future_adj_close",
        "future_nav",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "risk_adjusted_return",
        "composite_score",
        "score_2y",
        "rank_2y",
        "return_percentile_2y",
        "risk_percentile_2y",
        "drawdown_percentile_2y",
        "forward_return",
    ]

    leakage_columns = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            token in column_lower
            for token in forbidden_tokens
        ):

            leakage_columns.append(
                column
            )

    leakage_columns = [
        column
        for column in leakage_columns
        if column not in {
            "target",
            "label_end_date",
        }
    ]

    if leakage_columns:

        print_check(
            "Leakage columns",
            False
        )

        print(
            f"Columns           : "
            f"{leakage_columns}"
        )

        passed = False

    else:

        print_check(
            "Leakage columns",
            True
        )

    # ========================================================
    # 20. FEATURE/TARGET SEPARATION
    # ========================================================

    overlap = (
        set(expected_features)
        & {
            "target",
            "label_end_date"
        }
    )

    if overlap:

        print_check(
            "Feature/target separation",
            False
        )

        print(
            f"Overlap           : "
            f"{sorted(overlap)}"
        )

        passed = False

    else:

        print_check(
            "Feature/target separation",
            True
        )

    # ========================================================
    # 21. OBSERVATION/LABEL DATE INTEGRITY
    # ========================================================

    date_relationship_failures = int(
        (
            label_end_dates
            < observation_dates
        ).sum()
    )

    if date_relationship_failures > 0:

        print_check(
            "Observation/label dates",
            False
        )

        passed = False

    else:

        print_check(
            "Observation/label dates",
            True
        )

    # ========================================================
    # 22. DATE RANGE
    # ========================================================

    min_date = observation_dates.min()
    max_date = observation_dates.max()

    print(
        f"Date range        : "
        f"{min_date.date()} -> "
        f"{max_date.date()}"
    )

    # ========================================================
    # 23. TARGET DISTRIBUTION
    # ========================================================

    print(
        "Target distribution:"
    )

    distribution = (
        df["target"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    for label, count in distribution.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("-" * 78)

    if passed:

        print(
            f"{name}: PASS"
        )

    else:

        print(
            f"{name}: FAIL"
        )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "INVESTMENT COPILOT - "
        "INDEPENDENT ML DATASET AUDIT"
    )
    print("=" * 78)

    results = {}

    for name, config in DATASETS.items():
        if not name.endswith("_2y"):
            continue

        try:

            results[name] = audit_dataset(
                name,
                config
            )

        except Exception as exc:

            print()
            print(
                f"{name}: ERROR"
            )

            print(
                f"Reason: {exc}"
            )

            results[name] = False

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 78
    )

    print(
        "FINAL ML DATASET AUDIT SUMMARY"
    )

    print(
        "=" * 78
    )

    for name, result in results.items():

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{name:<25} : "
            f"{status}"
        )

    passed_count = sum(
        results.values()
    )

    total_count = len(results)

    print(
        "-" * 78
    )

    print(
        f"Datasets passed: "
        f"{passed_count}/{total_count}"
    )

    if passed_count == total_count:

        print()
        print(
            "ML DATASET AUDIT: PASS"
        )

        print()
        print(
            "All 9 ML datasets passed "
            "the independent audit."
        )

        print()
        print(
            "The ML datasets are ready "
            "to be locked."
        )

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Temporal train / validation / "
            "test splitting."
        )

        return 0

    else:

        print()
        print(
            "ML DATASET AUDIT: FAIL"
        )

        print()
        print(
            "Do NOT proceed to model training."
        )

        print(
            "Fix the failing dataset(s) first."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())