from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"


DATASETS = {
    "stocks": {
        "6m": {
            "train": ML_ROOT / "stocks" / "stock_6m_train.parquet",
            "validation": ML_ROOT / "stocks" / "stock_6m_validation.parquet",
            "test": ML_ROOT / "stocks" / "stock_6m_test.parquet",
        },
        "1y": {
            "train": ML_ROOT / "stocks" / "stock_1y_train.parquet",
            "validation": ML_ROOT / "stocks" / "stock_1y_validation.parquet",
            "test": ML_ROOT / "stocks" / "stock_1y_test.parquet",
        },
        "2y": {
            "train": ML_ROOT / "stocks" / "stock_2y_train.parquet",
            "validation": ML_ROOT / "stocks" / "stock_2y_validation.parquet",
            "test": ML_ROOT / "stocks" / "stock_2y_test.parquet",
        },
    },

    "etfs": {
        "6m": {
            "train": ML_ROOT / "etfs" / "etf_6m_train.parquet",
            "validation": ML_ROOT / "etfs" / "etf_6m_validation.parquet",
            "test": ML_ROOT / "etfs" / "etf_6m_test.parquet",
        },
        "1y": {
            "train": ML_ROOT / "etfs" / "etf_1y_train.parquet",
            "validation": ML_ROOT / "etfs" / "etf_1y_validation.parquet",
            "test": ML_ROOT / "etfs" / "etf_1y_test.parquet",
        },
        "2y": {
            "train": ML_ROOT / "etfs" / "etf_2y_train.parquet",
            "validation": ML_ROOT / "etfs" / "etf_2y_validation.parquet",
            "test": ML_ROOT / "etfs" / "etf_2y_test.parquet",
        },
    },

    "mutual_funds": {
        "6m": {
            "train": ML_ROOT / "mutual_funds" / "mutual_fund_6m_train.parquet",
            "validation": ML_ROOT / "mutual_funds" / "mutual_fund_6m_validation.parquet",
            "test": ML_ROOT / "mutual_funds" / "mutual_fund_6m_test.parquet",
        },
        "1y": {
            "train": ML_ROOT / "mutual_funds" / "mutual_fund_1y_train.parquet",
            "validation": ML_ROOT / "mutual_funds" / "mutual_fund_1y_validation.parquet",
            "test": ML_ROOT / "mutual_funds" / "mutual_fund_1y_test.parquet",
        },
        "2y": {
            "train": ML_ROOT / "mutual_funds" / "mutual_fund_2y_train.parquet",
            "validation": ML_ROOT / "mutual_funds" / "mutual_fund_2y_validation.parquet",
            "test": ML_ROOT / "mutual_funds" / "mutual_fund_2y_test.parquet",
        },
    },
}


EXPECTED_FEATURES = [
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


NON_FEATURE_COLUMNS = {
    "ticker",
    "date",
    "label_end_date",
    "target",
}


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"FAIL: {message}")


def check(condition, message, failures):
    if condition:
        print(f"  PASS: {message}")
    else:
        print(f"  FAIL: {message}")
        failures.append(message)


def safe_numeric(series):
    """
    Convert a pandas Series to numeric safely.

    This is especially important for the mutual-fund datasets,
    where Arrow-backed pandas types may otherwise behave
    differently with numpy operations.
    """
    return pd.to_numeric(series, errors="coerce")


def count_infinite_values(df, columns):
    total = 0

    for column in columns:
        numeric = safe_numeric(df[column])

        if numeric.notna().any():
            total += int(np.isinf(numeric.to_numpy()).sum())

    return total


def feature_missing_summary(df, features):
    result = {}

    for feature in features:
        missing = int(df[feature].isna().sum())

        if missing > 0:
            result[feature] = missing

    return result


def feature_infinite_summary(df, features):
    result = {}

    for feature in features:
        numeric = safe_numeric(df[feature])

        if numeric.notna().any():
            count = int(np.isinf(numeric.to_numpy()).sum())

            if count > 0:
                result[feature] = count

    return result


def feature_variance_summary(df, features):
    """
    Identify constant and near-zero-variance features.

    Near-zero variance threshold:
        variance <= 1e-12

    This is only an audit.
    No feature is removed here.
    """

    constant = []
    near_zero = []

    for feature in features:
        numeric = safe_numeric(df[feature])

        valid = numeric.dropna()

        if len(valid) == 0:
            continue

        unique_count = valid.nunique()

        if unique_count <= 1:
            constant.append(feature)
            continue

        variance = float(valid.var(ddof=0))

        if variance <= 1e-12:
            near_zero.append((feature, variance))

    return constant, near_zero


def target_summary(df):
    if "target" not in df.columns:
        return {}

    return df["target"].value_counts(dropna=False).to_dict()


def complete_feature_rows(df, features):
    return int(df[features].notna().all(axis=1).sum())


def describe_feature_dtypes(df, features):
    return {
        feature: str(df[feature].dtype)
        for feature in features
    }


def get_asset_column(df):
    if "ticker" not in df.columns:
        return None

    return "ticker"


# ============================================================
# SINGLE DATASET AUDIT
# ============================================================

def audit_single_dataset(asset_class, horizon, paths):
    failures = []

    print()
    print("=" * 80)
    print(f"{asset_class.upper()} | {horizon.upper()}")
    print("=" * 80)

    frames = {}

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    for split, path in paths.items():

        print(f"\nLoading {split}:")
        print(f"  {path}")

        if not path.exists():
            failures.append(
                f"{split} file does not exist: {path}"
            )
            continue

        try:
            df = pd.read_parquet(path)
            frames[split] = df
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
        except Exception as exc:
            failures.append(
                f"Could not read {split}: {exc}"
            )

    if len(frames) != 3:
        print("\n  Cannot continue: not all three split files were loaded.")
        return failures

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]

    # --------------------------------------------------------
    # Expected feature list
    # --------------------------------------------------------

    if asset_class == "mutual_funds":
        expected_features = MF_FEATURES
    else:
        expected_features = EXPECTED_FEATURES

    print()
    print("Expected feature count:", len(expected_features))

    # --------------------------------------------------------
    # 1. Required columns
    # --------------------------------------------------------

    print("\n[1] Required columns")

    expected_base_columns = [
        "ticker",
        "date",
        "label_end_date",
        "target",
    ]

    for split_name, df in frames.items():

        missing_base = [
            column
            for column in expected_base_columns
            if column not in df.columns
        ]

        missing_features = [
            feature
            for feature in expected_features
            if feature not in df.columns
        ]

        check(
            len(missing_base) == 0,
            f"{split_name}: all identifier/target columns present",
            failures,
        )

        check(
            len(missing_features) == 0,
            f"{split_name}: all 27 expected features present",
            failures,
        )

        if missing_base:
            print(f"    Missing base columns: {missing_base}")

        if missing_features:
            print(f"    Missing features: {missing_features}")

    # --------------------------------------------------------
    # 2. No unexpected columns
    # --------------------------------------------------------

    print("\n[2] Feature matrix boundary")

    allowed_columns = (
        set(expected_features)
        | NON_FEATURE_COLUMNS
    )

    for split_name, df in frames.items():

        unexpected = [
            column
            for column in df.columns
            if column not in allowed_columns
        ]

        check(
            len(unexpected) == 0,
            f"{split_name}: no unexpected columns",
            failures,
        )

        if unexpected:
            print(f"    Unexpected columns: {unexpected}")

    # --------------------------------------------------------
    # 3. Feature column consistency
    # --------------------------------------------------------

    print("\n[3] Feature-column consistency")

    train_features = [
        column
        for column in train.columns
        if column in expected_features
    ]

    validation_features = [
        column
        for column in validation.columns
        if column in expected_features
    ]

    test_features = [
        column
        for column in test.columns
        if column in expected_features
    ]

    check(
        train_features == validation_features == test_features,
        "Train/validation/test feature columns are identical",
        failures,
    )

    check(
        train_features == expected_features,
        "Feature ordering matches expected 27-feature schema",
        failures,
    )

    # --------------------------------------------------------
    # 4. Feature dtypes
    # --------------------------------------------------------

    print("\n[4] Feature dtypes")

    for split_name, df in frames.items():

        non_numeric = []

        for feature in expected_features:

            if feature not in df.columns:
                continue

            converted = safe_numeric(df[feature])

            # If original values are non-null but conversion
            # produces NaN, the feature contains non-numeric data.
            invalid = (
                df[feature].notna()
                & converted.isna()
            )

            if invalid.any():
                non_numeric.append(feature)

        check(
            len(non_numeric) == 0,
            f"{split_name}: all features are numerically interpretable",
            failures,
        )

        if non_numeric:
            print(
                f"    Non-numeric features: {non_numeric}"
            )

    # --------------------------------------------------------
    # 5. Missing values
    # --------------------------------------------------------

    print("\n[5] Missing feature values")

    total_missing = {}

    for split_name, df in frames.items():

        missing = feature_missing_summary(
            df,
            expected_features,
        )

        total_missing[split_name] = missing

        total = sum(missing.values())

        print(
            f"  {split_name}: {total:,} missing feature values"
        )

        if missing:
            print("    Per-feature:")
            for feature, count in missing.items():
                print(
                    f"      {feature}: {count:,}"
                )

    # --------------------------------------------------------
    # 6. Infinite values
    # --------------------------------------------------------

    print("\n[6] Infinite feature values")

    for split_name, df in frames.items():

        infinite = feature_infinite_summary(
            df,
            expected_features,
        )

        total = sum(infinite.values())

        print(
            f"  {split_name}: {total:,} infinite values"
        )

        check(
            total == 0,
            f"{split_name}: no infinite feature values",
            failures,
        )

        if infinite:
            for feature, count in infinite.items():
                print(
                    f"      {feature}: {count:,}"
                )

    # --------------------------------------------------------
    # 7. Constant / near-zero variance
    # --------------------------------------------------------

    print("\n[7] Feature variance")

    train_constant, train_near_zero = feature_variance_summary(
        train,
        expected_features,
    )

    validation_constant, validation_near_zero = feature_variance_summary(
        validation,
        expected_features,
    )

    test_constant, test_near_zero = feature_variance_summary(
        test,
        expected_features,
    )

    print(f"  Train constant features: {train_constant}")
    print(f"  Validation constant features: {validation_constant}")
    print(f"  Test constant features: {test_constant}")

    print(
        f"  Train near-zero-variance features: "
        f"{[x[0] for x in train_near_zero]}"
    )

    print(
        f"  Validation near-zero-variance features: "
        f"{[x[0] for x in validation_near_zero]}"
    )

    print(
        f"  Test near-zero-variance features: "
        f"{[x[0] for x in test_near_zero]}"
    )

    check(
        len(train_constant) == 0,
        "Train has no constant features",
        failures,
    )

    check(
        len(train_near_zero) == 0,
        "Train has no near-zero-variance features",
        failures,
    )

    # --------------------------------------------------------
    # 8. Target validity
    # --------------------------------------------------------

    print("\n[8] Target validity")

    allowed_targets = {"0", "1", "2", 0, 1, 2}

    for split_name, df in frames.items():

        if "target" not in df.columns:
            failures.append(
                f"{split_name}: target column missing"
            )
            continue

        null_targets = int(df["target"].isna().sum())

        target_values = set(
            df["target"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        invalid_targets = sorted(
            target_values
            - {"0", "1", "2"}
        )

        print(
            f"  {split_name}: "
            f"null={null_targets:,}, "
            f"values={sorted(target_values)}"
        )

        check(
            null_targets == 0,
            f"{split_name}: no null targets",
            failures,
        )

        check(
            len(invalid_targets) == 0,
            f"{split_name}: target values are only 0/1/2",
            failures,
        )

        counts = target_summary(df)

        for target_value in [0, 1, 2, "0", "1", "2"]:
            if target_value in counts:
                print(
                    f"    target={target_value}: "
                    f"{counts[target_value]:,}"
                )

    # --------------------------------------------------------
    # 9. Complete-case rows
    # --------------------------------------------------------

    print("\n[9] Complete feature rows")

    complete_counts = {}

    for split_name, df in frames.items():

        complete = complete_feature_rows(
            df,
            expected_features,
        )

        complete_counts[split_name] = complete

        percentage = (
            100.0 * complete / len(df)
            if len(df) > 0
            else 0.0
        )

        print(
            f"  {split_name}: "
            f"{complete:,}/{len(df):,} "
            f"({percentage:.2f}%) complete"
        )

    # --------------------------------------------------------
    # 10. Identifier / date sanity
    # --------------------------------------------------------

    print("\n[10] Identifier and date sanity")

    for split_name, df in frames.items():

        ticker_nulls = (
            int(df["ticker"].isna().sum())
            if "ticker" in df.columns
            else -1
        )

        date_nulls = (
            int(df["date"].isna().sum())
            if "date" in df.columns
            else -1
        )

        label_date_nulls = (
            int(df["label_end_date"].isna().sum())
            if "label_end_date" in df.columns
            else -1
        )

        print(
            f"  {split_name}: "
            f"ticker_null={ticker_nulls:,}, "
            f"date_null={date_nulls:,}, "
            f"label_end_date_null={label_date_nulls:,}"
        )

        check(
            ticker_nulls == 0,
            f"{split_name}: no null ticker/scheme identifiers",
            failures,
        )

        check(
            date_nulls == 0,
            f"{split_name}: no null dates",
            failures,
        )

        check(
            label_date_nulls == 0,
            f"{split_name}: no null label_end_date values",
            failures,
        )

    # --------------------------------------------------------
    # 11. Duplicate keys within split
    # --------------------------------------------------------

    print("\n[11] Duplicate observation keys")

    for split_name, df in frames.items():

        duplicates = int(
            df.duplicated(
                subset=["ticker", "date"]
            ).sum()
        )

        print(
            f"  {split_name}: {duplicates:,} duplicate keys"
        )

        check(
            duplicates == 0,
            f"{split_name}: no duplicate ticker/date observations",
            failures,
        )

    # --------------------------------------------------------
    # 12. Target leakage into feature matrix
    # --------------------------------------------------------

    print("\n[12] Target leakage boundary")

    forbidden_feature_columns = {
        "target",
        "label_end_date",
        "date",
        "ticker",
    }

    actual_feature_set = set(expected_features)

    check(
        actual_feature_set.isdisjoint(forbidden_feature_columns),
        "Expected feature set contains no identifier/target columns",
        failures,
    )

    # --------------------------------------------------------
    # 13. Feature value sanity
    # --------------------------------------------------------

    print("\n[13] Feature summary")

    print(
        f"\n{'Feature':35s} "
        f"{'Train valid':>12s} "
        f"{'Train min':>14s} "
        f"{'Train max':>14s}"
    )

    print("-" * 82)

    for feature in expected_features:

        numeric = safe_numeric(train[feature])

        valid = numeric.dropna()

        if len(valid) == 0:
            print(
                f"{feature:35s} "
                f"{0:12d} "
                f"{'NA':>14s} "
                f"{'NA':>14s}"
            )
            continue

        print(
            f"{feature:35s} "
            f"{len(valid):12,d} "
            f"{valid.min():14.6g} "
            f"{valid.max():14.6g}"
        )

    # --------------------------------------------------------
    # 14. Asset coverage
    # --------------------------------------------------------

    print("\n[14] Asset coverage")

    for split_name, df in frames.items():

        assets = (
            df["ticker"].nunique()
            if "ticker" in df.columns
            else 0
        )

        print(
            f"  {split_name}: {assets:,} assets"
        )

    train_assets = set(train["ticker"].unique())
    validation_assets = set(validation["ticker"].unique())
    test_assets = set(test["ticker"].unique())

    print(
        f"  Train ∩ Validation: "
        f"{len(train_assets & validation_assets):,}"
    )

    print(
        f"  Train ∩ Test: "
        f"{len(train_assets & test_assets):,}"
    )

    print(
        f"  Validation ∩ Test: "
        f"{len(validation_assets & test_assets):,}"
    )

    # Asset overlap is EXPECTED here.
    # We are doing temporal splits, not asset-disjoint splits.

    # --------------------------------------------------------
    # 15. Preprocessing readiness
    # --------------------------------------------------------

    print("\n[15] Preprocessing readiness")

    train_missing_features = [
        feature
        for feature, count in total_missing["train"].items()
        if count > 0
    ]

    if train_missing_features:
        print(
            "  Train features requiring missing-value handling:"
        )

        for feature in train_missing_features:
            count = total_missing["train"][feature]
            percentage = 100.0 * count / len(train)

            print(
                f"    {feature}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

        print()
        print(
            "  Recommendation: use train-fitted imputation "
            "for models that cannot accept NaN."
        )

    else:
        print(
            "  Train contains no missing feature values."
        )

    print()
    print(
        "  IMPORTANT: any imputer/scaler must be fitted "
        "ONLY on the training split."
    )

    print(
        "  Validation and test data must only be transformed "
        "using parameters learned from training data."
    )

    # --------------------------------------------------------
    # FINAL DATASET STATUS
    # --------------------------------------------------------

    if failures:
        print()
        print(
            f"DATASET STATUS: FAIL "
            f"({len(failures)} issue(s))"
        )

        for failure in failures:
            print(f"  - {failure}")

    else:
        print()
        print("DATASET STATUS: PASS")

    return failures


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ML TRAINING READINESS AUDIT")
    print("=" * 80)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Expected feature architecture:")
    print("  Stocks / ETFs : 27 features")
    print("  Mutual funds  : 27 features")
    print()
    print("Models must never use:")
    print("  ticker")
    print("  date")
    print("  label_end_date")
    print("  target")

    all_failures = []

    dataset_results = []

    # --------------------------------------------------------
    # Run all 9 datasets
    # --------------------------------------------------------

    for asset_class, horizons in DATASETS.items():

        for horizon, paths in horizons.items():

            failures = audit_single_dataset(
                asset_class,
                horizon,
                paths,
            )

            dataset_name = (
                f"{asset_class}_{horizon}"
            )

            dataset_results.append(
                (
                    dataset_name,
                    len(failures) == 0,
                )
            )

            all_failures.extend(
                [
                    f"{dataset_name}: {failure}"
                    for failure in failures
                ]
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TRAINING READINESS SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for dataset_name, status in dataset_results:

        if status:
            print(
                f"  PASS: {dataset_name}"
            )
            passed += 1
        else:
            print(
                f"  FAIL: {dataset_name}"
            )
            failed += 1

    print()
    print(f"Datasets passed: {passed}/9")
    print(f"Datasets failed: {failed}/9")

    if failed == 0:
        print()
        print("ML TRAINING READINESS AUDIT: PASS")
        print()
        print(
            "All 9 temporal datasets are structurally ready "
            "for baseline model training."
        )
        print()
        print("NEXT STEP:")
        print("Baseline model training.")
        print()
        print("Baseline sequence:")
        print("  1. Majority-class baseline")
        print("  2. Multinomial Logistic Regression")
        print("  3. Tree-based baseline")
        print("  4. Validation comparison")
        print("  5. Model selection")
        print("  6. Final untouched test evaluation")

        return 0

    else:
        print()
        print("ML TRAINING READINESS AUDIT: FAIL")
        print()
        print("Issues requiring investigation:")

        for failure in all_failures:
            print(f"  - {failure}")

        return 1


if __name__ == "__main__":
    sys.exit(main())