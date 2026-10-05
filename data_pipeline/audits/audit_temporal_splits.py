from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "stocks": {
        "6m": {
            "source": PROJECT_ROOT / "data/ml/stocks/stock_6m_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/stocks/stock_6m_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/stocks/stock_6m_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/stocks/stock_6m_test.parquet",
        },
        "1y": {
            "source": PROJECT_ROOT / "data/ml/stocks/stock_1y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/stocks/stock_1y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/stocks/stock_1y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/stocks/stock_1y_test.parquet",
        },
        "2y": {
            "source": PROJECT_ROOT / "data/ml/stocks/stock_2y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/stocks/stock_2y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/stocks/stock_2y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/stocks/stock_2y_test.parquet",
        },
    },

    "etfs": {
        "6m": {
            "source": PROJECT_ROOT / "data/ml/etfs/etf_6m_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/etfs/etf_6m_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/etfs/etf_6m_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/etfs/etf_6m_test.parquet",
        },
        "1y": {
            "source": PROJECT_ROOT / "data/ml/etfs/etf_1y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/etfs/etf_1y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/etfs/etf_1y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/etfs/etf_1y_test.parquet",
        },
        "2y": {
            "source": PROJECT_ROOT / "data/ml/etfs/etf_2y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/etfs/etf_2y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/etfs/etf_2y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/etfs/etf_2y_test.parquet",
        },
    },

    "mutual_funds": {
        "6m": {
            "source": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_6m_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_6m_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_6m_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_6m_test.parquet",
        },
        "1y": {
            "source": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_1y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_1y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_1y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_1y_test.parquet",
        },
        "2y": {
            "source": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_ml.parquet",
            "train": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_train.parquet",
            "validation": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_validation.parquet",
            "test": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_test.parquet",
        },
    },
}


SPLIT_POLICY = {
    "6m": {
        "train_boundary": 0.60,
        "test_boundary": 0.85,
    },
    "1y": {
        "train_boundary": 0.55,
        "test_boundary": 0.85,
    },
    "2y": {
        "train_boundary": 0.40,
        "test_boundary": 0.85,
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize_dates(df):
    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"]
    ).dt.normalize()

    return df


def get_boundary_date(dates, fraction):
    unique_dates = np.sort(
        pd.Series(dates)
        .dropna()
        .unique()
    )

    if len(unique_dates) == 0:
        raise ValueError(
            "No valid dates found."
        )

    index = int(
        round(
            (len(unique_dates) - 1)
            * fraction
        )
    )

    index = max(
        0,
        min(index, len(unique_dates) - 1)
    )

    return pd.Timestamp(
        unique_dates[index]
    )


def make_key_set(df):
    return set(
        zip(
            df["ticker"].astype(str),
            pd.to_datetime(df["date"]).dt.normalize(),
        )
    )


def get_label_distribution(df):
    """
    Handles:
      0 / 1 / 2
      '0' / '1' / '2'
      Weak / Neutral / Attractive
      weak / neutral / attractive
    """

    if len(df) == 0:
        return {
            "0": 0.0,
            "1": 0.0,
            "2": 0.0,
        }

    values = (
        df["target"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": "0",
        "1": "1",
        "2": "2",
        "weak": "0",
        "neutral": "1",
        "attractive": "2",
    }

    normalized = values.map(mapping)

    counts = (
        normalized
        .value_counts(normalize=True)
        * 100
    )

    return {
        "0": counts.get("0", 0.0),
        "1": counts.get("1", 0.0),
        "2": counts.get("2", 0.0),
    }


def check_required_columns(df, name):
    required = {
        "ticker",
        "date",
        "label_end_date",
        "target",
    }

    missing = required - set(df.columns)

    if missing:
        raise AssertionError(
            f"{name}: missing columns: "
            f"{sorted(missing)}"
        )


def assert_no_duplicates(df, name):
    duplicates = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicates:
        raise AssertionError(
            f"{name}: {duplicates:,} duplicate "
            f"ticker/date keys."
        )


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(
    dataset_name,
    horizon,
    paths,
):
    print()
    print("=" * 100)
    print(
        f"AUDITING {dataset_name.upper()} | "
        f"{horizon.upper()}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Load source and split files
    # --------------------------------------------------------

    source = normalize_dates(
        pd.read_parquet(paths["source"])
    )

    train = normalize_dates(
        pd.read_parquet(paths["train"])
    )

    validation = normalize_dates(
        pd.read_parquet(paths["validation"])
    )

    test = normalize_dates(
        pd.read_parquet(paths["test"])
    )

    # --------------------------------------------------------
    # Schema checks
    # --------------------------------------------------------

    check_required_columns(
        source,
        "SOURCE",
    )

    check_required_columns(
        train,
        "TRAIN",
    )

    check_required_columns(
        validation,
        "VALIDATION",
    )

    check_required_columns(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    assert_no_duplicates(
        source,
        "SOURCE",
    )

    assert_no_duplicates(
        train,
        "TRAIN",
    )

    assert_no_duplicates(
        validation,
        "VALIDATION",
    )

    assert_no_duplicates(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # Calculate expected boundaries independently
    # --------------------------------------------------------

    policy = SPLIT_POLICY[horizon]

    validation_start = get_boundary_date(
        source["date"],
        policy["train_boundary"],
    )

    test_start = get_boundary_date(
        source["date"],
        policy["test_boundary"],
    )

    # --------------------------------------------------------
    # Reconstruct expected partitions from SOURCE
    # --------------------------------------------------------

    expected_train_mask = (
        source["date"] < validation_start
    )

    expected_validation_mask = (
        (source["date"] >= validation_start)
        & (source["date"] < test_start)
        & (source["label_end_date"] < test_start)
    )

    expected_test_mask = (
        source["date"] >= test_start
    )

    expected_train = source.loc[
        expected_train_mask
    ]

    expected_validation = source.loc[
        expected_validation_mask
    ]

    expected_test = source.loc[
        expected_test_mask
    ]

    # --------------------------------------------------------
    # Key sets
    # --------------------------------------------------------

    source_keys = make_key_set(source)

    train_keys = make_key_set(train)
    validation_keys = make_key_set(validation)
    test_keys = make_key_set(test)

    expected_train_keys = make_key_set(
        expected_train
    )

    expected_validation_keys = make_key_set(
        expected_validation
    )

    expected_test_keys = make_key_set(
        expected_test
    )

    # --------------------------------------------------------
    # Exact partition equality
    # --------------------------------------------------------

    if train_keys != expected_train_keys:
        missing = expected_train_keys - train_keys
        extra = train_keys - expected_train_keys

        raise AssertionError(
            f"TRAIN key mismatch. "
            f"Missing={len(missing):,}, "
            f"Extra={len(extra):,}"
        )

    if validation_keys != expected_validation_keys:
        missing = (
            expected_validation_keys
            - validation_keys
        )

        extra = (
            validation_keys
            - expected_validation_keys
        )

        raise AssertionError(
            f"VALIDATION key mismatch. "
            f"Missing={len(missing):,}, "
            f"Extra={len(extra):,}"
        )

    if test_keys != expected_test_keys:
        missing = expected_test_keys - test_keys
        extra = test_keys - expected_test_keys

        raise AssertionError(
            f"TEST key mismatch. "
            f"Missing={len(missing):,}, "
            f"Extra={len(extra):,}"
        )

    # --------------------------------------------------------
    # Split overlap checks
    # --------------------------------------------------------

    if train_keys & validation_keys:
        raise AssertionError(
            "Train/validation overlap."
        )

    if train_keys & test_keys:
        raise AssertionError(
            "Train/test overlap."
        )

    if validation_keys & test_keys:
        raise AssertionError(
            "Validation/test overlap."
        )

    # --------------------------------------------------------
    # Source coverage
    #
    # The split files intentionally exclude purged rows.
    # Therefore:
    #
    # source =
    # train + validation + test + purged
    # --------------------------------------------------------

    assigned_keys = (
        train_keys
        | validation_keys
        | test_keys
    )

    if not assigned_keys.issubset(
        source_keys
    ):
        raise AssertionError(
            "Split contains keys not present "
            "in source dataset."
        )

    purged_keys = (
        source_keys
        - assigned_keys
    )

    # --------------------------------------------------------
    # Reconstruct expected purged keys
    # --------------------------------------------------------

    expected_purged_mask = (
        (source["date"] >= validation_start)
        & (source["date"] < test_start)
        & (source["label_end_date"] >= test_start)
    )

    expected_purged = source.loc[
        expected_purged_mask
    ]

    expected_purged_keys = make_key_set(
        expected_purged
    )

    if purged_keys != expected_purged_keys:
        raise AssertionError(
            "Purged key set does not match "
            "the expected target-overlap rows."
        )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    if len(train):
        if not (
            train["date"] < validation_start
        ).all():
            raise AssertionError(
                "Train contains dates >= validation_start."
            )

    if len(validation):

        valid_date_range = (
            (validation["date"] >= validation_start)
            & (validation["date"] < test_start)
        )

        if not valid_date_range.all():
            raise AssertionError(
                "Validation contains an invalid "
                "observation date."
            )

        if not (
            validation["label_end_date"]
            < test_start
        ).all():
            raise AssertionError(
                "VALIDATION LEAKAGE: "
                "a validation target reaches "
                "into the test period."
            )

    if len(test):
        if not (
            test["date"] >= test_start
        ).all():
            raise AssertionError(
                "Test contains dates before test_start."
            )

    # --------------------------------------------------------
    # Feature-column consistency
    # --------------------------------------------------------

    if list(train.columns) != list(source.columns):
        raise AssertionError(
            "Train columns differ from source columns."
        )

    if list(validation.columns) != list(source.columns):
        raise AssertionError(
            "Validation columns differ from source columns."
        )

    if list(test.columns) != list(source.columns):
        raise AssertionError(
            "Test columns differ from source columns."
        )

    # --------------------------------------------------------
    # Row accounting
    # --------------------------------------------------------

    expected_total = (
        len(train)
        + len(validation)
        + len(test)
        + len(expected_purged)
    )

    if expected_total != len(source):
        raise AssertionError(
            "Row accounting mismatch."
        )

    # --------------------------------------------------------
    # Asset coverage
    # --------------------------------------------------------

    source_assets = set(
        source["ticker"].astype(str)
    )

    train_assets = set(
        train["ticker"].astype(str)
    )

    validation_assets = set(
        validation["ticker"].astype(str)
    )

    test_assets = set(
        test["ticker"].astype(str)
    )

    # No split may contain an unknown asset.
    if not train_assets.issubset(source_assets):
        raise AssertionError(
            "Train contains unknown assets."
        )

    if not validation_assets.issubset(source_assets):
        raise AssertionError(
            "Validation contains unknown assets."
        )

    if not test_assets.issubset(source_assets):
        raise AssertionError(
            "Test contains unknown assets."
        )

    # --------------------------------------------------------
    # Target integrity
    # --------------------------------------------------------

    source_targets = set(
        source["target"]
        .astype(str)
        .unique()
    )

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        split_targets = set(
            df["target"]
            .astype(str)
            .unique()
        )

        if not split_targets.issubset(
            source_targets
        ):
            raise AssertionError(
                f"{name}: contains target values "
                f"not present in source."
            )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    train_dist = get_label_distribution(
        train
    )

    validation_dist = get_label_distribution(
        validation
    )

    test_dist = get_label_distribution(
        test
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    total = len(source)

    print(
        f"Source rows        : {len(source):,}"
    )

    print(
        f"Train rows         : {len(train):,} "
        f"({len(train) / total * 100:.2f}%)"
    )

    print(
        f"Validation rows    : {len(validation):,} "
        f"({len(validation) / total * 100:.2f}%)"
    )

    print(
        f"Test rows          : {len(test):,} "
        f"({len(test) / total * 100:.2f}%)"
    )

    print(
        f"Purged rows        : {len(expected_purged):,} "
        f"({len(expected_purged) / total * 100:.2f}%)"
    )

    print()
    print(
        f"Validation start   : "
        f"{validation_start.date()}"
    )

    print(
        f"Test start         : "
        f"{test_start.date()}"
    )

    print()
    print(
        f"Assets             : "
        f"{len(source_assets)}"
    )

    print(
        f"Train assets       : "
        f"{len(train_assets)}"
    )

    print(
        f"Validation assets  : "
        f"{len(validation_assets)}"
    )

    print(
        f"Test assets        : "
        f"{len(test_assets)}"
    )

    print()
    print("Label distribution:")

    print(
        f"  Train      : "
        f"0={train_dist['0']:.2f}% | "
        f"1={train_dist['1']:.2f}% | "
        f"2={train_dist['2']:.2f}%"
    )

    print(
        f"  Validation : "
        f"0={validation_dist['0']:.2f}% | "
        f"1={validation_dist['1']:.2f}% | "
        f"2={validation_dist['2']:.2f}%"
    )

    print(
        f"  Test       : "
        f"0={test_dist['0']:.2f}% | "
        f"1={test_dist['1']:.2f}% | "
        f"2={test_dist['2']:.2f}%"
    )

    print()
    print("Checks:")
    print("  Source/split key equality       : PASS")
    print("  No duplicate split keys         : PASS")
    print("  No cross-split overlap          : PASS")
    print("  Purged-key reconstruction       : PASS")
    print("  Temporal boundary correctness   : PASS")
    print("  Validation target leakage       : PASS")
    print("  Feature-column consistency      : PASS")
    print("  Target-value integrity          : PASS")
    print("  Asset integrity                 : PASS")
    print("  Row accounting                  : PASS")

    print()
    print(
        "TEMPORAL SPLIT AUDIT: PASS"
    )

    return {
        "dataset": dataset_name,
        "horizon": horizon,
        "source_rows": len(source),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "purged_rows": len(expected_purged),
        "source_assets": len(source_assets),
        "train_assets": len(train_assets),
        "validation_assets": len(validation_assets),
        "test_assets": len(test_assets),
        "validation_start": validation_start,
        "test_start": test_start,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("INDEPENDENT TEMPORAL SPLIT AUDIT")
    print("=" * 100)

    results = []

    passed = 0
    failed = 0

    for dataset_name, horizons in DATASETS.items():

        for horizon, paths in horizons.items():

            try:

                result = audit_dataset(
                    dataset_name,
                    horizon,
                    paths,
                )

                results.append(result)
                passed += 1

            except Exception as exc:

                failed += 1

                print()
                print("=" * 100)
                print(
                    f"TEMPORAL SPLIT AUDIT: FAIL | "
                    f"{dataset_name.upper()} | "
                    f"{horizon.upper()}"
                )
                print("=" * 100)

                print(
                    f"ERROR: {exc}"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL AUDIT SUMMARY")
    print("=" * 100)

    print(
        f"Datasets passed : {passed}/9"
    )

    print(
        f"Datasets failed : {failed}/9"
    )

    if results:

        results_df = pd.DataFrame(
            results
        )

        print()

        print(
            results_df[
                [
                    "dataset",
                    "horizon",
                    "source_rows",
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                    "purged_rows",
                    "source_assets",
                    "train_assets",
                    "validation_assets",
                    "test_assets",
                    "validation_start",
                    "test_start",
                ]
            ].to_string(index=False)
        )

    print()

    if failed == 0:

        print(
            "TEMPORAL SPLIT AUDIT: PASS"
        )

        print(
            "All 9 temporal train/validation/test "
            "datasets passed the independent audit."
        )

        print()
        print(
            "The temporal split layer is ready to be LOCKED."
        )

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Baseline model training."
        )

    else:

        print(
            "TEMPORAL SPLIT AUDIT: FAIL"
        )

        print(
            "Do NOT lock the split datasets."
        )

        print(
            "Fix the failed dataset(s) before proceeding."
        )


if __name__ == "__main__":
    main()