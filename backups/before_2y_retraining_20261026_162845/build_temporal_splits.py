from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "stocks": {
        "6m": PROJECT_ROOT / "data/ml/stocks/stock_6m_ml.parquet",
        "1y": PROJECT_ROOT / "data/ml/stocks/stock_1y_ml.parquet",
        "2y": PROJECT_ROOT / "data/ml/stocks/stock_2y_ml.parquet",
    },
    "etfs": {
        "6m": PROJECT_ROOT / "data/ml/etfs/etf_6m_ml.parquet",
        "1y": PROJECT_ROOT / "data/ml/etfs/etf_1y_ml.parquet",
        "2y": PROJECT_ROOT / "data/ml/etfs/etf_2y_ml.parquet",
    },
    "mutual_funds": {
        "6m": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_6m_ml.parquet",
        "1y": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_1y_ml.parquet",
        "2y": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_ml.parquet",
    },
}


# ------------------------------------------------------------
# Locked temporal policies
# ------------------------------------------------------------

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
    """
    Normalize date columns without changing their calendar dates.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"]
    ).dt.normalize()

    return df


def get_boundary_date(dates, fraction):
    """
    Select the calendar date corresponding to the requested
    percentile of the unique observation dates.
    """

    unique_dates = np.sort(
        pd.Series(dates)
        .dropna()
        .unique()
    )

    if len(unique_dates) == 0:
        raise ValueError(
            "Dataset contains no valid observation dates."
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


def validate_input(df, dataset_name):
    """
    Validate the minimum schema required before splitting.
    """

    required_columns = {
        "ticker",
        "date",
        "label_end_date",
        "target",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name}: missing required columns: "
            f"{sorted(missing)}"
        )

    if df["ticker"].isna().any():
        raise ValueError(
            f"{dataset_name}: null ticker values found."
        )

    if df["date"].isna().any():
        raise ValueError(
            f"{dataset_name}: null date values found."
        )

    if df["label_end_date"].isna().any():
        raise ValueError(
            f"{dataset_name}: null label_end_date values found."
        )

    duplicate_count = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"{dataset_name}: "
            f"{duplicate_count:,} duplicate ticker/date keys."
        )

    if (df["label_end_date"] < df["date"]).any():
        raise ValueError(
            f"{dataset_name}: label_end_date occurs before date."
        )


def calculate_label_distribution(df):
    """
    Return target distribution as percentages.

    Targets are converted to strings so this works for both:
      numeric 0/1/2
      string '0'/'1'/'2'
    """

    if len(df) == 0:
        return {
            "0": 0.0,
            "1": 0.0,
            "2": 0.0,
        }

    distribution = (
        df["target"]
        .astype(str)
        .value_counts(normalize=True)
        * 100
    )

    return {
        "0": distribution.get("0", 0.0),
        "1": distribution.get("1", 0.0),
        "2": distribution.get("2", 0.0),
    }


def split_dataset(
    df,
    train_boundary_fraction,
    test_boundary_fraction,
):
    """
    Apply the strict temporal split.

    TRAIN:
        date < validation_start

    VALIDATION:
        validation_start <= date < test_start
        AND label_end_date < test_start

    TEST:
        date >= test_start

    Purged:
        rows that fall into the validation date interval but whose
        target window reaches into the test period.
    """

    if train_boundary_fraction >= test_boundary_fraction:
        raise ValueError(
            "Train boundary must be earlier than test boundary."
        )

    validation_start = get_boundary_date(
        df["date"],
        train_boundary_fraction,
    )

    test_start = get_boundary_date(
        df["date"],
        test_boundary_fraction,
    )

    if validation_start >= test_start:
        raise ValueError(
            "Computed validation_start is not earlier than test_start."
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_mask = (
        df["date"] < validation_start
    )

    # --------------------------------------------------------
    # VALIDATION
    #
    # Crucial leakage condition:
    # label_end_date must finish before test_start.
    # --------------------------------------------------------

    validation_mask = (
        (df["date"] >= validation_start)
        & (df["date"] < test_start)
        & (df["label_end_date"] < test_start)
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_mask = (
        df["date"] >= test_start
    )

    # --------------------------------------------------------
    # PURGED
    # --------------------------------------------------------

    assigned_mask = (
        train_mask
        | validation_mask
        | test_mask
    )

    purged_mask = ~assigned_mask

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()
    test = df.loc[test_mask].copy()
    purged = df.loc[purged_mask].copy()

    return (
        train,
        validation,
        test,
        purged,
        validation_start,
        test_start,
    )


def audit_split(
    original,
    train,
    validation,
    test,
    purged,
    validation_start,
    test_start,
):
    """
    Independent checks on the split.
    """

    errors = []

    # --------------------------------------------------------
    # Row accounting
    # --------------------------------------------------------

    total_partitioned = (
        len(train)
        + len(validation)
        + len(test)
        + len(purged)
    )

    if total_partitioned != len(original):
        errors.append(
            "Row accounting mismatch."
        )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    for name, df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        duplicates = df.duplicated(
            subset=["ticker", "date"]
        ).sum()

        if duplicates:
            errors.append(
                f"{name}: {duplicates:,} duplicate keys."
            )

    # --------------------------------------------------------
    # Cross-split key overlap
    # --------------------------------------------------------

    train_keys = set(
        zip(
            train["ticker"].astype(str),
            train["date"],
        )
    )

    validation_keys = set(
        zip(
            validation["ticker"].astype(str),
            validation["date"],
        )
    )

    test_keys = set(
        zip(
            test["ticker"].astype(str),
            test["date"],
        )
    )

    if train_keys & validation_keys:
        errors.append(
            "Train/validation key overlap detected."
        )

    if train_keys & test_keys:
        errors.append(
            "Train/test key overlap detected."
        )

    if validation_keys & test_keys:
        errors.append(
            "Validation/test key overlap detected."
        )

    # --------------------------------------------------------
    # TRAIN temporal rule
    # --------------------------------------------------------

    if len(train):
        if not (train["date"] < validation_start).all():
            errors.append(
                "Train contains observations on/after "
                "validation_start."
            )

    # --------------------------------------------------------
    # VALIDATION temporal rules
    # --------------------------------------------------------

    if len(validation):

        if not (
            (
                validation["date"]
                >= validation_start
            )
            & (
                validation["date"]
                < test_start
            )
        ).all():
            errors.append(
                "Validation contains invalid observation dates."
            )

        if not (
            validation["label_end_date"]
            < test_start
        ).all():
            errors.append(
                "Validation contains target windows "
                "crossing into test."
            )

    # --------------------------------------------------------
    # TEST temporal rule
    # --------------------------------------------------------

    if len(test):
        if not (
            test["date"] >= test_start
        ).all():
            errors.append(
                "Test contains observations before test_start."
            )

    # --------------------------------------------------------
    # Purged rows
    # --------------------------------------------------------

    if len(purged):

        expected_purge = (
            (purged["date"] >= validation_start)
            & (purged["date"] < test_start)
            & (purged["label_end_date"] >= test_start)
        )

        if not expected_purge.all():
            errors.append(
                "Purged rows do not satisfy the expected "
                "target-overlap condition."
            )

    return errors


def print_summary(
    dataset_name,
    horizon,
    original,
    train,
    validation,
    test,
    purged,
    validation_start,
    test_start,
):
    total = len(original)

    def pct(n):
        return (
            (n / total) * 100
            if total
            else 0.0
        )

    all_assets = set(
        original["ticker"].astype(str)
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

    train_labels = calculate_label_distribution(
        train
    )

    validation_labels = calculate_label_distribution(
        validation
    )

    test_labels = calculate_label_distribution(
        test
    )

    print()
    print("=" * 100)
    print(
        f"{dataset_name.upper()} | {horizon.upper()}"
    )
    print("=" * 100)

    print(
        f"Original rows       : {len(original):,}"
    )

    print(
        f"Date range          : "
        f"{original['date'].min().date()} -> "
        f"{original['date'].max().date()}"
    )

    print(
        f"Validation start    : "
        f"{validation_start.date()}"
    )

    print(
        f"Test start          : "
        f"{test_start.date()}"
    )

    print()
    print(
        f"Train               : "
        f"{len(train):,} "
        f"({pct(len(train)):.2f}%)"
    )

    print(
        f"Validation          : "
        f"{len(validation):,} "
        f"({pct(len(validation)):.2f}%)"
    )

    print(
        f"Test                : "
        f"{len(test):,} "
        f"({pct(len(test)):.2f}%)"
    )

    print(
        f"Purged              : "
        f"{len(purged):,} "
        f"({pct(len(purged)):.2f}%)"
    )

    print()
    print(
        f"Train assets        : "
        f"{len(train_assets)}/{len(all_assets)}"
    )

    print(
        f"Validation assets   : "
        f"{len(validation_assets)}/{len(all_assets)}"
    )

    print(
        f"Test assets         : "
        f"{len(test_assets)}/{len(all_assets)}"
    )

    print()
    print("Label distribution:")

    print(
        f"  Train      : "
        f"0={train_labels['0']:.2f}% | "
        f"1={train_labels['1']:.2f}% | "
        f"2={train_labels['2']:.2f}%"
    )

    print(
        f"  Validation : "
        f"0={validation_labels['0']:.2f}% | "
        f"1={validation_labels['1']:.2f}% | "
        f"2={validation_labels['2']:.2f}%"
    )

    print(
        f"  Test       : "
        f"0={test_labels['0']:.2f}% | "
        f"1={test_labels['1']:.2f}% | "
        f"2={test_labels['2']:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("BUILDING TEMPORAL TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 100)

    print()
    print("LOCKED SPLIT POLICY:")
    print()

    for horizon, policy in SPLIT_POLICY.items():
        print(
            f"  {horizon.upper():>3}: "
            f"train={policy['train_boundary']:.0%}, "
            f"test={policy['test_boundary']:.0%}"
        )

    print()
    print("Leakage protection:")
    print(
        "  Validation rows require "
        "label_end_date < test_start."
    )
    print()

    split_results = []

    for dataset_name, horizons in DATASETS.items():

        for horizon, input_path in horizons.items():

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input dataset not found:\n{input_path}"
                )

            print()
            print(
                f"Loading: {input_path}"
            )

            df = pd.read_parquet(
                input_path
            )

            df = normalize_dates(df)

            validate_input(
                df,
                f"{dataset_name}_{horizon}",
            )

            df = df.sort_values(
                ["date", "ticker"]
            ).reset_index(drop=True)

            policy = SPLIT_POLICY[horizon]

            (
                train,
                validation,
                test,
                purged,
                validation_start,
                test_start,
            ) = split_dataset(
                df,
                policy["train_boundary"],
                policy["test_boundary"],
            )

            errors = audit_split(
                df,
                train,
                validation,
                test,
                purged,
                validation_start,
                test_start,
            )

            if errors:
                print()
                print(
                    "SPLIT AUDIT FAILED:"
                )

                for error in errors:
                    print(
                        f"  - {error}"
                    )

                raise RuntimeError(
                    f"Split audit failed for "
                    f"{dataset_name}_{horizon}."
                )

            print_summary(
                dataset_name,
                horizon,
                df,
                train,
                validation,
                test,
                purged,
                validation_start,
                test_start,
            )

            # ------------------------------------------------
            # Output paths
            # ------------------------------------------------

            output_dir = (
                PROJECT_ROOT
                / "data"
                / "ml"
                / dataset_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            train_path = (
                output_dir
                / f"{dataset_name[:-1] if dataset_name.endswith('s') else dataset_name}_{horizon}_train.parquet"
            )

            validation_path = (
                output_dir
                / f"{dataset_name[:-1] if dataset_name.endswith('s') else dataset_name}_{horizon}_validation.parquet"
            )

            test_path = (
                output_dir
                / f"{dataset_name[:-1] if dataset_name.endswith('s') else dataset_name}_{horizon}_test.parquet"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            train.to_parquet(
                train_path,
                index=False,
            )

            validation.to_parquet(
                validation_path,
                index=False,
            )

            test.to_parquet(
                test_path,
                index=False,
            )

            split_results.append(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "original_rows": len(df),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "purged_rows": len(purged),
                    "validation_start": validation_start,
                    "test_start": test_start,
                    "train_assets": train["ticker"].nunique(),
                    "validation_assets": validation["ticker"].nunique(),
                    "test_assets": test["ticker"].nunique(),
                    "train_path": str(train_path),
                    "validation_path": str(validation_path),
                    "test_path": str(test_path),
                }
            )

            print()
            print(
                "SPLIT AUDIT: PASS"
            )

            print(
                f"Saved train      : {train_path}"
            )

            print(
                f"Saved validation : {validation_path}"
            )

            print(
                f"Saved test       : {test_path}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        split_results
    )

    print()
    print("=" * 100)
    print("FINAL SPLIT SUMMARY")
    print("=" * 100)

    print(
        results_df[
            [
                "dataset",
                "horizon",
                "original_rows",
                "train_rows",
                "validation_rows",
                "test_rows",
                "purged_rows",
                "train_assets",
                "validation_assets",
                "test_assets",
                "validation_start",
                "test_start",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "TEMPORAL SPLIT BUILD: PASS"
    )

    print(
        "All 9 ML datasets were split successfully."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Run the independent temporal split audit "
        "before locking the split datasets."
    )


if __name__ == "__main__":
    main()