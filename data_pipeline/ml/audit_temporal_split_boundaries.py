from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ML_DIR = BASE_DIR / "data" / "ml"

DATASETS = {
    "stocks_6m": ML_DIR / "stocks" / "stock_6m_ml.parquet",
    "stocks_1y": ML_DIR / "stocks" / "stock_1y_ml.parquet",
    "stocks_2y": ML_DIR / "stocks" / "stock_2y_ml.parquet",

    "etfs_6m": ML_DIR / "etfs" / "etf_6m_ml.parquet",
    "etfs_1y": ML_DIR / "etfs" / "etf_1y_ml.parquet",
    "etfs_2y": ML_DIR / "etfs" / "etf_2y_ml.parquet",

    "mutual_funds_6m": ML_DIR / "mutual_funds" / "mutual_fund_6m_ml.parquet",
    "mutual_funds_1y": ML_DIR / "mutual_funds" / "mutual_fund_1y_ml.parquet",
    "mutual_funds_2y": ML_DIR / "mutual_funds" / "mutual_fund_2y_ml.parquet",
}


# ============================================================
# HELPERS
# ============================================================

def percentile_date(series, percentile):
    """
    Return the date at a specified percentile of the
    observation-date distribution.
    """
    values = series.sort_values().reset_index(drop=True)

    if len(values) == 0:
        return None

    position = int(np.floor(percentile * (len(values) - 1)))

    return values.iloc[position]


def print_separator():
    print("=" * 90)


def analyze_dataset(name, path):
    print_separator()
    print(f"DATASET: {name}")
    print(f"FILE:    {path}")

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        return None

    df = pd.read_parquet(path)

    required_columns = [
        "ticker",
        "date",
        "label_end_date",
        "target",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        print(f"STATUS: MISSING REQUIRED COLUMNS: {missing}")
        return None

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"],
        errors="coerce"
    )

    # Remove timezone if present.
    if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    if (
        hasattr(df["label_end_date"].dt, "tz")
        and df["label_end_date"].dt.tz is not None
    ):
        df["label_end_date"] = df["label_end_date"].dt.tz_localize(None)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    total_rows = len(df)
    assets = df["ticker"].nunique()

    min_date = df["date"].min()
    max_date = df["date"].max()

    min_label_end = df["label_end_date"].min()
    max_label_end = df["label_end_date"].max()

    print()
    print("BASIC DATASET INFORMATION")
    print(f"Rows:                 {total_rows:,}")
    print(f"Assets:               {assets:,}")
    print(f"Observation start:    {min_date.date()}")
    print(f"Observation end:      {max_date.date()}")
    print(f"Label-end start:      {min_label_end.date()}")
    print(f"Label-end end:        {max_label_end.date()}")

    # --------------------------------------------------------
    # Candidate chronological boundaries
    #
    # 60% train
    # 20% validation
    # 20% test
    #
    # IMPORTANT:
    # These are based on observation-date distribution only.
    # They are NOT final boundaries.
    # --------------------------------------------------------

    train_candidate = percentile_date(df["date"], 0.60)
    test_candidate = percentile_date(df["date"], 0.80)

    print()
    print("CANDIDATE CHRONOLOGICAL BOUNDARIES")
    print(f"60% observation-date boundary: {train_candidate.date()}")
    print(f"80% observation-date boundary: {test_candidate.date()}")

    # --------------------------------------------------------
    # Candidate split counts BEFORE PURGING
    # --------------------------------------------------------

    train_candidate_mask = df["date"] < train_candidate

    validation_candidate_mask = (
        (df["date"] >= train_candidate)
        & (df["date"] < test_candidate)
    )

    test_candidate_mask = df["date"] >= test_candidate

    print()
    print("CANDIDATE SPLIT COUNTS BEFORE PURGING")

    print(
        f"Train:       {train_candidate_mask.sum():,} "
        f"({train_candidate_mask.mean() * 100:.2f}%)"
    )

    print(
        f"Validation:  {validation_candidate_mask.sum():,} "
        f"({validation_candidate_mask.mean() * 100:.2f}%)"
    )

    print(
        f"Test:        {test_candidate_mask.sum():,} "
        f"({test_candidate_mask.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # PURGE ANALYSIS
    #
    # Training observation must have:
    #
    # label_end_date < validation_start
    #
    # Validation observation must have:
    #
    # label_end_date < test_start
    #
    # Otherwise its future target period overlaps the
    # following split.
    # --------------------------------------------------------

    train_cross_boundary = train_candidate_mask & (
        df["label_end_date"] >= train_candidate
    )

    validation_cross_boundary = validation_candidate_mask & (
        df["label_end_date"] >= test_candidate
    )

    print()
    print("PURGE ANALYSIS")

    print(
        f"Train rows crossing validation boundary: "
        f"{train_cross_boundary.sum():,}"
    )

    print(
        f"Validation rows crossing test boundary: "
        f"{validation_cross_boundary.sum():,}"
    )

    # --------------------------------------------------------
    # SAFE COUNTS USING CANDIDATE BOUNDARIES
    # --------------------------------------------------------

    safe_train_mask = (
        (df["date"] < train_candidate)
        & (df["label_end_date"] < train_candidate)
    )

    safe_validation_mask = (
        (df["date"] >= train_candidate)
        & (df["date"] < test_candidate)
        & (df["label_end_date"] < test_candidate)
    )

    safe_test_mask = (
        df["date"] >= test_candidate
    )

    print()
    print("SAFE SPLIT COUNTS AFTER PURGING")

    print(
        f"Train:       {safe_train_mask.sum():,} "
        f"({safe_train_mask.mean() * 100:.2f}%)"
    )

    print(
        f"Validation:  {safe_validation_mask.sum():,} "
        f"({safe_validation_mask.mean() * 100:.2f}%)"
    )

    print(
        f"Test:        {safe_test_mask.sum():,} "
        f"({safe_test_mask.mean() * 100:.2f}%)"
    )

    purged_total = (
        total_rows
        - safe_train_mask.sum()
        - safe_validation_mask.sum()
        - safe_test_mask.sum()
    )

    print(
        f"Total purged/unused: {purged_total:,} "
        f"({purged_total / total_rows * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # ACTUAL DATE RANGES AFTER PURGING
    # --------------------------------------------------------

    print()
    print("SAFE SPLIT DATE RANGES")

    train_rows = df.loc[safe_train_mask]

    validation_rows = df.loc[safe_validation_mask]

    test_rows = df.loc[safe_test_mask]

    if len(train_rows) > 0:
        print(
            f"Train observation dates:      "
            f"{train_rows['date'].min().date()} → "
            f"{train_rows['date'].max().date()}"
        )

        print(
            f"Train label-end dates:         "
            f"{train_rows['label_end_date'].min().date()} → "
            f"{train_rows['label_end_date'].max().date()}"
        )
    else:
        print("Train: EMPTY")

    if len(validation_rows) > 0:
        print(
            f"Validation observation dates:  "
            f"{validation_rows['date'].min().date()} → "
            f"{validation_rows['date'].max().date()}"
        )

        print(
            f"Validation label-end dates:    "
            f"{validation_rows['label_end_date'].min().date()} → "
            f"{validation_rows['label_end_date'].max().date()}"
        )
    else:
        print("Validation: EMPTY")

    if len(test_rows) > 0:
        print(
            f"Test observation dates:        "
            f"{test_rows['date'].min().date()} → "
            f"{test_rows['date'].max().date()}"
        )

        print(
            f"Test label-end dates:          "
            f"{test_rows['label_end_date'].min().date()} → "
            f"{test_rows['label_end_date'].max().date()}"
        )
    else:
        print("Test: EMPTY")

    # --------------------------------------------------------
    # ASSET COVERAGE
    # --------------------------------------------------------

    print()
    print("ASSET COVERAGE")

    train_assets = train_rows["ticker"].nunique()
    validation_assets = validation_rows["ticker"].nunique()
    test_assets = test_rows["ticker"].nunique()

    print(f"Train assets:       {train_assets:,}")
    print(f"Validation assets:  {validation_assets:,}")
    print(f"Test assets:        {test_assets:,}")

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("LABEL DISTRIBUTION")

    for split_name, split_df in [
        ("Train", train_rows),
        ("Validation", validation_rows),
        ("Test", test_rows),
    ]:

        print(f"\n{split_name}:")

        if len(split_df) == 0:
            print("  EMPTY")
            continue

        distribution = (
            split_df["target"]
            .astype(str)
            .value_counts(normalize=False)
            .sort_index()
        )

        percentages = (
            split_df["target"]
            .astype(str)
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )

        for label in distribution.index:
            print(
                f"  Target {label}: "
                f"{distribution[label]:,} "
                f"({percentages[label]:.2f}%)"
            )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    print()
    print("LEAKAGE CHECK")

    train_max_label_end = (
        train_rows["label_end_date"].max()
        if len(train_rows) > 0
        else None
    )

    validation_max_label_end = (
        validation_rows["label_end_date"].max()
        if len(validation_rows) > 0
        else None
    )

    train_safe = (
        train_max_label_end < train_candidate
        if train_max_label_end is not None
        else False
    )

    validation_safe = (
        validation_max_label_end < test_candidate
        if validation_max_label_end is not None
        else False
    )

    print(
        f"Train target windows end before validation: "
        f"{'PASS' if train_safe else 'FAIL'}"
    )

    print(
        f"Validation target windows end before test: "
        f"{'PASS' if validation_safe else 'FAIL'}"
    )

    # --------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------

    return {
        "dataset": name,
        "rows": total_rows,
        "assets": assets,
        "observation_start": min_date,
        "observation_end": max_date,
        "label_end_start": min_label_end,
        "label_end_end": max_label_end,
        "candidate_train_start": min_date,
        "candidate_validation_start": train_candidate,
        "candidate_test_start": test_candidate,
        "train_before_purge": train_candidate_mask.sum(),
        "validation_before_purge": validation_candidate_mask.sum(),
        "test_before_purge": test_candidate_mask.sum(),
        "train_purged": train_cross_boundary.sum(),
        "validation_purged": validation_cross_boundary.sum(),
        "train_after_purge": safe_train_mask.sum(),
        "validation_after_purge": safe_validation_mask.sum(),
        "test_after_purge": safe_test_mask.sum(),
        "total_purged": purged_total,
        "train_assets": train_assets,
        "validation_assets": validation_assets,
        "test_assets": test_assets,
        "train_leakage_check": train_safe,
        "validation_leakage_check": validation_safe,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_separator()
    print("TEMPORAL SPLIT BOUNDARY DIAGNOSTIC")
    print_separator()

    print()
    print("Purpose:")
    print("Calculate chronological 60/20/20 candidate boundaries")
    print("and determine the amount of purging required to prevent")
    print("future-target overlap between train, validation and test.")
    print()
    print("IMPORTANT:")
    print("No datasets will be modified.")
    print("No split files will be created.")
    print()

    results = []

    for name, path in DATASETS.items():

        result = analyze_dataset(name, path)

        if result is not None:
            results.append(result)

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    print()
    print_separator()
    print("SUMMARY")
    print_separator()

    if not results:
        print("No datasets were successfully analyzed.")
        return

    summary = pd.DataFrame(results)

    display_columns = [
        "dataset",
        "rows",
        "assets",
        "candidate_validation_start",
        "candidate_test_start",
        "train_purged",
        "validation_purged",
        "train_after_purge",
        "validation_after_purge",
        "test_after_purge",
        "total_purged",
    ]

    summary_display = summary[display_columns].copy()

    for column in [
        "candidate_validation_start",
        "candidate_test_start",
    ]:
        summary_display[column] = (
            summary_display[column]
            .dt.strftime("%Y-%m-%d")
        )

    print(summary_display.to_string(index=False))

    # ========================================================
    # OVERALL CHECK
    # ========================================================

    print()
    print_separator()
    print("BOUNDARY DIAGNOSTIC COMPLETE")
    print_separator()

    leakage_pass = (
        summary["train_leakage_check"].all()
        and summary["validation_leakage_check"].all()
    )

    if leakage_pass:
        print("Candidate purging logic: PASS")
    else:
        print("Candidate purging logic: REVIEW REQUIRED")

    print()
    print("NEXT STEP:")
    print(
        "Review this output before creating the final temporal "
        "train/validation/test split files."
    )
    print()


if __name__ == "__main__":
    main()