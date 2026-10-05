from pathlib import Path

import numpy as np
import pandas as pd


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
# SEARCH CONFIGURATION
# ============================================================

# Candidate split proportions based on observation dates.
#
# These are NOT final splits.
#
# We search a broad range so that 2Y datasets can have a
# sufficiently large validation period after purging.

TRAIN_BOUNDARIES = [
    0.45,
    0.50,
    0.55,
    0.60,
]

TEST_BOUNDARIES = [
    0.75,
    0.80,
    0.85,
    0.90,
]

# Minimum acceptable fraction of the ORIGINAL dataset.
MIN_TRAIN_FRACTION = 0.25
MIN_VALIDATION_FRACTION = 0.10
MIN_TEST_FRACTION = 0.10

# Minimum asset coverage relative to the full dataset.
MIN_TRAIN_ASSET_COVERAGE = 0.50
MIN_VALIDATION_ASSET_COVERAGE = 0.50
MIN_TEST_ASSET_COVERAGE = 0.80


# ============================================================
# HELPERS
# ============================================================

def get_percentile_date(date_series, percentile):
    """
    Return the observation date corresponding to a percentile
    of the sorted observation dates.
    """
    dates = date_series.sort_values().reset_index(drop=True)

    position = int(np.floor(percentile * (len(dates) - 1)))

    return dates.iloc[position]


def normalize_dates(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"],
        errors="coerce",
    )

    # Remove timezone if present.
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    if df["label_end_date"].dt.tz is not None:
        df["label_end_date"] = df["label_end_date"].dt.tz_localize(None)

    return df


def evaluate_candidate(
    df,
    train_boundary_date,
    test_boundary_date,
):
    """
    Evaluate one candidate pair of chronological boundaries.

    Leakage-safe definitions:

    TRAIN:
        date < train_boundary
        label_end_date < train_boundary

    VALIDATION:
        train_boundary <= date < test_boundary
        label_end_date < test_boundary

    TEST:
        date >= test_boundary

    Rows that fail these conditions are purged.
    """

    train_mask = (
        (df["date"] < train_boundary_date)
        & (df["label_end_date"] < train_boundary_date)
    )

    validation_mask = (
        (df["date"] >= train_boundary_date)
        & (df["date"] < test_boundary_date)
        & (df["label_end_date"] < test_boundary_date)
    )

    test_mask = df["date"] >= test_boundary_date

    train = df.loc[train_mask]
    validation = df.loc[validation_mask]
    test = df.loc[test_mask]

    total = len(df)

    train_fraction = len(train) / total
    validation_fraction = len(validation) / total
    test_fraction = len(test) / total

    purged = total - len(train) - len(validation) - len(test)
    purged_fraction = purged / total

    total_assets = df["ticker"].nunique()

    train_assets = train["ticker"].nunique()
    validation_assets = validation["ticker"].nunique()
    test_assets = test["ticker"].nunique()

    train_asset_coverage = (
        train_assets / total_assets
        if total_assets > 0
        else 0
    )

    validation_asset_coverage = (
        validation_assets / total_assets
        if total_assets > 0
        else 0
    )

    test_asset_coverage = (
        test_assets / total_assets
        if total_assets > 0
        else 0
    )

    # --------------------------------------------------------
    # Label distribution imbalance
    #
    # We measure the largest absolute difference between each
    # split's label percentage and the full dataset percentage.
    # --------------------------------------------------------

    full_distribution = (
        df["target"]
        .astype(str)
        .value_counts(normalize=True)
    )

    def max_label_deviation(split):
        if len(split) == 0:
            return 1.0

        split_distribution = (
            split["target"]
            .astype(str)
            .value_counts(normalize=True)
        )

        labels = set(full_distribution.index) | set(
            split_distribution.index
        )

        deviations = []

        for label in labels:
            full_value = full_distribution.get(label, 0.0)
            split_value = split_distribution.get(label, 0.0)

            deviations.append(
                abs(split_value - full_value)
            )

        return max(deviations)

    train_label_deviation = max_label_deviation(train)
    validation_label_deviation = max_label_deviation(validation)
    test_label_deviation = max_label_deviation(test)

    max_label_deviation_all = max(
        train_label_deviation,
        validation_label_deviation,
        test_label_deviation,
    )

    # --------------------------------------------------------
    # Hard feasibility checks
    # --------------------------------------------------------

    train_ok = (
        train_fraction >= MIN_TRAIN_FRACTION
        and train_assets / total_assets
        >= MIN_TRAIN_ASSET_COVERAGE
    )

    validation_ok = (
        validation_fraction >= MIN_VALIDATION_FRACTION
        and validation_assets / total_assets
        >= MIN_VALIDATION_ASSET_COVERAGE
    )

    test_ok = (
        test_fraction >= MIN_TEST_FRACTION
        and test_assets / total_assets
        >= MIN_TEST_ASSET_COVERAGE
    )

    feasible = train_ok and validation_ok and test_ok

    # --------------------------------------------------------
    # Score candidate
    #
    # Higher score is better.
    #
    # We prefer:
    # - more train data
    # - more validation data
    # - more test data
    # - less purging
    # - better asset coverage
    # - balanced labels
    # --------------------------------------------------------

    if feasible:
        score = (
            2.0 * train_fraction
            + 2.0 * validation_fraction
            + 1.5 * test_fraction
            + 1.0 * train_asset_coverage
            + 1.0 * validation_asset_coverage
            + 1.0 * test_asset_coverage
            - 2.0 * purged_fraction
            - 1.0 * max_label_deviation_all
        )
    else:
        score = -999.0

    return {
        "train_boundary": train_boundary_date,
        "test_boundary": test_boundary_date,

        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),

        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,

        "purged_rows": purged,
        "purged_fraction": purged_fraction,

        "train_assets": train_assets,
        "validation_assets": validation_assets,
        "test_assets": test_assets,

        "train_asset_coverage": train_asset_coverage,
        "validation_asset_coverage": validation_asset_coverage,
        "test_asset_coverage": test_asset_coverage,

        "train_label_deviation": train_label_deviation,
        "validation_label_deviation": validation_label_deviation,
        "test_label_deviation": test_label_deviation,
        "max_label_deviation": max_label_deviation_all,

        "feasible": feasible,
        "score": score,
    }


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(name, path):

    print()
    print("=" * 100)
    print(f"DATASET: {name}")
    print("=" * 100)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_parquet(path)

    required = [
        "ticker",
        "date",
        "label_end_date",
        "target",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    df = normalize_dates(df)

    df = df.dropna(
        subset=[
            "ticker",
            "date",
            "label_end_date",
            "target",
        ]
    )

    print()
    print("DATASET INFORMATION")
    print(f"Rows:             {len(df):,}")
    print(f"Assets:           {df['ticker'].nunique():,}")
    print(f"Observation:      {df['date'].min().date()} → {df['date'].max().date()}")
    print(
        f"Label end:        "
        f"{df['label_end_date'].min().date()} → "
        f"{df['label_end_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Candidate dates
    # --------------------------------------------------------

    train_candidates = [
        get_percentile_date(df["date"], p)
        for p in TRAIN_BOUNDARIES
    ]

    test_candidates = [
        get_percentile_date(df["date"], p)
        for p in TEST_BOUNDARIES
    ]

    print()
    print("SEARCHING CANDIDATE BOUNDARIES...")
    print(
        f"Train percentiles: {TRAIN_BOUNDARIES}"
    )
    print(
        f"Test percentiles:  {TEST_BOUNDARIES}"
    )

    results = []

    for train_date in train_candidates:

        for test_date in test_candidates:

            # Test boundary must occur after train boundary.
            if test_date <= train_date:
                continue

            result = evaluate_candidate(
                df,
                train_date,
                test_date,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No candidates generated.")
        return

    feasible = results_df[
        results_df["feasible"]
    ].copy()

    # --------------------------------------------------------
    # Print all candidates
    # --------------------------------------------------------

    print()
    print("ALL CANDIDATES")
    print("-" * 100)

    display_columns = [
        "train_boundary",
        "test_boundary",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "purged_fraction",
        "train_assets",
        "validation_assets",
        "test_assets",
        "max_label_deviation",
        "feasible",
        "score",
    ]

    display_df = results_df[display_columns].copy()

    display_df["train_boundary"] = (
        display_df["train_boundary"]
        .dt.strftime("%Y-%m-%d")
    )

    display_df["test_boundary"] = (
        display_df["test_boundary"]
        .dt.strftime("%Y-%m-%d")
    )

    for column in [
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "purged_fraction",
        "max_label_deviation",
    ]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    print(
        display_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # Feasible candidates
    # --------------------------------------------------------

    print()
    print("FEASIBLE CANDIDATES")
    print("-" * 100)

    if feasible.empty:
        print(
            "NO FEASIBLE CANDIDATE FOUND "
            "under the current minimum requirements."
        )

        print()
        print("This dataset requires a wider search.")
        return

    feasible = feasible.sort_values(
        "score",
        ascending=False,
    )

    top = feasible.head(5).copy()

    top_display = top[display_columns].copy()

    top_display["train_boundary"] = (
        top_display["train_boundary"]
        .dt.strftime("%Y-%m-%d")
    )

    top_display["test_boundary"] = (
        top_display["test_boundary"]
        .dt.strftime("%Y-%m-%d")
    )

    for column in [
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "purged_fraction",
        "max_label_deviation",
    ]:
        top_display[column] = (
            top_display[column] * 100
        ).round(2)

    print(
        top_display.to_string(index=False)
    )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best = feasible.iloc[0]

    print()
    print("BEST CANDIDATE")
    print("-" * 100)

    print(
        f"Train boundary:       "
        f"{best['train_boundary'].date()}"
    )

    print(
        f"Test boundary:        "
        f"{best['test_boundary'].date()}"
    )

    print(
        f"Train rows:            "
        f"{int(best['train_rows']):,} "
        f"({best['train_fraction'] * 100:.2f}%)"
    )

    print(
        f"Validation rows:       "
        f"{int(best['validation_rows']):,} "
        f"({best['validation_fraction'] * 100:.2f}%)"
    )

    print(
        f"Test rows:             "
        f"{int(best['test_rows']):,} "
        f"({best['test_fraction'] * 100:.2f}%)"
    )

    print(
        f"Purged rows:           "
        f"{int(best['purged_rows']):,} "
        f"({best['purged_fraction'] * 100:.2f}%)"
    )

    print(
        f"Train assets:          "
        f"{int(best['train_assets'])}"
    )

    print(
        f"Validation assets:     "
        f"{int(best['validation_assets'])}"
    )

    print(
        f"Test assets:           "
        f"{int(best['test_assets'])}"
    )

    print(
        f"Maximum label deviation: "
        f"{best['max_label_deviation'] * 100:.2f} percentage points"
    )

    print(
        f"Candidate score:       "
        f"{best['score']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("TEMPORAL SPLIT CANDIDATE SEARCH")
    print("=" * 100)

    print()
    print("This script:")
    print("1. Does NOT modify any ML dataset.")
    print("2. Does NOT create train/validation/test files.")
    print("3. Searches multiple chronological boundary combinations.")
    print("4. Purges observations whose target windows cross boundaries.")
    print("5. Reports feasible candidates.")
    print()

    for name, path in DATASETS.items():
        analyze_dataset(name, path)

    print()
    print("=" * 100)
    print("TEMPORAL SPLIT CANDIDATE SEARCH COMPLETE")
    print("=" * 100)

    print()
    print("NEXT STEP:")
    print(
        "Review the candidate boundaries before creating "
        "the final temporal split files."
    )


if __name__ == "__main__":
    main()