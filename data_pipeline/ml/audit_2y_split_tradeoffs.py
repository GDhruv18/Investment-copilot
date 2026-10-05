from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "stocks_2y": PROJECT_ROOT / "data/ml/stocks/stock_2y_ml.parquet",
    "etfs_2y": PROJECT_ROOT / "data/ml/etfs/etf_2y_ml.parquet",
    "mutual_funds_2y": PROJECT_ROOT / "data/ml/mutual_funds/mutual_fund_2y_ml.parquet",
}

TRAIN_BOUNDARIES = [0.30, 0.35, 0.40, 0.45, 0.50]
TEST_BOUNDARIES = [0.80, 0.85, 0.90, 0.925, 0.95]

MIN_TRAIN_FRACTION = 0.20
MIN_VALIDATION_FRACTION = 0.05
MIN_TEST_FRACTION = 0.05

MIN_TRAIN_ASSET_COVERAGE = 0.30
MIN_VALIDATION_ASSET_COVERAGE = 0.30
MIN_TEST_ASSET_COVERAGE = 0.80


# ============================================================
# HELPERS
# ============================================================

def normalize_dates(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["label_end_date"] = pd.to_datetime(df["label_end_date"]).dt.normalize()

    return df


def get_date_boundary(dates, fraction):
    """
    Select the calendar date at the requested observation-date fraction.
    """
    unique_dates = np.sort(pd.Series(dates).dropna().unique())

    if len(unique_dates) == 0:
        return None

    index = int(round((len(unique_dates) - 1) * fraction))
    index = max(0, min(index, len(unique_dates) - 1))

    return pd.Timestamp(unique_dates[index])


def calculate_label_balance(df):
    if len(df) == 0:
        return {
            "weak_pct": np.nan,
            "neutral_pct": np.nan,
            "attractive_pct": np.nan,
            "max_deviation_pp": np.nan,
        }

    counts = df["target"].astype(str).value_counts(normalize=True)

    weak = counts.get("0", 0.0) * 100
    neutral = counts.get("1", 0.0) * 100
    attractive = counts.get("2", 0.0) * 100

    deviations = [
        abs(weak - 30.0),
        abs(neutral - 40.0),
        abs(attractive - 30.0),
    ]

    return {
        "weak_pct": weak,
        "neutral_pct": neutral,
        "attractive_pct": attractive,
        "max_deviation_pp": max(deviations),
    }


def evaluate_candidate(
    df,
    train_fraction,
    test_fraction,
):
    """
    Strict temporal split:

        TRAIN:
            date < validation_start

        VALIDATION:
            validation_start <= date < test_start
            AND label_end_date < test_start

        TEST:
            date >= test_start

    Any row whose target window crosses a split boundary is purged.
    """

    unique_dates = np.sort(df["date"].dropna().unique())

    if len(unique_dates) < 10:
        return None

    validation_start = get_date_boundary(unique_dates, train_fraction)
    test_start = get_date_boundary(unique_dates, test_fraction)

    if validation_start is None or test_start is None:
        return None

    if validation_start >= test_start:
        return None

    train_mask = df["date"] < validation_start

    validation_mask = (
        (df["date"] >= validation_start)
        & (df["date"] < test_start)
        & (df["label_end_date"] < test_start)
    )

    test_mask = df["date"] >= test_start

    assigned_mask = train_mask | validation_mask | test_mask

    purged_mask = ~assigned_mask

    train = df.loc[train_mask]
    validation = df.loc[validation_mask]
    test = df.loc[test_mask]
    purged = df.loc[purged_mask]

    total = len(df)

    train_fraction_actual = len(train) / total if total else 0
    validation_fraction_actual = len(validation) / total if total else 0
    test_fraction_actual = len(test) / total if total else 0
    purged_fraction_actual = len(purged) / total if total else 0

    all_assets = set(df["ticker"].astype(str).unique())

    train_assets = set(train["ticker"].astype(str).unique())
    validation_assets = set(validation["ticker"].astype(str).unique())
    test_assets = set(test["ticker"].astype(str).unique())

    train_asset_coverage = (
        len(train_assets) / len(all_assets)
        if all_assets
        else 0
    )

    validation_asset_coverage = (
        len(validation_assets) / len(all_assets)
        if all_assets
        else 0
    )

    test_asset_coverage = (
        len(test_assets) / len(all_assets)
        if all_assets
        else 0
    )

    train_balance = calculate_label_balance(train)
    validation_balance = calculate_label_balance(validation)
    test_balance = calculate_label_balance(test)

    return {
        "train_fraction_requested": train_fraction,
        "test_fraction_requested": test_fraction,

        "validation_start": validation_start,
        "test_start": test_start,

        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "purged_rows": len(purged),

        "train_pct": train_fraction_actual * 100,
        "validation_pct": validation_fraction_actual * 100,
        "test_pct": test_fraction_actual * 100,
        "purged_pct": purged_fraction_actual * 100,

        "total_assets": len(all_assets),

        "train_assets": len(train_assets),
        "validation_assets": len(validation_assets),
        "test_assets": len(test_assets),

        "train_asset_coverage_pct": train_asset_coverage * 100,
        "validation_asset_coverage_pct": validation_asset_coverage * 100,
        "test_asset_coverage_pct": test_asset_coverage * 100,

        "train_max_label_deviation_pp": train_balance["max_deviation_pp"],
        "validation_max_label_deviation_pp": validation_balance[
            "max_deviation_pp"
        ],
        "test_max_label_deviation_pp": test_balance[
            "max_deviation_pp"
        ],

        "train_weak_pct": train_balance["weak_pct"],
        "train_neutral_pct": train_balance["neutral_pct"],
        "train_attractive_pct": train_balance["attractive_pct"],

        "validation_weak_pct": validation_balance["weak_pct"],
        "validation_neutral_pct": validation_balance["neutral_pct"],
        "validation_attractive_pct": validation_balance["attractive_pct"],

        "test_weak_pct": test_balance["weak_pct"],
        "test_neutral_pct": test_balance["neutral_pct"],
        "test_attractive_pct": test_balance["attractive_pct"],
    }


def is_feasible(result):
    if result is None:
        return False

    return (
        result["train_pct"] / 100 >= MIN_TRAIN_FRACTION
        and result["validation_pct"] / 100 >= MIN_VALIDATION_FRACTION
        and result["test_pct"] / 100 >= MIN_TEST_FRACTION
        and result["train_asset_coverage_pct"] / 100
        >= MIN_TRAIN_ASSET_COVERAGE
        and result["validation_asset_coverage_pct"] / 100
        >= MIN_VALIDATION_ASSET_COVERAGE
        and result["test_asset_coverage_pct"] / 100
        >= MIN_TEST_ASSET_COVERAGE
    )


def calculate_score(result):
    """
    Diagnostic score only.

    This is NOT an optimization objective for the eventual model.

    Higher is better:
      - more train data
      - more validation data
      - more test data
      - better asset coverage
      - lower purge
      - better label balance
    """

    if result is None:
        return -np.inf

    size_score = (
        0.30 * result["train_pct"]
        + 0.20 * result["validation_pct"]
        + 0.10 * result["test_pct"]
    )

    coverage_score = (
        0.10 * result["train_asset_coverage_pct"]
        + 0.10 * result["validation_asset_coverage_pct"]
        + 0.10 * result["test_asset_coverage_pct"]
    )

    purge_penalty = 0.10 * result["purged_pct"]

    balance_deviation = max(
        result["train_max_label_deviation_pp"],
        result["validation_max_label_deviation_pp"],
        result["test_max_label_deviation_pp"],
    )

    balance_penalty = 0.10 * balance_deviation

    return (
        size_score
        + coverage_score
        - purge_penalty
        - balance_penalty
    )


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    print("=" * 100)
    print("2Y TEMPORAL SPLIT TRADE-OFF DIAGNOSTIC")
    print("=" * 100)

    print()
    print("Candidate train boundaries:", TRAIN_BOUNDARIES)
    print("Candidate test boundaries :", TEST_BOUNDARIES)

    print()
    print("Strict leakage rule:")
    print("  TRAIN      : date < validation_start")
    print("  VALIDATION : validation_start <= date < test_start")
    print("               AND label_end_date < test_start")
    print("  TEST       : date >= test_start")
    print()

    for dataset_name, path in DATASETS.items():

        print("\n" + "=" * 100)
        print(dataset_name.upper())
        print("=" * 100)

        if not path.exists():
            print(f"ERROR: Dataset not found: {path}")
            continue

        df = pd.read_parquet(path)
        df = normalize_dates(df)

        required = {
            "ticker",
            "date",
            "label_end_date",
            "target",
        }

        missing = required - set(df.columns)

        if missing:
            print(f"ERROR: Missing required columns: {sorted(missing)}")
            continue

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        print(f"Rows        : {len(df):,}")
        print(f"Assets      : {df['ticker'].nunique():,}")
        print(f"Date range  : {df['date'].min().date()} -> "
              f"{df['date'].max().date()}")
        print(f"Target range: {df['label_end_date'].min().date()} -> "
              f"{df['label_end_date'].max().date()}")

        dataset_results = []

        for train_fraction in TRAIN_BOUNDARIES:
            for test_fraction in TEST_BOUNDARIES:

                result = evaluate_candidate(
                    df,
                    train_fraction,
                    test_fraction,
                )

                if result is None:
                    continue

                result["dataset"] = dataset_name
                result["feasible"] = is_feasible(result)
                result["score"] = calculate_score(result)

                dataset_results.append(result)
                all_results.append(result)

        result_df = pd.DataFrame(dataset_results)

        if result_df.empty:
            print("No valid candidates found.")
            continue

        feasible_df = result_df[result_df["feasible"]].copy()

        print()
        print(f"Total candidates evaluated : {len(result_df)}")
        print(f"Feasible candidates        : {len(feasible_df)}")

        # ----------------------------------------------------
        # Best feasible candidates
        # ----------------------------------------------------

        if not feasible_df.empty:

            best = (
                feasible_df
                .sort_values("score", ascending=False)
                .head(10)
            )

            print()
            print("-" * 100)
            print("TOP 10 FEASIBLE CANDIDATES")
            print("-" * 100)

            display_columns = [
                "train_fraction_requested",
                "test_fraction_requested",
                "validation_start",
                "test_start",
                "train_pct",
                "validation_pct",
                "test_pct",
                "purged_pct",
                "train_assets",
                "validation_assets",
                "test_assets",
                "train_max_label_deviation_pp",
                "validation_max_label_deviation_pp",
                "test_max_label_deviation_pp",
                "score",
            ]

            print(
                best[display_columns].to_string(
                    index=False
                )
            )

        else:

            print()
            print("-" * 100)
            print("NO FEASIBLE CANDIDATE")
            print("-" * 100)

        # ----------------------------------------------------
        # Pareto-style candidates
        # ----------------------------------------------------

        print()
        print("-" * 100)
        print("LOW-PURGE / HIGH-COVERAGE CANDIDATES")
        print("-" * 100)

        pareto_view = (
            result_df
            .sort_values(
                [
                    "purged_pct",
                    "test_asset_coverage_pct",
                    "validation_asset_coverage_pct",
                ],
                ascending=[True, False, False],
            )
            .head(10)
        )

        print(
            pareto_view[
                [
                    "train_fraction_requested",
                    "test_fraction_requested",
                    "validation_start",
                    "test_start",
                    "train_pct",
                    "validation_pct",
                    "test_pct",
                    "purged_pct",
                    "train_assets",
                    "validation_assets",
                    "test_assets",
                    "train_asset_coverage_pct",
                    "validation_asset_coverage_pct",
                    "test_asset_coverage_pct",
                    "score",
                ]
            ].to_string(index=False)
        )

    # ========================================================
    # CROSS-DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 100)
    print("CROSS-DATASET SUMMARY")
    print("=" * 100)

    if not all_results:
        print("No candidates were evaluated.")
        return

    all_df = pd.DataFrame(all_results)

    for dataset_name in DATASETS:

        dataset_df = all_df[
            all_df["dataset"] == dataset_name
        ].copy()

        if dataset_df.empty:
            continue

        feasible_df = dataset_df[
            dataset_df["feasible"]
        ].copy()

        print()
        print(f"{dataset_name}:")

        if feasible_df.empty:
            print("  FEASIBLE: NO")

            best_relaxed = (
                dataset_df
                .sort_values("score", ascending=False)
                .head(1)
            )

            if not best_relaxed.empty:
                row = best_relaxed.iloc[0]

                print(
                    f"  Best relaxed candidate: "
                    f"train={row['train_fraction_requested']:.3f}, "
                    f"test={row['test_fraction_requested']:.3f}"
                )

                print(
                    f"  Actual split: "
                    f"train={row['train_pct']:.2f}%, "
                    f"validation={row['validation_pct']:.2f}%, "
                    f"test={row['test_pct']:.2f}%, "
                    f"purged={row['purged_pct']:.2f}%"
                )

                print(
                    f"  Asset coverage: "
                    f"train={row['train_asset_coverage_pct']:.2f}%, "
                    f"validation={row['validation_asset_coverage_pct']:.2f}%, "
                    f"test={row['test_asset_coverage_pct']:.2f}%"
                )

        else:
            best = (
                feasible_df
                .sort_values("score", ascending=False)
                .iloc[0]
            )

            print("  FEASIBLE: YES")
            print(
                f"  Boundary: "
                f"train={best['train_fraction_requested']:.3f}, "
                f"test={best['test_fraction_requested']:.3f}"
            )
            print(
                f"  Dates: "
                f"validation_start={best['validation_start'].date()}, "
                f"test_start={best['test_start'].date()}"
            )
            print(
                f"  Split: "
                f"train={best['train_pct']:.2f}%, "
                f"validation={best['validation_pct']:.2f}%, "
                f"test={best['test_pct']:.2f}%, "
                f"purged={best['purged_pct']:.2f}%"
            )
            print(
                f"  Assets: "
                f"train={best['train_assets']}, "
                f"validation={best['validation_assets']}, "
                f"test={best['test_assets']}"
            )

    # ========================================================
    # CONSISTENT PHILOSOPHY CHECK
    # ========================================================

    print("\n" + "=" * 100)
    print("COMMON SPLITTING PHILOSOPHY CHECK")
    print("=" * 100)

    feasible_all = all_df[
        all_df["feasible"]
    ].copy()

    if feasible_all.empty:
        print("No globally feasible candidate exists.")
        return

    # Find boundary pairs that are feasible for ALL datasets.
    boundary_pairs = (
        feasible_all[
            [
                "train_fraction_requested",
                "test_fraction_requested",
            ]
        ]
        .drop_duplicates()
    )

    common_pairs = []

    for _, pair in boundary_pairs.iterrows():

        train_fraction = pair["train_fraction_requested"]
        test_fraction = pair["test_fraction_requested"]

        subset = feasible_all[
            (feasible_all["train_fraction_requested"] == train_fraction)
            & (feasible_all["test_fraction_requested"] == test_fraction)
        ]

        if set(subset["dataset"]) == set(DATASETS.keys()):
            common_pairs.append(
                (
                    train_fraction,
                    test_fraction,
                )
            )

    if common_pairs:
        print("Common feasible boundary pairs:")

        for train_fraction, test_fraction in common_pairs:
            print(
                f"  train={train_fraction:.3f}, "
                f"test={test_fraction:.3f}"
            )

    else:
        print(
            "NO COMMON FEASIBLE BOUNDARY PAIR exists across "
            "Stocks 2Y, ETFs 2Y, and Mutual Funds 2Y."
        )

        print()
        print(
            "This does NOT mean the methodology is wrong."
        )

        print(
            "It means the available historical coverage differs "
            "substantially across asset classes, especially for ETFs."
        )


if __name__ == "__main__":
    main()