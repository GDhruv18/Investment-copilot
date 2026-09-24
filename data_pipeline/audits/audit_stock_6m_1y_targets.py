from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "stock_6m_1y_targets.parquet"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "stock_forward_outcomes.parquet"
)

EXPECTED_STOCKS = 100
MIN_COHORT = 10

ALLOWED_LABELS = {
    "Weak",
    "Neutral",
    "Attractive",
}


# ============================================================
# HELPER
# ============================================================

def audit_horizon(df, horizon, return_column):
    score_column = f"score_{horizon}"
    label_column = f"label_{horizon}"
    cohort_column = f"cohort_size_{horizon}"

    print("\n" + "-" * 70)
    print(f"{horizon.upper()} TARGET AUDIT")
    print("-" * 70)

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    valid_returns = df[return_column].notna()
    labeled = df[label_column].notna()
    scored = df[score_column].notna()

    print(f"Valid forward returns : {valid_returns.sum():,}")
    print(f"Scored observations    : {scored.sum():,}")
    print(f"Labeled observations   : {labeled.sum():,}")

    # --------------------------------------------------------
    # Score / label consistency
    # --------------------------------------------------------

    score_without_label = (
        df[score_column].notna()
        & df[label_column].isna()
    ).sum()

    label_without_score = (
        df[label_column].notna()
        & df[score_column].isna()
    ).sum()

    print(
        f"Score without label    : {score_without_label:,}"
    )
    print(
        f"Label without score    : {label_without_score:,}"
    )

    if score_without_label != 0:
        raise ValueError(
            f"{horizon}: score exists without label."
        )

    if label_without_score != 0:
        raise ValueError(
            f"{horizon}: label exists without score."
        )

    # --------------------------------------------------------
    # Scores must be in [0,1]
    # --------------------------------------------------------

    invalid_scores = df.loc[
        df[score_column].notna()
        & ~df[score_column].between(0, 1),
        score_column,
    ]

    print(
        f"Invalid scores         : {len(invalid_scores):,}"
    )

    if len(invalid_scores) != 0:
        raise ValueError(
            f"{horizon}: scores outside [0,1]."
        )

    # --------------------------------------------------------
    # Labels must be valid
    # --------------------------------------------------------

    invalid_labels = df.loc[
        df[label_column].notna()
        & ~df[label_column].isin(ALLOWED_LABELS),
        label_column,
    ]

    print(
        f"Invalid labels         : {len(invalid_labels):,}"
    )

    if len(invalid_labels) != 0:
        raise ValueError(
            f"{horizon}: invalid labels detected."
        )

    # --------------------------------------------------------
    # Every labeled row must have a valid return
    # --------------------------------------------------------

    labels_without_return = (
        df[label_column].notna()
        & df[return_column].isna()
    ).sum()

    print(
        f"Labels without return  : "
        f"{labels_without_return:,}"
    )

    if labels_without_return != 0:
        raise ValueError(
            f"{horizon}: labels exist without returns."
        )

    # --------------------------------------------------------
    # Cohort validation
    # --------------------------------------------------------

    labeled_cohorts = df.loc[
        labeled,
        cohort_column
    ]

    invalid_cohorts = (
        labeled_cohorts.isna()
        | (labeled_cohorts < MIN_COHORT)
    ).sum()

    print(
        f"Invalid labeled cohorts: "
        f"{invalid_cohorts:,}"
    )

    if invalid_cohorts != 0:
        raise ValueError(
            f"{horizon}: labeled rows have insufficient cohorts."
        )

    # --------------------------------------------------------
    # Verify score actually equals date-level percentile rank
    # --------------------------------------------------------

    verification = df.loc[
        valid_returns
        & df[cohort_column].ge(MIN_COHORT),
        [
            "ticker",
            "observation_date",
            return_column,
            score_column,
        ],
    ].copy()

    expected_scores = (
        verification
        .groupby("observation_date")[return_column]
        .rank(
            pct=True,
            method="average",
        )
    )

    score_difference = (
        verification[score_column].to_numpy()
        - expected_scores.to_numpy()
    )

    max_difference = np.nanmax(
        np.abs(score_difference)
    )

    print(
        f"Maximum score difference: "
        f"{max_difference:.12f}"
    )

    if not np.isclose(
        max_difference,
        0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{horizon}: stored scores do not match "
            "date-level percentile ranking."
        )

    # --------------------------------------------------------
    # Verify label boundaries
    # --------------------------------------------------------

    labeled_df = df.loc[
        labeled,
        [score_column, label_column]
    ]

    weak_invalid = (
        (labeled_df[label_column] == "Weak")
        & (labeled_df[score_column] > 0.30)
    ).sum()

    neutral_invalid_low = (
        (labeled_df[label_column] == "Neutral")
        & (labeled_df[score_column] <= 0.30)
    ).sum()

    neutral_invalid_high = (
        (labeled_df[label_column] == "Neutral")
        & (labeled_df[score_column] > 0.70)
    ).sum()

    attractive_invalid = (
        (labeled_df[label_column] == "Attractive")
        & (labeled_df[score_column] <= 0.70)
    ).sum()

    print(
        f"Weak boundary errors    : {weak_invalid:,}"
    )
    print(
        f"Neutral lower errors    : "
        f"{neutral_invalid_low:,}"
    )
    print(
        f"Neutral upper errors    : "
        f"{neutral_invalid_high:,}"
    )
    print(
        f"Attractive errors       : "
        f"{attractive_invalid:,}"
    )

    if any([
        weak_invalid,
        neutral_invalid_low,
        neutral_invalid_high,
        attractive_invalid,
    ]):
        raise ValueError(
            f"{horizon}: label boundary errors detected."
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    counts = (
        df.loc[labeled, label_column]
        .value_counts()
        .reindex(
            ["Weak", "Neutral", "Attractive"],
            fill_value=0,
        )
    )

    total = counts.sum()

    print("\nLabel distribution:")

    for label, count in counts.items():
        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {label:<10}: "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Cohort statistics
    # --------------------------------------------------------

    if not labeled_cohorts.empty:

        print("\nLabeled cohort statistics:")
        print(
            f"  Minimum : "
            f"{labeled_cohorts.min():.0f}"
        )
        print(
            f"  Median  : "
            f"{labeled_cohorts.median():.0f}"
        )
        print(
            f"  Maximum : "
            f"{labeled_cohorts.max():.0f}"
        )

    return {
        "valid_returns": int(valid_returns.sum()),
        "scored": int(scored.sum()),
        "labeled": int(labeled.sum()),
        "max_score_difference": float(max_difference),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOCK 6M / 1Y TARGET AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Target file not found:\n{INPUT_FILE}"
        )

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found:\n{SOURCE_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_FILE)
    source = pd.read_parquet(SOURCE_FILE)

    print("\nINPUT")
    print("-" * 70)

    print(
        f"Target rows : {len(df):,}"
    )

    print(
        f"Source rows : {len(source):,}"
    )

    print(
        f"Target stocks: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "ticker",
        "observation_date",
        "forward_return_6m",
        "forward_return_1y",
        "score_6m",
        "label_6m",
        "cohort_size_6m",
        "score_1y",
        "label_1y",
        "cohort_size_1y",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # DATE NORMALIZATION
    # --------------------------------------------------------

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    source["observation_date"] = pd.to_datetime(
        source["observation_date"]
    )

    # --------------------------------------------------------
    # UNIVERSE VALIDATION
    # --------------------------------------------------------

    stock_count = df["ticker"].nunique()

    print(
        f"Expected stocks: {EXPECTED_STOCKS}"
    )
    print(
        f"Actual stocks  : {stock_count}"
    )

    if stock_count != EXPECTED_STOCKS:
        raise ValueError(
            "Stock universe count changed."
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "ticker",
            "observation_date",
        ]
    ).sum()

    print(
        f"Duplicate ticker/date rows: "
        f"{duplicates}"
    )

    if duplicates != 0:
        raise ValueError(
            "Duplicate ticker/date rows detected."
        )

    # --------------------------------------------------------
    # DYNAMIC SAFETY CHECK
    # --------------------------------------------------------

    dynamic_rows = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .eq("DYNAMIC.NS")
        .sum()
    )

    print(
        f"DYNAMIC.NS rows: {dynamic_rows}"
    )

    if dynamic_rows != 0:
        raise ValueError(
            "DYNAMIC.NS found in stock target layer."
        )

    # --------------------------------------------------------
    # ROW COUNT PRESERVATION
    # --------------------------------------------------------

    print(
        f"Row count preserved: "
        f"{len(df) == len(source)}"
    )

    if len(df) != len(source):
        raise ValueError(
            "Target layer changed source row count."
        )

    # --------------------------------------------------------
    # SOURCE FORWARD RETURN COMPARISON
    # --------------------------------------------------------

    target_returns = df[
        [
            "ticker",
            "observation_date",
            "forward_return_6m",
            "forward_return_1y",
        ]
    ].copy()

    source_returns = source[
        [
            "ticker",
            "observation_date",
            "forward_return_6m",
            "forward_return_1y",
        ]
    ].copy()

    comparison = target_returns.merge(
        source_returns,
        on=[
            "ticker",
            "observation_date",
        ],
        how="outer",
        suffixes=(
            "_target",
            "_source",
            ),
        indicator=True,
    )

    unexpected_rows = (
        comparison["_merge"] != "both"
    ).sum()

    print(
        f"Rows missing from source/target: "
        f"{unexpected_rows}"
    )

    if unexpected_rows != 0:
        raise ValueError(
            "Target/source row sets differ."
        )

    six_month_match = np.allclose(
        comparison["forward_return_6m_target"]
        .to_numpy(),
        comparison["forward_return_6m_source"]
        .to_numpy(),
        equal_nan=True,
    )

    one_year_match = np.allclose(
        comparison["forward_return_1y_target"]
        .to_numpy(),
        comparison["forward_return_1y_source"]
        .to_numpy(),
        equal_nan=True,
    )

    print(
        f"6M returns unchanged: "
        f"{six_month_match}"
    )

    print(
        f"1Y returns unchanged: "
        f"{one_year_match}"
    )

    if not six_month_match:
        raise ValueError(
            "6M forward returns changed."
        )

    if not one_year_match:
        raise ValueError(
            "1Y forward returns changed."
        )

    # --------------------------------------------------------
    # HORIZON AUDITS
    # --------------------------------------------------------

    audit_horizon(
        df,
        horizon="6m",
        return_column="forward_return_6m",
    )

    audit_horizon(
        df,
        horizon="1y",
        return_column="forward_return_1y",
    )

    # --------------------------------------------------------
    # TARGET/SOURCE DATE RANGE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATE RANGE VALIDATION")
    print("-" * 70)

    target_min = df["observation_date"].min()
    target_max = df["observation_date"].max()

    source_min = source["observation_date"].min()
    source_max = source["observation_date"].max()

    print(
        f"Target: "
        f"{target_min.date()} -> {target_max.date()}"
    )

    print(
        f"Source: "
        f"{source_min.date()} -> {source_max.date()}"
    )

    if target_min != source_min:
        raise ValueError(
            "Target minimum date changed."
        )

    if target_max != source_max:
        raise ValueError(
            "Target maximum date changed."
        )

    # --------------------------------------------------------
    # RELOAD VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RELOAD VALIDATION")
    print("=" * 70)

    reloaded = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Reloaded rows  : "
        f"{len(reloaded):,}"
    )

    print(
        f"Reloaded stocks: "
        f"{reloaded['ticker'].nunique()}"
    )

    if len(reloaded) != len(df):
        raise ValueError(
            "Reloaded row count mismatch."
        )

    if (
        reloaded["ticker"].nunique()
        != EXPECTED_STOCKS
    ):
        raise ValueError(
            "Reloaded stock count mismatch."
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: PASS")
    print("=" * 70)

    print(
        "Stock 6M / 1Y target layer passed "
        "structural, ranking, label, cohort, "
        "source-integrity, and reload validation."
    )


if __name__ == "__main__":
    main()