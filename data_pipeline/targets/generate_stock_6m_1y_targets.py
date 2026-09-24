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
    / "stock_forward_outcomes.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "stock_6m_1y_targets.parquet"
)

EXPECTED_STOCKS = 100
MIN_COHORT = 10

HORIZONS = {
    "6m": "forward_return_6m",
    "1y": "forward_return_1y",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def assign_label(score):
    """
    Convert percentile score into the locked 3-class target.

    Bottom 30%  -> Weak
    Middle 40%  -> Neutral
    Top 30%     -> Attractive
    """

    if pd.isna(score):
        return np.nan

    if score <= 0.30:
        return "Weak"

    if score <= 0.70:
        return "Neutral"

    return "Attractive"


def generate_horizon_targets(df, horizon, return_column):
    """
    Generate cross-sectional targets for one horizon.

    Ranking is performed independently for every observation date,
    using only stocks that have a valid forward return.

    Stocks are ranked against stocks only.
    """

    score_column = f"score_{horizon}"
    label_column = f"label_{horizon}"
    cohort_column = f"cohort_size_{horizon}"

    df[score_column] = np.nan
    df[label_column] = np.nan
    df[cohort_column] = np.nan

    # --------------------------------------------------------
    # Identify rows with valid forward returns
    # --------------------------------------------------------

    valid_mask = df[return_column].notna()

    valid = df.loc[
        valid_mask,
        ["observation_date", "ticker", return_column]
    ].copy()

    if valid.empty:
        print(f"\n{horizon.upper()} — no valid observations found.")
        return df

    # --------------------------------------------------------
    # Cohort size for every observation date
    # --------------------------------------------------------

    cohort_sizes = (
        valid.groupby("observation_date")["ticker"]
        .nunique()
    )

    df.loc[
        valid_mask,
        cohort_column
    ] = df.loc[
        valid_mask,
        "observation_date"
    ].map(cohort_sizes)

    # --------------------------------------------------------
    # Only dates with sufficient cross-sectional coverage
    # --------------------------------------------------------

    eligible_dates = cohort_sizes[
        cohort_sizes >= MIN_COHORT
    ].index

    eligible_mask = (
        valid["observation_date"].isin(eligible_dates)
    )

    eligible = valid.loc[eligible_mask].copy()

    if eligible.empty:
        print(
            f"\n{horizon.upper()} — no dates have "
            f"minimum cohort of {MIN_COHORT}."
        )
        return df

    # --------------------------------------------------------
    # Cross-sectional percentile rank
    #
    # Higher forward return = higher score.
    #
    # Ranking happens independently on every date.
    # --------------------------------------------------------

    eligible[score_column] = (
        eligible
        .groupby("observation_date")[return_column]
        .rank(
            pct=True,
            method="average"
        )
    )

    # --------------------------------------------------------
    # Convert scores to labels
    # --------------------------------------------------------

    eligible[label_column] = (
        eligible[score_column]
        .apply(assign_label)
    )

    # --------------------------------------------------------
    # Merge scores and labels back into main dataframe
    # --------------------------------------------------------

    target_values = eligible[
        [
            "observation_date",
            "ticker",
            score_column,
            label_column,
        ]
    ].copy()

    df = df.merge(
        target_values,
        on=["observation_date", "ticker"],
        how="left",
        suffixes=("", "_new"),
    )

    # The merge creates the original initialized columns plus
    # "_new" versions. Replace the initialized columns.

    df[score_column] = df[f"{score_column}_new"]
    df[label_column] = df[f"{label_column}_new"]

    df.drop(
        columns=[
            f"{score_column}_new",
            f"{label_column}_new",
        ],
        inplace=True,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    labeled = df[label_column].notna()

    print(f"\n{'-' * 60}")
    print(f"{horizon.upper()} TARGET GENERATION")
    print(f"{'-' * 60}")

    print(f"Valid forward returns : {valid_mask.sum():,}")
    print(f"Eligible observations  : {labeled.sum():,}")
    print(f"Eligible dates         : {len(eligible_dates):,}")
    print(
        f"Minimum cohort         : {MIN_COHORT}"
    )

    print("\nLabel distribution:")

    label_counts = (
        df.loc[labeled, label_column]
        .value_counts()
        .reindex(
            ["Weak", "Neutral", "Attractive"],
            fill_value=0,
        )
    )

    total_labeled = label_counts.sum()

    for label, count in label_counts.items():
        percentage = (
            count / total_labeled * 100
            if total_labeled > 0
            else 0
        )

        print(
            f"  {label:<10}: "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    if labeled.any():
        print(
            f"\nScore range: "
            f"{df.loc[labeled, score_column].min():.6f}"
            f" -> "
            f"{df.loc[labeled, score_column].max():.6f}"
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL STOCK 6M / 1Y TARGET GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print("\nINPUT VALIDATION")
    print("-" * 70)

    df = pd.read_parquet(INPUT_FILE)

    print(f"Input rows: {len(df):,}")

    required_columns = [
        "ticker",
        "observation_date",
        "forward_return_6m",
        "forward_return_1y",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # NORMALIZE DATE
    # --------------------------------------------------------

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    # --------------------------------------------------------
    # BASIC STRUCTURE CHECKS
    # --------------------------------------------------------

    stock_count = df["ticker"].nunique()

    print(f"Stocks: {stock_count}")
    print(
        f"Date range: "
        f"{df['observation_date'].min().date()} "
        f"-> "
        f"{df['observation_date'].max().date()}"
    )

    if stock_count != EXPECTED_STOCKS:
        raise ValueError(
            f"Expected {EXPECTED_STOCKS} stocks, "
            f"found {stock_count}."
        )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=["ticker", "observation_date"]
    ).sum()

    print(f"Duplicate ticker/date rows: {duplicates}")

    if duplicates != 0:
        raise ValueError(
            "Duplicate ticker/date rows detected."
        )

    # --------------------------------------------------------
    # DYNAMIC.NS SAFETY CHECK
    # --------------------------------------------------------

    dynamic_count = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .eq("DYNAMIC.NS")
        .sum()
    )

    print(f"DYNAMIC.NS rows: {dynamic_count}")

    if dynamic_count != 0:
        raise ValueError(
            "DYNAMIC.NS found in stock universe."
        )

    # --------------------------------------------------------
    # SOURCE RETURN VALIDATION
    # --------------------------------------------------------

    for horizon, return_column in HORIZONS.items():

        invalid_infinite = np.isinf(
            df[return_column]
            .dropna()
            .to_numpy()
        ).sum()

        print(
            f"{horizon.upper()} valid returns: "
            f"{df[return_column].notna().sum():,}"
        )

        print(
            f"{horizon.upper()} infinite returns: "
            f"{invalid_infinite}"
        )

        if invalid_infinite != 0:
            raise ValueError(
                f"Infinite {horizon} returns detected."
            )

    # --------------------------------------------------------
    # KEEP EXACT SOURCE COPY FOR VALIDATION
    # --------------------------------------------------------

    source_returns = df[
        [
            "ticker",
            "observation_date",
            "forward_return_6m",
            "forward_return_1y",
        ]
    ].copy()

    # --------------------------------------------------------
    # GENERATE 6M TARGETS
    # --------------------------------------------------------

    df = generate_horizon_targets(
        df,
        horizon="6m",
        return_column="forward_return_6m",
    )

    # --------------------------------------------------------
    # GENERATE 1Y TARGETS
    # --------------------------------------------------------

    df = generate_horizon_targets(
        df,
        horizon="1y",
        return_column="forward_return_1y",
    )

    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    preferred_columns = [
        "ticker",
        "observation_date",
        "observation_price",

        "target_date_6m",
        "future_date_6m",
        "future_price_6m",
        "forward_return_6m",

        "score_6m",
        "label_6m",
        "cohort_size_6m",

        "target_date_1y",
        "future_date_1y",
        "future_price_1y",
        "forward_return_1y",

        "score_1y",
        "label_1y",
        "cohort_size_1y",
    ]

    existing_preferred = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    remaining_columns = [
        col
        for col in df.columns
        if col not in existing_preferred
    ]

    df = df[
        existing_preferred + remaining_columns
    ]

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(f"Output rows: {len(df):,}")
    print(f"Output stocks: {df['ticker'].nunique()}")

    # Row count must remain identical.
    if len(df) != len(source_returns):
        raise ValueError(
            "Output row count changed."
        )

    # Stock count must remain identical.
    if df["ticker"].nunique() != EXPECTED_STOCKS:
        raise ValueError(
            "Output stock count changed."
        )

    # Duplicate check after target generation.
    output_duplicates = df.duplicated(
        subset=["ticker", "observation_date"]
    ).sum()

    print(
        f"Output duplicate ticker/date rows: "
        f"{output_duplicates}"
    )

    if output_duplicates != 0:
        raise ValueError(
            "Duplicates introduced during target generation."
        )

    # --------------------------------------------------------
    # Verify raw forward returns were not changed.
    # --------------------------------------------------------

    comparison = df[
        [
            "ticker",
            "observation_date",
            "forward_return_6m",
            "forward_return_1y",
        ]
    ].merge(
        source_returns,
        on=["ticker", "observation_date"],
        how="inner",
        suffixes=("_output", "_source"),
    )

    six_month_match = np.allclose(
        comparison["forward_return_6m_output"].fillna(np.nan),
        comparison["forward_return_6m_source"].fillna(np.nan),
        equal_nan=True,
    )

    one_year_match = np.allclose(
        comparison["forward_return_1y_output"].fillna(np.nan),
        comparison["forward_return_1y_source"].fillna(np.nan),
        equal_nan=True,
    )

    print(
        f"6M forward returns unchanged: "
        f"{six_month_match}"
    )

    print(
        f"1Y forward returns unchanged: "
        f"{one_year_match}"
    )

    if not six_month_match:
        raise ValueError(
            "6M forward returns were modified."
        )

    if not one_year_match:
        raise ValueError(
            "1Y forward returns were modified."
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    allowed_labels = {
        "Weak",
        "Neutral",
        "Attractive",
    }

    for horizon in ["6m", "1y"]:

        label_column = f"label_{horizon}"
        score_column = f"score_{horizon}"

        invalid_labels = (
            df[label_column]
            .dropna()
            .loc[
                ~df[label_column]
                .dropna()
                .isin(allowed_labels)
            ]
        )

        if len(invalid_labels) != 0:
            raise ValueError(
                f"Invalid {horizon} labels detected."
            )

        invalid_scores = df[
            score_column
        ].dropna().loc[
            ~df[score_column]
            .dropna()
            .between(0, 1)
        ]

        if len(invalid_scores) != 0:
            raise ValueError(
                f"Invalid {horizon} scores detected."
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # RELOAD VALIDATION
    # --------------------------------------------------------

    print("\nRELOAD VALIDATION")
    print("-" * 70)

    reloaded = pd.read_parquet(
        OUTPUT_FILE
    )

    print(
        f"Reloaded rows: "
        f"{len(reloaded):,}"
    )

    print(
        f"Reloaded stocks: "
        f"{reloaded['ticker'].nunique()}"
    )

    if len(reloaded) != len(df):
        raise ValueError(
            "Reloaded row count does not match."
        )

    if (
        reloaded["ticker"].nunique()
        != EXPECTED_STOCKS
    ):
        raise ValueError(
            "Reloaded stock count does not match."
        )

    # --------------------------------------------------------
    # FINAL LABEL SUMMARY
    # --------------------------------------------------------

    print("\nFINAL LABEL SUMMARY")

    for horizon in ["6m", "1y"]:

        label_column = f"label_{horizon}"

        counts = (
            reloaded[label_column]
            .value_counts()
            .reindex(
                ["Weak", "Neutral", "Attractive"],
                fill_value=0,
            )
        )

        total = counts.sum()

        print(f"\n{horizon.upper()}:")

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

    print("\n" + "=" * 70)
    print("STATUS: PASS")
    print("=" * 70)
    print(
        "Stock 6M / 1Y target generation completed "
        "without modifying source forward outcomes."
    )


if __name__ == "__main__":
    main()