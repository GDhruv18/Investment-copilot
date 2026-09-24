from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/targets/etf_2y_diagnostics.parquet"
)

EXPECTED_ETF_COUNT = 41

COMPONENTS = [
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",
]


# ============================================================
# HELPERS
# ============================================================

def print_distribution(series, name):
    """Print useful distribution statistics."""

    series = series.dropna()

    print(f"\n{name}")
    print("-" * 70)

    print(f"Count:              {len(series):,}")
    print(f"Mean:               {series.mean():.6f}")
    print(f"Std:                {series.std():.6f}")
    print(f"Min:                {series.min():.6f}")

    for label, q in [
        ("1%", 0.01),
        ("5%", 0.05),
        ("10%", 0.10),
        ("25%", 0.25),
        ("Median", 0.50),
        ("75%", 0.75),
        ("90%", 0.90),
        ("95%", 0.95),
        ("99%", 0.99),
    ]:
        print(
            f"{label:<20}"
            f"{series.quantile(q):.6f}"
        )

    print(f"Max:                {series.max():.6f}")


def print_rank_behavior(df):
    """
    Examine how the four components rank ETFs on each date.

    This does not create labels. It only measures the
    cross-sectional rank relationships.
    """

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL RANK RELATIONSHIPS")
    print("=" * 70)

    temp = df[
        ["Ticker", "Date"] + COMPONENTS
    ].copy()

    for component in COMPONENTS:

        rank_column = f"{component}_rank"

        # Higher return is better.
        # Higher volatility is worse.
        # Less negative drawdown is better.
        # Higher risk-adjusted return is better.
        ascending = component in [
            "volatility_2y",
        ]

        temp[rank_column] = (
            temp.groupby("Date")[component]
            .rank(
                method="average",
                ascending=ascending,
                pct=True
            )
        )

    rank_columns = [
        f"{component}_rank"
        for component in COMPONENTS
    ]

    rank_data = temp[rank_columns].dropna()

    print(
        f"\nObservations with all four ranks: "
        f"{len(rank_data):,}"
    )

    if len(rank_data) == 0:
        print("No complete observations available.")
        return

    print("\nMean cross-sectional percentile")
    print("-" * 70)

    for component in COMPONENTS:

        column = f"{component}_rank"

        print(
            f"{component:<35}"
            f"{rank_data[column].mean():.6f}"
        )

    print("\nRank correlation — Spearman")
    print("-" * 70)

    print(
        rank_data.corr(
            method="spearman"
        ).to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("2Y+ ETF COMPONENT RELATIONSHIP AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Rows:               {len(df):,}"
    )

    print(
        f"Tickers:            "
        f"{df['Ticker'].nunique():,}"
    )

    print(
        f"Date range:         "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Universe validation
    # --------------------------------------------------------

    print("\nUNIVERSE VALIDATION")
    print("-" * 70)

    ticker_count = df["Ticker"].nunique()

    print(
        f"Expected ETFs:      {EXPECTED_ETF_COUNT}"
    )

    print(
        f"Actual ETFs:        {ticker_count}"
    )

    if ticker_count != EXPECTED_ETF_COUNT:
        raise ValueError(
            "ETF universe size mismatch."
        )

    if "DYNAMIC.NS" in set(
        df["Ticker"].unique()
    ):
        raise ValueError(
            "DYNAMIC.NS is present."
        )

    print("DYNAMIC.NS present: False")
    print("Universe check:     PASS")

    # --------------------------------------------------------
    # Component validation
    # --------------------------------------------------------

    print("\nCOMPONENT VALIDATION")
    print("-" * 70)

    missing_columns = [
        column
        for column in COMPONENTS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing components: {missing_columns}"
        )

    print("All four components: PASS")

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    print("\nCOMPONENT COVERAGE")
    print("-" * 70)

    for component in COMPONENTS:

        valid_count = (
            df[component]
            .notna()
            .sum()
        )

        missing_count = (
            df[component]
            .isna()
            .sum()
        )

        coverage = (
            valid_count / len(df)
        )

        print(
            f"{component:<35}"
            f"Valid: {valid_count:>8,}   "
            f"Missing: {missing_count:>8,}   "
            f"Coverage: {coverage:>7.2%}"
        )

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPONENT DISTRIBUTIONS")
    print("=" * 70)

    print_distribution(
        df["annualized_return_2y"],
        "2Y ANNUALIZED RETURN"
    )

    print_distribution(
        df["volatility_2y"],
        "2Y ANNUALIZED VOLATILITY"
    )

    print_distribution(
        df["max_drawdown_2y"],
        "2Y MAXIMUM DRAWDOWN"
    )

    print_distribution(
        df["risk_adjusted_return_2y"],
        "2Y RISK-ADJUSTED RETURN"
    )

    # --------------------------------------------------------
    # Pearson correlations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PEARSON CORRELATION")
    print("=" * 70)

    pearson = df[COMPONENTS].corr(
        method="pearson"
    )

    print(
        pearson.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Spearman correlations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION")
    print("=" * 70)

    spearman = df[COMPONENTS].corr(
        method="spearman"
    )

    print(
        spearman.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Pairwise complete observations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIRWISE COMPLETE OBSERVATION COUNTS")
    print("=" * 70)

    for i, component_a in enumerate(COMPONENTS):

        for component_b in COMPONENTS[i + 1:]:

            pair = df[
                [
                    component_a,
                    component_b
                ]
            ].dropna()

            print(
                f"{component_a:<35}"
                f"<-> {component_b:<35}"
                f"{len(pair):,}"
            )

    # --------------------------------------------------------
    # Cross-sectional rank relationships
    # --------------------------------------------------------

    print_rank_behavior(df)

    # --------------------------------------------------------
    # Component sign / interpretation checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPONENT INTERPRETATION CHECK")
    print("=" * 70)

    print(
        """
Return:
    Higher is better.

Volatility:
    Lower is generally preferable from a risk perspective.

Maximum drawdown:
    Less negative is preferable.

Risk-adjusted return:
    Higher is better.
"""
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        "2Y component relationship audit completed."
    )

    print(
        "NO LABELS WERE CREATED."
    )

    print(
        "NO DATA WAS MODIFIED."
    )


if __name__ == "__main__":
    main()