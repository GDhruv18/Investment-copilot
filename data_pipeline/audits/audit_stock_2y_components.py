"""
Audit Stock 2Y Components

Purpose:
    Study relationships between:
        1. Annualized return
        2. Annualized volatility
        3. Maximum drawdown
        4. Raw risk-adjusted return

No scoring formula is created or locked here.

The audit includes:
    - Pearson correlations
    - Spearman correlations
    - Cross-sectional rank correlations
    - Basic component distributions
    - Missing-value coverage
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "targets"
    / "stock_2y_diagnostics.parquet"
)


# ============================================================
# CONSTANTS
# ============================================================

COMPONENTS = [
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",
]


DISPLAY_NAMES = {
    "annualized_return_2y": "Annualized Return",
    "annualized_volatility_2y": "Annualized Volatility",
    "max_drawdown_2y": "Maximum Drawdown",
    "risk_adjusted_return_2y": "Risk-Adjusted Return",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print("=" * 70)
    print("STOCK 2Y COMPONENT RELATIONSHIP AUDIT")
    print("=" * 70)

    print()
    print(f"Input file : {INPUT_FILE}")
    print(f"Rows       : {len(df):,}")
    print(
        f"Stocks     : "
        f"{df['ticker'].nunique()}"
    )

    required = [
        "ticker",
        "date",
        "future_date_2y",
        *COMPONENTS,
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# VALID 2Y DATA
# ============================================================

def prepare_valid_data(df):

    valid = df[
        df["future_date_2y"].notna()
    ].copy()

    valid = valid[
        valid[COMPONENTS]
        .notna()
        .all(axis=1)
    ].copy()

    valid = valid[
        np.isfinite(
            valid[COMPONENTS]
        ).all(axis=1)
    ].copy()

    print()
    print("=" * 70)
    print("VALID DATA COVERAGE")
    print("=" * 70)

    print(
        f"Total diagnostic rows : {len(df):,}"
    )

    print(
        f"Valid 2Y rows         : {len(valid):,}"
    )

    print(
        f"Missing/incomplete    : "
        f"{len(df) - len(valid):,}"
    )

    print(
        f"Coverage              : "
        f"{len(valid) / len(df) * 100:.2f}%"
    )

    return valid


# ============================================================
# COMPONENT COVERAGE
# ============================================================

def print_component_coverage(df):

    print()
    print("=" * 70)
    print("COMPONENT COVERAGE")
    print("=" * 70)

    for col in COMPONENTS:

        valid_count = (
            df[col]
            .notna()
            .sum()
        )

        print(
            f"{DISPLAY_NAMES[col]:30s} "
            f"{valid_count:>10,} / "
            f"{len(df):,} "
            f"({valid_count / len(df) * 100:.2f}%)"
        )


# ============================================================
# PEARSON CORRELATION
# ============================================================

def print_pearson(valid):

    print()
    print("=" * 70)
    print("PEARSON CORRELATION")
    print("=" * 70)

    corr = valid[
        COMPONENTS
    ].corr(
        method="pearson"
    )

    print(
        corr.to_string(
            float_format=lambda x: f"{x: .4f}"
        )
    )

    print()
    print("Pairwise Pearson correlations:")

    for i in range(len(COMPONENTS)):

        for j in range(i + 1, len(COMPONENTS)):

            a = COMPONENTS[i]
            b = COMPONENTS[j]

            value = corr.loc[a, b]

            print(
                f"  {DISPLAY_NAMES[a]} "
                f"vs "
                f"{DISPLAY_NAMES[b]}: "
                f"{value:.4f}"
            )


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def print_spearman(valid):

    print()
    print("=" * 70)
    print("SPEARMAN CORRELATION")
    print("=" * 70)

    corr = valid[
        COMPONENTS
    ].corr(
        method="spearman"
    )

    print(
        corr.to_string(
            float_format=lambda x: f"{x: .4f}"
        )
    )

    print()
    print("Pairwise Spearman correlations:")

    for i in range(len(COMPONENTS)):

        for j in range(i + 1, len(COMPONENTS)):

            a = COMPONENTS[i]
            b = COMPONENTS[j]

            value = corr.loc[a, b]

            print(
                f"  {DISPLAY_NAMES[a]} "
                f"vs "
                f"{DISPLAY_NAMES[b]}: "
                f"{value:.4f}"
            )


# ============================================================
# CROSS-SECTIONAL RANK CORRELATIONS
# ============================================================

def calculate_rank_correlations(valid):

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANK CORRELATIONS")
    print("=" * 70)

    ranked_frames = []

    for date, group in valid.groupby(
        "date",
        sort=False
    ):

        group = group.copy()

        # ----------------------------------------------------
        # Percentile ranks within each date
        # ----------------------------------------------------

        group["return_rank"] = (
            group[
                "annualized_return_2y"
            ].rank(
                pct=True,
                method="average"
            )
        )

        # Higher volatility = worse.
        # Convert to "risk quality":
        # low volatility -> high percentile.
        group["risk_rank"] = (
            group[
                "annualized_volatility_2y"
            ].rank(
                pct=True,
                method="average"
            )
        )

        group["risk_quality_rank"] = (
            1.0 - group["risk_rank"]
        )

        # Less-negative drawdown = better.
        group["drawdown_rank"] = (
            group[
                "max_drawdown_2y"
            ].rank(
                pct=True,
                method="average"
            )
        )

        # Raw risk-adjusted return:
        # higher = better.
        group["risk_adjusted_rank"] = (
            group[
                "risk_adjusted_return_2y"
            ].rank(
                pct=True,
                method="average"
            )
        )

        ranked_frames.append(
            group[
                [
                    "ticker",
                    "date",
                    "return_rank",
                    "risk_quality_rank",
                    "drawdown_rank",
                    "risk_adjusted_rank",
                ]
            ]
        )

    ranked = pd.concat(
        ranked_frames,
        ignore_index=True
    )

    rank_columns = [
        "return_rank",
        "risk_quality_rank",
        "drawdown_rank",
        "risk_adjusted_rank",
    ]

    corr = ranked[
        rank_columns
    ].corr(
        method="pearson"
    )

    print()
    print(
        corr.to_string(
            float_format=lambda x: f"{x: .4f}"
        )
    )

    print()
    print("Pairwise cross-sectional rank correlations:")

    pairs = [
        (
            "return_rank",
            "risk_quality_rank",
            "Return rank vs Risk-quality rank",
        ),
        (
            "return_rank",
            "drawdown_rank",
            "Return rank vs Drawdown rank",
        ),
        (
            "return_rank",
            "risk_adjusted_rank",
            "Return rank vs Risk-adjusted rank",
        ),
        (
            "risk_quality_rank",
            "drawdown_rank",
            "Risk-quality rank vs Drawdown rank",
        ),
        (
            "risk_quality_rank",
            "risk_adjusted_rank",
            "Risk-quality rank vs Risk-adjusted rank",
        ),
        (
            "drawdown_rank",
            "risk_adjusted_rank",
            "Drawdown rank vs Risk-adjusted rank",
        ),
    ]

    for a, b, label in pairs:

        value = corr.loc[a, b]

        print(
            f"  {label}: {value:.4f}"
        )

    return ranked


# ============================================================
# COMPONENT DISTRIBUTIONS
# ============================================================

def print_distributions(valid):

    print()
    print("=" * 70)
    print("COMPONENT DISTRIBUTIONS")
    print("=" * 70)

    for col in COMPONENTS:

        series = valid[col]

        print()
        print(
            DISPLAY_NAMES[col]
        )
        print("-" * 70)

        stats = series.describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )

        print(
            stats.to_string(
                float_format=lambda x: f"{x:.6f}"
            )
        )


# ============================================================
# RISK-ADJUSTED EXTREMES
# ============================================================

def print_risk_adjusted_extremes(valid):

    print()
    print("=" * 70)
    print("RISK-ADJUSTED RETURN EXTREMES")
    print("=" * 70)

    columns = [
        "ticker",
        "date",
        "annualized_return_2y",
        "annualized_volatility_2y",
        "max_drawdown_2y",
        "risk_adjusted_return_2y",
    ]

    highest = valid.nlargest(
        10,
        "risk_adjusted_return_2y"
    )[columns]

    lowest = valid.nsmallest(
        10,
        "risk_adjusted_return_2y"
    )[columns]

    print()
    print("TOP 10 RISK-ADJUSTED RETURNS")
    print("-" * 70)

    print(
        highest.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("BOTTOM 10 RISK-ADJUSTED RETURNS")
    print("-" * 70)

    print(
        lowest.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# RETURN / RISK EXTREMES
# ============================================================

def print_extreme_counts(valid):

    print()
    print("=" * 70)
    print("EXTREME VALUE COUNTS")
    print("=" * 70)

    return_series = (
        valid["annualized_return_2y"]
    )

    volatility = (
        valid["annualized_volatility_2y"]
    )

    drawdown = (
        valid["max_drawdown_2y"]
    )

    print()
    print("ANNUALIZED RETURN")
    print("-" * 70)

    for threshold in [
        0.50,
        1.00,
        2.00,
        5.00,
    ]:
        print(
            f"Return > {threshold * 100:.0f}%: "
            f"{(return_series > threshold).sum():,}"
        )

    for threshold in [
        -0.25,
        -0.50,
        -0.75,
        -0.90,
    ]:
        print(
            f"Return < {threshold * 100:.0f}%: "
            f"{(return_series < threshold).sum():,}"
        )

    print()
    print("VOLATILITY")
    print("-" * 70)

    for threshold in [
        0.20,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
    ]:
        print(
            f"Volatility > {threshold * 100:.0f}%: "
            f"{(volatility > threshold).sum():,}"
        )

    print()
    print("DRAWDOWN")
    print("-" * 70)

    for threshold in [
        -0.25,
        -0.50,
        -0.60,
        -0.75,
    ]:
        print(
            f"Drawdown < {threshold * 100:.0f}%: "
            f"{(drawdown < threshold).sum():,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    print_component_coverage(
        df
    )

    valid = prepare_valid_data(
        df
    )

    print_distributions(
        valid
    )

    print_pearson(
        valid
    )

    print_spearman(
        valid
    )

    calculate_rank_correlations(
        valid
    )

    print_risk_adjusted_extremes(
        valid
    )

    print_extreme_counts(
        valid
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        "No scoring formula was created."
    )

    print(
        "No labels were created."
    )


if __name__ == "__main__":
    main()