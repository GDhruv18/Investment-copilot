from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "targets"
    / "etf_2y_diagnostics.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ETF 2Y CROSS-SECTIONAL COHORT AUDIT")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

valid = df[
    df["annualized_return_2y"].notna()
    & df["volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
].copy()

print(f"\nTotal rows: {len(df):,}")
print(f"Valid 2Y observations: {len(valid):,}")
print(f"ETFs in valid dataset: {valid['Ticker'].nunique()}")


# ============================================================
# ETF COUNT AVAILABLE ON EACH DATE
# ============================================================

cohort = (
    valid
    .groupby("Date")["Ticker"]
    .nunique()
    .rename("etf_count")
    .reset_index()
)

print("\n" + "=" * 80)
print("1. CROSS-SECTIONAL COHORT SIZE")
print("=" * 80)

print(
    cohort["etf_count"].describe(
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
    ).to_string()
)


# ============================================================
# EXACT COUNTS BY COHORT SIZE
# ============================================================

print("\n" + "=" * 80)
print("2. DATES BY NUMBER OF AVAILABLE ETFs")
print("=" * 80)

distribution = (
    cohort["etf_count"]
    .value_counts()
    .sort_index()
)

print(distribution.to_string())


# ============================================================
# LOW-COHORT DATES
# ============================================================

print("\n" + "=" * 80)
print("3. EARLIEST LOW-COHORT DATES")
print("=" * 80)

for threshold in [3, 5, 10, 15, 20, 25, 30]:

    subset = cohort[
        cohort["etf_count"] <= threshold
    ]

    print(
        f"\nDates with <= {threshold} ETFs: "
        f"{len(subset):,}"
    )

    if len(subset) > 0:
        print(
            subset.head(20).to_string(index=False)
        )


# ============================================================
# MINIMUM COHORT THRESHOLD EFFECT
# ============================================================

print("\n" + "=" * 80)
print("4. EFFECT OF MINIMUM COHORT SIZE")
print("=" * 80)

thresholds = [3, 5, 10, 15, 20, 25, 30]

for threshold in thresholds:

    eligible_dates = cohort[
        cohort["etf_count"] >= threshold
    ]["Date"]

    eligible = valid[
        valid["Date"].isin(eligible_dates)
    ]

    print(
        f"\nMinimum cohort = {threshold}"
    )

    print(
        f"Eligible dates       : {len(eligible_dates):,}"
    )

    print(
        f"Eligible observations: {len(eligible):,}"
    )

    print(
        f"ETF count            : "
        f"{eligible['Ticker'].nunique()}"
    )


# ============================================================
# DATE RANGE OF COHORT SIZES
# ============================================================

print("\n" + "=" * 80)
print("5. COHORT SIZE OVER TIME")
print("=" * 80)

yearly = (
    valid
    .assign(year=valid["Date"].dt.year)
    .groupby("year")["Ticker"]
    .nunique()
)

print(
    yearly.to_string()
)


# ============================================================
# FIRST VALID 2Y OBSERVATION PER ETF
# ============================================================

print("\n" + "=" * 80)
print("6. FIRST VALID 2Y OBSERVATION PER ETF")
print("=" * 80)

first_valid = (
    valid
    .groupby("Ticker")["Date"]
    .min()
    .sort_values()
)

print(first_valid.to_string())


# ============================================================
# ETF COVERAGE BY YEAR
# ============================================================

print("\n" + "=" * 80)
print("7. ETF COVERAGE BY YEAR")
print("=" * 80)

coverage = (
    valid
    .assign(year=valid["Date"].dt.year)
    .groupby("year")["Ticker"]
    .nunique()
)

print(
    coverage.to_string()
)


# ============================================================
# LABEL FEASIBILITY
# ============================================================

print("\n" + "=" * 80)
print("8. LABEL FEASIBILITY")
print("=" * 80)

print(
    """
For a 30% / 40% / 30% label split:

    Weak       = bottom 30%
    Neutral    = middle 40%
    Attractive = top 30%

A very small cross-sectional cohort makes these
percentile labels unstable.

Examples:

    2 ETFs  -> practically no meaningful 30/40/30 split
    3 ETFs  -> extremely coarse
    5 ETFs  -> still coarse
    10 ETFs -> substantially better
    20+ ETFs -> much more stable
"""
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"\nMinimum ETF cohort : "
    f"{cohort['etf_count'].min()}"
)

print(
    f"Median ETF cohort  : "
    f"{cohort['etf_count'].median():.0f}"
)

print(
    f"Maximum ETF cohort : "
    f"{cohort['etf_count'].max()}"
)

print(
    f"\nDates with < 5 ETFs : "
    f"{(cohort['etf_count'] < 5).sum():,}"
)

print(
    f"Dates with < 10 ETFs: "
    f"{(cohort['etf_count'] < 10).sum():,}"
)

print(
    f"Dates with < 15 ETFs: "
    f"{(cohort['etf_count'] < 15).sum():,}"
)

print(
    f"Dates with < 20 ETFs: "
    f"{(cohort['etf_count'] < 20).sum():,}"
)

print("\nAudit complete.")
print("=" * 80)