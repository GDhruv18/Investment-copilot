"""
Audit Stock 2Y Cross-Sectional Cohorts

Purpose:
    Determine how many stocks have valid 2Y outcomes on each
    observation date.

No labels or scoring formula are created.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "targets"
    / "stock_2y_diagnostics.parquet"
)


# ============================================================
# LOAD
# ============================================================

def main():

    print("=" * 70)
    print("STOCK 2Y CROSS-SECTIONAL COHORT AUDIT")
    print("=" * 70)

    df = pd.read_parquet(
        INPUT_FILE
    )

    valid = df[
        df["future_date_2y"].notna()
    ].copy()

    print()
    print(
        f"Total diagnostic rows : {len(df):,}"
    )

    print(
        f"Valid 2Y observations : {len(valid):,}"
    )

    print(
        f"Stocks represented     : "
        f"{valid['ticker'].nunique()}"
    )

    # ========================================================
    # COHORT SIZE BY DATE
    # ========================================================

    cohort = (
        valid
        .groupby("date")["ticker"]
        .nunique()
        .rename("stock_count")
        .reset_index()
    )

    print()
    print("=" * 70)
    print("COHORT SIZE DISTRIBUTION")
    print("=" * 70)

    print(
        cohort["stock_count"].describe(
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

    # ========================================================
    # EXACT COUNTS
    # ========================================================

    print()
    print("=" * 70)
    print("COHORT THRESHOLD COUNTS")
    print("=" * 70)

    thresholds = [
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]

    total_dates = len(cohort)

    for threshold in thresholds:

        count = (
            cohort["stock_count"]
            <= threshold
        ).sum()

        print(
            f"Dates with <= {threshold:2d} stocks: "
            f"{count:>5,} "
            f"({count / total_dates * 100:6.2f}%)"
        )

    # ========================================================
    # CANDIDATE MINIMUM COHORT IMPACT
    # ========================================================

    print()
    print("=" * 70)
    print("MINIMUM COHORT IMPACT")
    print("=" * 70)

    for minimum in [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]:

        eligible_dates = cohort[
            cohort["stock_count"] >= minimum
        ]

        eligible_date_set = set(
            eligible_dates["date"]
        )

        eligible_rows = valid[
            valid["date"].isin(
                eligible_date_set
            )
        ]

        print(
            f"Minimum cohort {minimum:2d}: "
            f"{len(eligible_dates):>5,} dates | "
            f"{len(eligible_rows):>8,} observations"
        )

    # ========================================================
    # YEARLY REPRESENTATION
    # ========================================================

    print()
    print("=" * 70)
    print("YEARLY COHORT REPRESENTATION")
    print("=" * 70)

    valid["year"] = (
        pd.to_datetime(
            valid["date"]
        ).dt.year
    )

    yearly = (
        valid
        .groupby("year")["ticker"]
        .nunique()
        .reset_index(
            name="stocks_with_valid_2y"
        )
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # ========================================================
    # LOWEST-COHORT DATES
    # ========================================================

    print()
    print("=" * 70)
    print("LOWEST-COHORT DATES")
    print("=" * 70)

    lowest = cohort.nsmallest(
        20,
        "stock_count"
    )

    print(
        lowest.to_string(
            index=False
        )
    )

    # ========================================================
    # HIGHEST-COHORT DATES
    # ========================================================

    print()
    print("=" * 70)
    print("HIGHEST-COHORT DATES")
    print("=" * 70)

    highest = cohort.nlargest(
        10,
        "stock_count"
    )

    print(
        highest.to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("COHORT AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"Number of cohort dates : {len(cohort):,}"
    )

    print(
        f"Minimum cohort size    : "
        f"{cohort['stock_count'].min()}"
    )

    print(
        f"Median cohort size     : "
        f"{cohort['stock_count'].median():.0f}"
    )

    print(
        f"Maximum cohort size    : "
        f"{cohort['stock_count'].max()}"
    )

    print()
    print(
        "No scoring formula was created."
    )

    print(
        "No labels were created."
    )


if __name__ == "__main__":
    main()