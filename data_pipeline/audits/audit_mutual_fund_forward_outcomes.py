from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "mutual_fund_forward_outcomes.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "targets"
    / "mutual_fund_forward_outcome_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_distribution(series, name):
    valid = series.dropna()

    print(f"\n{name} distribution:")
    print(
        valid.describe(
            percentiles=[
                0.001,
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                0.999,
            ]
        )
    )


def print_extreme_counts(df, column, name):
    valid = df[column].dropna()

    print(f"\n{name} extreme counts:")

    thresholds = [
        (1.0, "> +100%"),
        (2.0, "> +200%"),
        (5.0, "> +500%"),
        (10.0, "> +1,000%"),
        (20.0, "> +2,000%"),
        (50.0, "> +5,000%"),
        (100.0, "> +10,000%"),
    ]

    for threshold, label in thresholds:
        count = (valid > threshold).sum()
        print(f"{label:>12}: {count:,}")

    negative_thresholds = [
        (-0.10, "< -10%"),
        (-0.25, "< -25%"),
        (-0.50, "< -50%"),
        (-0.75, "< -75%"),
        (-0.90, "< -90%"),
    ]

    for threshold, label in negative_thresholds:
        count = (valid < threshold).sum()
        print(f"{label:>12}: {count:,}")


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 70)
    print("MUTUAL FUND FORWARD OUTCOME ANOMALY AUDIT")
    print("=" * 70)

    print(f"\nInput: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(
        f"Schemes: {df['scheme_code'].nunique()}"
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    for column in [
        "target_date_6m",
        "future_date_6m",
        "target_date_1y",
        "future_date_1y",
    ]:
        df[column] = pd.to_datetime(
            df[column]
        ).dt.normalize()

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def basic_validation(df):

    print("\n" + "=" * 70)
    print("BASIC VALIDATION")
    print("=" * 70)

    assert df["scheme_code"].nunique() == 100

    duplicate_count = df.duplicated(
        ["scheme_code", "date"]
    ).sum()

    print(
        f"Duplicate scheme/date rows: {duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate scheme/date rows found."
        )

    invalid_nav = (
        df["nav"].isna()
        | ~np.isfinite(df["nav"])
        | (df["nav"] <= 0)
    ).sum()

    print(
        f"Invalid NAV rows: {invalid_nav}"
    )

    if invalid_nav != 0:
        raise RuntimeError(
            "Invalid NAV values found."
        )

    print("Basic validation: PASS")


# ============================================================
# EXTREME OBSERVATIONS
# ============================================================

def extreme_observations(df):

    print("\n" + "=" * 70)
    print("EXTREME FORWARD RETURNS")
    print("=" * 70)

    for column, name in [
        ("forward_return_6m", "6M"),
        ("forward_return_1y", "1Y"),
    ]:

        print_extreme_counts(
            df,
            column,
            name
        )

        print(f"\nTop 20 {name} returns:")

        columns = [
            "scheme_code",
            "date",
            "nav",
            f"target_date_{'6m' if name == '6M' else '1y'}",
            f"future_date_{'6m' if name == '6M' else '1y'}",
            column,
        ]

        top = (
            df.loc[
                df[column].notna(),
                columns
            ]
            .sort_values(
                column,
                ascending=False
            )
            .head(20)
        )

        print(
            top.to_string(
                index=False
            )
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

def distributions(df):

    print("\n" + "=" * 70)
    print("OVERALL DISTRIBUTIONS")
    print("=" * 70)

    print_distribution(
        df["forward_return_6m"],
        "6M forward return"
    )

    print_distribution(
        df["forward_return_1y"],
        "1Y forward return"
    )


# ============================================================
# SCHEME-LEVEL STATISTICS
# ============================================================

def scheme_statistics(df):

    print("\n" + "=" * 70)
    print("SCHEME-LEVEL STATISTICS")
    print("=" * 70)

    rows = []

    for scheme_code, group in df.groupby(
        "scheme_code",
        sort=True
    ):

        returns_6m = group[
            "forward_return_6m"
        ].dropna()

        returns_1y = group[
            "forward_return_1y"
        ].dropna()

        row = {
            "scheme_code": scheme_code,

            "rows": len(group),

            "valid_6m": len(returns_6m),
            "valid_1y": len(returns_1y),

            "median_6m": (
                returns_6m.median()
                if len(returns_6m)
                else np.nan
            ),

            "median_1y": (
                returns_1y.median()
                if len(returns_1y)
                else np.nan
            ),

            "max_6m": (
                returns_6m.max()
                if len(returns_6m)
                else np.nan
            ),

            "max_1y": (
                returns_1y.max()
                if len(returns_1y)
                else np.nan
            ),

            "min_6m": (
                returns_6m.min()
                if len(returns_6m)
                else np.nan
            ),

            "min_1y": (
                returns_1y.min()
                if len(returns_1y)
                else np.nan
            ),

            "gt_100pct_6m": (
                (returns_6m > 1.0).sum()
            ),

            "gt_500pct_6m": (
                (returns_6m > 5.0).sum()
            ),

            "gt_1000pct_6m": (
                (returns_6m > 10.0).sum()
            ),

            "gt_100pct_1y": (
                (returns_1y > 1.0).sum()
            ),

            "gt_500pct_1y": (
                (returns_1y > 5.0).sum()
            ),

            "gt_1000pct_1y": (
                (returns_1y > 10.0).sum()
            ),
        }

        rows.append(row)

    stats = pd.DataFrame(rows)

    print(
        "\nSchemes with 6M return > +100%:"
    )

    print(
        stats.loc[
            stats["gt_100pct_6m"] > 0,
            [
                "scheme_code",
                "valid_6m",
                "max_6m",
                "gt_100pct_6m",
                "gt_500pct_6m",
                "gt_1000pct_6m",
            ],
        ]
        .sort_values(
            "max_6m",
            ascending=False
        )
        .to_string(index=False)
    )

    print(
        "\nSchemes with 1Y return > +100%:"
    )

    print(
        stats.loc[
            stats["gt_100pct_1y"] > 0,
            [
                "scheme_code",
                "valid_1y",
                "max_1y",
                "gt_100pct_1y",
                "gt_500pct_1y",
                "gt_1000pct_1y",
            ],
        ]
        .sort_values(
            "max_1y",
            ascending=False
        )
        .to_string(index=False)
    )

    return stats


# ============================================================
# EXTREME RETURN DATE CONCENTRATION
# ============================================================

def extreme_date_analysis(df):

    print("\n" + "=" * 70)
    print("EXTREME RETURN DATE CONCENTRATION")
    print("=" * 70)

    for column, name in [
        ("forward_return_6m", "6M"),
        ("forward_return_1y", "1Y"),
    ]:

        extreme = df.loc[
            df[column] > 1.0,
            [
                "scheme_code",
                "date",
                column,
            ],
        ].copy()

        print(
            f"\n{name}: observations > +100%: "
            f"{len(extreme):,}"
        )

        if extreme.empty:
            continue

        date_counts = (
            extreme
            .groupby("date")
            .size()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        print(
            f"\nTop dates contributing to "
            f"{name} > +100%:"
        )

        print(
            date_counts.to_string()
        )


# ============================================================
# NAV JUMP DIAGNOSTICS
# ============================================================

def nav_jump_analysis(df):

    print("\n" + "=" * 70)
    print("NAV JUMP DIAGNOSTICS")
    print("=" * 70)

    work = df[
        [
            "scheme_code",
            "date",
            "nav",
        ]
    ].copy()

    work = work.sort_values(
        ["scheme_code", "date"]
    )

    work["daily_nav_return"] = (
        work
        .groupby("scheme_code")["nav"]
        .pct_change()
    )

    print(
        "\nDaily NAV changes exceeding +50%:"
    )

    positive = work.loc[
        work["daily_nav_return"] > 0.50,
        [
            "scheme_code",
            "date",
            "nav",
            "daily_nav_return",
        ],
    ].sort_values(
        "daily_nav_return",
        ascending=False
    )

    print(
        positive.head(50).to_string(
            index=False
        )
    )

    print(
        f"\nTotal daily NAV changes > +50%: "
        f"{len(positive):,}"
    )

    print(
        "\nDaily NAV changes below -50%:"
    )

    negative = work.loc[
        work["daily_nav_return"] < -0.50,
        [
            "scheme_code",
            "date",
            "nav",
            "daily_nav_return",
        ],
    ].sort_values(
        "daily_nav_return"
    )

    print(
        negative.head(50).to_string(
            index=False
        )
    )

    print(
        f"\nTotal daily NAV changes < -50%: "
        f"{len(negative):,}"
    )


# ============================================================
# SAVE SCHEME AUDIT
# ============================================================

def save_scheme_audit(stats):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stats.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("SCHEME AUDIT SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    basic_validation(
        df
    )

    distributions(
        df
    )

    extreme_observations(
        df
    )

    stats = scheme_statistics(
        df
    )

    extreme_date_analysis(
        df
    )

    nav_jump_analysis(
        df
    )

    save_scheme_audit(
        stats
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nNo observations were deleted or modified."
    )

    print(
        "No clipping or outlier treatment has been applied."
    )

    print(
        "FINAL STATUS: DIAGNOSTIC AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()