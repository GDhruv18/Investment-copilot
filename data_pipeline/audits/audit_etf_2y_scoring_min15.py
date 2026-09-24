from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# ETF 2Y SCORING AUDIT — MINIMUM COHORT SIZE = 15
# ============================================================

INPUT_FILE = Path("data/targets/etf_2y_diagnostics.parquet")

MIN_COHORT = 15

print("=" * 80)
print("ETF 2Y SCORING CANDIDATES — MINIMUM COHORT SIZE = 15")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

required_columns = [
    "Ticker",
    "Date",
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Date"] = pd.to_datetime(df["Date"])

valid = df[
    df["annualized_return_2y"].notna()
    & df["volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
].copy()

print(f"\nTotal rows: {len(df):,}")
print(f"Valid 2Y rows before cohort filter: {len(valid):,}")
print(f"ETFs before cohort filter: {valid['Ticker'].nunique()}")


# ============================================================
# 2. CALCULATE COHORT SIZE
# ============================================================

cohort_size = (
    valid.groupby("Date")["Ticker"]
    .nunique()
    .rename("cohort_size")
)

valid = valid.join(cohort_size, on="Date")

eligible = valid[valid["cohort_size"] >= MIN_COHORT].copy()

print("\n" + "=" * 80)
print("COHORT FILTER")
print("=" * 80)

print(f"Minimum cohort size: {MIN_COHORT}")
print(f"Eligible dates: {eligible['Date'].nunique():,}")
print(f"Eligible observations: {len(eligible):,}")
print(f"ETFs represented: {eligible['Ticker'].nunique()}")


# ============================================================
# 3. CROSS-SECTIONAL PERCENTILES
# ============================================================

# Higher return = better
eligible["return_pct"] = (
    eligible.groupby("Date")["annualized_return_2y"]
    .rank(method="average", pct=True)
)

# Lower volatility = better
eligible["volatility_pct"] = (
    eligible.groupby("Date")["volatility_2y"]
    .rank(method="average", pct=True)
)

eligible["risk_pct"] = 1.0 - eligible["volatility_pct"]


# Less negative drawdown = better
eligible["drawdown_pct"] = (
    eligible.groupby("Date")["max_drawdown_2y"]
    .rank(method="average", pct=True)
)


# ============================================================
# 4. CANDIDATE SCORING METHODS
# ============================================================

# Candidate A:
# Return 50%
# Risk   25%
# Drawdown 25%
eligible["score_A"] = (
    0.50 * eligible["return_pct"]
    + 0.25 * eligible["risk_pct"]
    + 0.25 * eligible["drawdown_pct"]
)

# Candidate B:
# Return 60%
# Risk   20%
# Drawdown 20%
eligible["score_B"] = (
    0.60 * eligible["return_pct"]
    + 0.20 * eligible["risk_pct"]
    + 0.20 * eligible["drawdown_pct"]
)

# Candidate C:
# Return 50%
# Risk   30%
# Drawdown 20%
eligible["score_C"] = (
    0.50 * eligible["return_pct"]
    + 0.30 * eligible["risk_pct"]
    + 0.20 * eligible["drawdown_pct"]
)


# ============================================================
# 5. SCORE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("SCORE DISTRIBUTIONS")
print("=" * 80)

for candidate in ["A", "B", "C"]:

    s = eligible[f"score_{candidate}"]

    print(f"\nCandidate {candidate}")

    print(f"Mean   : {s.mean():.6f}")
    print(f"Std    : {s.std():.6f}")
    print(f"Min    : {s.min():.6f}")
    print(f"25%    : {s.quantile(0.25):.6f}")
    print(f"Median : {s.median():.6f}")
    print(f"75%    : {s.quantile(0.75):.6f}")
    print(f"Max    : {s.max():.6f}")


# ============================================================
# 6. 30 / 40 / 30 LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("30 / 40 / 30 LABEL DISTRIBUTION")
print("=" * 80)

for candidate in ["A", "B", "C"]:

    score_col = f"score_{candidate}"
    label_col = f"label_{candidate}"

    eligible[label_col] = (
        eligible.groupby("Date")[score_col]
        .transform(
            lambda x: pd.qcut(
                x,
                q=[0, 0.30, 0.70, 1.0],
                labels=["Weak", "Neutral", "Attractive"],
                duplicates="drop"
            )
        )
    )

    counts = eligible[label_col].value_counts()

    total = counts.sum()

    print(f"\nCandidate {candidate}")

    for label in ["Weak", "Neutral", "Attractive"]:
        count = counts.get(label, 0)
        pct = count / total * 100

        print(
            f"{label:<12}: "
            f"{count:>8,} "
            f"({pct:>6.2f}%)"
        )


# ============================================================
# 7. ETF CONCENTRATION IN TOP 30%
# ============================================================

print("\n" + "=" * 80)
print("TOP 30% ETF CONCENTRATION")
print("=" * 80)

for candidate in ["A", "B", "C"]:

    label_col = f"label_{candidate}"

    top = eligible[
        eligible[label_col] == "Attractive"
    ]

    counts = (
        top["Ticker"]
        .value_counts()
        .head(15)
    )

    print(f"\nCandidate {candidate}")

    for ticker, count in counts.items():

        total_ticker = (
            eligible["Ticker"] == ticker
        ).sum()

        pct = count / total_ticker * 100

        print(
            f"{ticker:<18} "
            f"{count:>6,} / {total_ticker:>6,} "
            f"({pct:>6.2f}%)"
        )


# ============================================================
# 8. LIQUID ETF BEHAVIOR
# ============================================================

print("\n" + "=" * 80)
print("LIQUID ETF BEHAVIOR")
print("=" * 80)

for ticker in ["LIQUIDBEES.NS", "LIQUIDCASE.NS"]:

    subset = eligible[
        eligible["Ticker"] == ticker
    ]

    if subset.empty:
        print(f"\n{ticker}: no eligible observations")
        continue

    print(f"\n{ticker}")
    print(f"Eligible observations: {len(subset):,}")

    for candidate in ["A", "B", "C"]:

        label_col = f"label_{candidate}"

        attractive_pct = (
            (subset[label_col] == "Attractive").mean()
            * 100
        )

        neutral_pct = (
            (subset[label_col] == "Neutral").mean()
            * 100
        )

        weak_pct = (
            (subset[label_col] == "Weak").mean()
            * 100
        )

        print(
            f"Candidate {candidate}: "
            f"Weak {weak_pct:.2f}% | "
            f"Neutral {neutral_pct:.2f}% | "
            f"Attractive {attractive_pct:.2f}%"
        )


# ============================================================
# 9. SCORE CORRELATIONS
# ============================================================

print("\n" + "=" * 80)
print("SCORE CORRELATIONS")
print("=" * 80)

corr = eligible[
    ["score_A", "score_B", "score_C"]
].corr()

print(corr.to_string(float_format=lambda x: f"{x:.6f}"))


# ============================================================
# 10. TOP-30% JACCARD SIMILARITY
# ============================================================

print("\n" + "=" * 80)
print("TOP-30% JACCARD SIMILARITY")
print("=" * 80)


def top30_set(group, candidate):

    label_col = f"label_{candidate}"

    return set(
        group.loc[
            group[label_col] == "Attractive",
            "Ticker"
        ]
    )


for c1, c2 in [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
]:

    intersections = []
    unions = []

    for _, group in eligible.groupby("Date"):

        set1 = top30_set(group, c1)
        set2 = top30_set(group, c2)

        union = set1 | set2

        if union:
            intersections.append(
                len(set1 & set2)
            )
            unions.append(
                len(union)
            )

    jaccard = (
        sum(intersections) /
        sum(unions)
        if sum(unions) > 0
        else np.nan
    )

    print(
        f"{c1} vs {c2}: "
        f"{jaccard:.6f}"
    )


# ============================================================
# 11. PER-ETF MEDIAN SCORES
# ============================================================

print("\n" + "=" * 80)
print("PER-ETF MEDIAN SCORES")
print("=" * 80)

for candidate in ["A", "B", "C"]:

    summary = (
        eligible
        .groupby("Ticker")[f"score_{candidate}"]
        .median()
        .sort_values(ascending=False)
    )

    print(f"\nCandidate {candidate}")

    print(
        summary.head(15).to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nLowest 10:")

    print(
        summary.tail(10).to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 12. EXTREME TOP OBSERVATIONS
# ============================================================

print("\n" + "=" * 80)
print("EXTREME TOP OBSERVATIONS")
print("=" * 80)

for candidate in ["A", "B", "C"]:

    print(f"\nCandidate {candidate}")

    cols = [
        "Date",
        "Ticker",
        "cohort_size",
        "annualized_return_2y",
        "volatility_2y",
        "max_drawdown_2y",
        f"score_{candidate}",
    ]

    extreme = (
        eligible
        .nlargest(20, f"score_{candidate}")[cols]
    )

    print(
        extreme.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 13. COHORT-SIZE EFFECT
# ============================================================

print("\n" + "=" * 80)
print("SCORE BEHAVIOR BY COHORT SIZE")
print("=" * 80)

eligible["cohort_bucket"] = pd.cut(
    eligible["cohort_size"],
    bins=[14, 15, 19, 24, 29, 34, 39, 40],
    labels=[
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40",
        "40+",
    ],
    include_lowest=True
)

bucket_summary = (
    eligible
    .groupby("cohort_bucket", observed=True)
    .agg(
        observations=("Ticker", "size"),
        dates=("Date", "nunique"),
        median_A=("score_A", "median"),
        median_B=("score_B", "median"),
        median_C=("score_C", "median"),
    )
)

print(
    bucket_summary.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"Minimum cohort used: {MIN_COHORT}")
print(f"Eligible dates: {eligible['Date'].nunique():,}")
print(f"Eligible observations: {len(eligible):,}")
print(f"ETFs represented: {eligible['Ticker'].nunique()}")

print("\nCandidate formulas:")
print("A = Return 50% + Risk 25% + Drawdown 25%")
print("B = Return 60% + Risk 20% + Drawdown 20%")
print("C = Return 50% + Risk 30% + Drawdown 20%")

print("\nNo source files were modified.")
print("No final 2Y labels were generated.")

print("=" * 80)
print("Audit complete.")
print("=" * 80)