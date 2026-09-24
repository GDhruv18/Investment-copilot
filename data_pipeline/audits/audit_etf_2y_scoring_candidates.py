from pathlib import Path

import numpy as np
import pandas as pd


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("ETF 2Y+ SCORING CANDIDATE AUDIT")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

valid = df[
    df["annualized_return_2y"].notna()
    & df["volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
].copy()

print(f"\nTotal rows: {len(df):,}")
print(f"Valid 2Y observations: {len(valid):,}")
print(f"ETFs: {valid['Ticker'].nunique()}")


# ============================================================
# CROSS-SECTIONAL NORMALIZATION
# ============================================================
#
# IMPORTANT:
# We normalize within each observation date.
#
# This prevents the model target from simply rewarding
# historical market regimes where every ETF had high/low
# absolute returns.
#
# Higher percentile = better.
#
# Return:
#     higher is better
#
# Volatility:
#     lower is better
#
# Drawdown:
#     less negative is better
#
# Therefore volatility is inverted.
#
# ============================================================

valid["return_pct"] = (
    valid.groupby("Date")["annualized_return_2y"]
    .rank(pct=True)
)

valid["volatility_pct"] = (
    valid.groupby("Date")["volatility_2y"]
    .rank(pct=True)
)

valid["drawdown_pct"] = (
    valid.groupby("Date")["max_drawdown_2y"]
    .rank(pct=True)
)

# Invert volatility because lower volatility is better.
valid["risk_pct"] = 1.0 - valid["volatility_pct"]


# ============================================================
# CANDIDATE SCORING FORMULAS
# ============================================================

candidate_weights = {
    "A_Return50_Risk25_DD25": {
        "return": 0.50,
        "risk": 0.25,
        "drawdown": 0.25,
    },

    "B_Return60_Risk20_DD20": {
        "return": 0.60,
        "risk": 0.20,
        "drawdown": 0.20,
    },

    "C_Return50_Risk30_DD20": {
        "return": 0.50,
        "risk": 0.30,
        "drawdown": 0.20,
    },

    "D_Return40_Risk30_DD30": {
        "return": 0.40,
        "risk": 0.30,
        "drawdown": 0.30,
    },
}


for name, weights in candidate_weights.items():

    valid[name] = (
        weights["return"] * valid["return_pct"]
        + weights["risk"] * valid["risk_pct"]
        + weights["drawdown"] * valid["drawdown_pct"]
    )


# ============================================================
# BASIC SCORE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("1. SCORE DISTRIBUTIONS")
print("=" * 80)

for name in candidate_weights:

    print(f"\n{name}")

    print(
        valid[name].describe(
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
# TOP 30% / MIDDLE 40% / BOTTOM 30%
# ============================================================

print("\n" + "=" * 80)
print("2. CANDIDATE LABEL DISTRIBUTIONS")
print("=" * 80)

for name in candidate_weights:

    labels = pd.Series(
        np.select(
            [
                valid[name] >= valid.groupby("Date")[name]
                .transform("quantile", 0.70),

                valid[name] <= valid.groupby("Date")[name]
                .transform("quantile", 0.30),
            ],
            [
                2,
                0,
            ],
            default=1,
        ),
        index=valid.index,
    )

    print(f"\n{name}")

    print(
        labels.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Weak",
                1: "Neutral",
                2: "Attractive",
            }
        )
        .to_string()
    )

    print("\nPercentages:")

    print(
        (
            labels.value_counts(normalize=True)
            .sort_index()
            .rename(
                index={
                    0: "Weak",
                    1: "Neutral",
                    2: "Attractive",
                }
            )
            * 100
        ).round(2).to_string()
    )


# ============================================================
# TOP ETF FREQUENCY
# ============================================================

print("\n" + "=" * 80)
print("3. TOP-30% ETF CONCENTRATION")
print("=" * 80)

for name in candidate_weights:

    threshold = (
        valid.groupby("Date")[name]
        .transform("quantile", 0.70)
    )

    top = valid[
        valid[name] >= threshold
    ]

    counts = (
        top["Ticker"]
        .value_counts()
        .head(15)
    )

    print(f"\n{name}")

    print(counts.to_string())


# ============================================================
# LIQUID ETF BEHAVIOR
# ============================================================

print("\n" + "=" * 80)
print("4. LIQUID ETF BEHAVIOR")
print("=" * 80)

liquid_tickers = [
    "LIQUIDBEES.NS",
    "LIQUIDCASE.NS",
]

liquid = valid[
    valid["Ticker"].isin(liquid_tickers)
].copy()

for ticker in liquid_tickers:

    ticker_data = liquid[
        liquid["Ticker"] == ticker
    ]

    print(f"\n{ticker}")
    print(f"Observations: {len(ticker_data):,}")

    for name in candidate_weights:

        threshold = (
            valid.groupby("Date")[name]
            .transform("quantile", 0.70)
        )

        # Align threshold with valid index.
        ticker_threshold = threshold.loc[ticker_data.index]

        top_count = (
            ticker_data[name] >= ticker_threshold
        ).sum()

        percentage = (
            top_count / len(ticker_data) * 100
            if len(ticker_data) > 0
            else 0
        )

        print(
            f"{name}: "
            f"top 30% = {top_count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# CORRELATION BETWEEN CANDIDATE SCORES
# ============================================================

print("\n" + "=" * 80)
print("5. CORRELATION BETWEEN CANDIDATE SCORES")
print("=" * 80)

score_columns = list(candidate_weights.keys())

correlation = valid[score_columns].corr(
    method="spearman"
)

print(
    correlation.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# RANK AGREEMENT
# ============================================================

print("\n" + "=" * 80)
print("6. TOP-30% RANK AGREEMENT")
print("=" * 80)

top_masks = {}

for name in candidate_weights:

    threshold = (
        valid.groupby("Date")[name]
        .transform("quantile", 0.70)
    )

    top_masks[name] = (
        valid[name] >= threshold
    )


for i in range(len(score_columns)):

    for j in range(i + 1, len(score_columns)):

        a = score_columns[i]
        b = score_columns[j]

        agreement = (
            top_masks[a] & top_masks[b]
        ).sum()

        either = (
            top_masks[a] | top_masks[b]
        ).sum()

        jaccard = (
            agreement / either
            if either > 0
            else np.nan
        )

        print(
            f"{a} vs {b}: "
            f"Jaccard = {jaccard:.4f}"
        )


# ============================================================
# SCORE COMPONENT CONTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("7. COMPONENT CONTRIBUTIONS")
print("=" * 80)

component_columns = [
    "return_pct",
    "risk_pct",
    "drawdown_pct",
]

print(
    valid[component_columns]
    .describe(
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
    .to_string()
)


# ============================================================
# EXAMPLE EXTREME OBSERVATIONS
# ============================================================

print("\n" + "=" * 80)
print("8. EXTREME OBSERVATIONS UNDER EACH CANDIDATE")
print("=" * 80)

display_columns = [
    "Ticker",
    "Date",
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
    "return_pct",
    "risk_pct",
    "drawdown_pct",
]

for name in candidate_weights:

    print(f"\n{name} — TOP 10")

    print(
        valid.sort_values(
            name,
            ascending=False
        )[display_columns + [name]]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# PER-ETF MEDIAN SCORES
# ============================================================

print("\n" + "=" * 80)
print("9. MEDIAN SCORE BY ETF")
print("=" * 80)

median_scores = (
    valid
    .groupby("Ticker")[score_columns]
    .median()
)

for name in score_columns:

    print(f"\n{name}")

    print(
        median_scores[name]
        .sort_values(ascending=False)
        .to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"\nCandidate formulas tested: "
    f"{len(candidate_weights)}"
)

print(
    "Normalization: cross-sectional percentile "
    "within each observation date"
)

print(
    "Return direction: higher is better"
)

print(
    "Volatility direction: lower is better"
)

print(
    "Drawdown direction: less negative is better"
)

print(
    "\nNo target labels were created."
)

print(
    "No source data was modified."
)

print("\nAudit complete.")
print("=" * 80)