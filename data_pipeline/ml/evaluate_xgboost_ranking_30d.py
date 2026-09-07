from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FOLDER = (
    PROJECT_ROOT
    / "data_pipeline"
    / "ml"
    / "output"
    / "models"
)

PREDICTION_FILE = (
    MODEL_FOLDER
    / "xgboost_stock_30d_validation_predictions.csv"
)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

print("=" * 70)
print("LOADING XGBOOST VALIDATION PREDICTIONS")
print("=" * 70)

df = pd.read_csv(PREDICTION_FILE)

print(f"Validation rows: {len(df):,}")
print(f"Stocks: {df['ticker'].nunique()}")
print(
    f"Date range: "
    f"{df['date'].min()} → {df['date'].max()}"
)


# ============================================================
# CALCULATE CROSS-SECTIONAL RANKS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CROSS-SECTIONAL RANKINGS")
print("=" * 70)

# Rank stocks for each date based on predicted return.
# Higher predicted return = better rank.
df["predicted_rank"] = (
    df.groupby("date")["predicted_30d"]
    .rank(
        ascending=False,
        method="average"
    )
)

# Rank stocks for each date based on actual future return.
df["actual_rank"] = (
    df.groupby("date")["target_30d"]
    .rank(
        ascending=False,
        method="average"
    )
)


# ============================================================
# NORMALIZED RANKS
# ============================================================

# Convert ranks into percentiles.
#
# 1.0 = highest-ranked stock
# 0.0 = lowest-ranked stock

def normalized_rank(group):
    n = len(group)

    if n <= 1:
        return pd.Series(
            np.full(n, 0.5),
            index=group.index
        )

    return 1 - (
        group.rank(
            ascending=True,
            method="average"
        ) - 1
    ) / (n - 1)


df["predicted_percentile"] = (
    df.groupby("date")["predicted_30d"]
    .transform(normalized_rank)
)

df["actual_percentile"] = (
    df.groupby("date")["target_30d"]
    .transform(normalized_rank)
)


# ============================================================
# DAILY RANK CORRELATION
# ============================================================

daily_correlations = []

for date, group in df.groupby("date"):

    if len(group) < 2:
        continue

    correlation = (
        group["predicted_percentile"]
        .corr(group["actual_percentile"])
    )

    if pd.notna(correlation):
        daily_correlations.append(correlation)


daily_correlations = np.array(daily_correlations)


# ============================================================
# INFORMATION COEFFICIENT
# ============================================================

mean_rank_correlation = (
    daily_correlations.mean()
    if len(daily_correlations) > 0
    else np.nan
)

median_rank_correlation = (
    np.median(daily_correlations)
    if len(daily_correlations) > 0
    else np.nan
)

positive_days = (
    (daily_correlations > 0).mean()
    if len(daily_correlations) > 0
    else np.nan
)


# ============================================================
# TOP-N PORTFOLIO TEST
# ============================================================

print("\n" + "=" * 70)
print("TOP-N RANKING TEST")
print("=" * 70)

top_n = 10

top_returns = []

for date, group in df.groupby("date"):

    group = group.sort_values(
        "predicted_30d",
        ascending=False
    )

    top_stocks = group.head(top_n)

    if len(top_stocks) == 0:
        continue

    average_return = top_stocks["target_30d"].mean()

    top_returns.append(average_return)


top_returns = np.array(top_returns)


# ============================================================
# ALL-STOCK BENCHMARK
# ============================================================

all_stock_returns = (
    df.groupby("date")["target_30d"]
    .mean()
    .values
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SECTIONAL RANKING RESULTS")
print("=" * 70)

print(
    f"Trading dates evaluated:       "
    f"{len(daily_correlations):,}"
)

print(
    f"Mean rank correlation (IC):     "
    f"{mean_rank_correlation:.6f}"
)

print(
    f"Median rank correlation:        "
    f"{median_rank_correlation:.6f}"
)

print(
    f"Positive-IC days:               "
    f"{positive_days:.2%}"
)

print(
    f"\nTop-{top_n} average 30D return: "
    f"{top_returns.mean():.4%}"
)

print(
    f"All-stock average 30D return:   "
    f"{all_stock_returns.mean():.4%}"
)


# ============================================================
# TOP-N OUTPERFORMANCE
# ============================================================

outperformance = (
    top_returns.mean()
    - all_stock_returns.mean()
)

print(
    f"\nTop-{top_n} outperformance:      "
    f"{outperformance:.4%}"
)


# ============================================================
# INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
Information Coefficient (IC):
Measures whether stocks ranked highly by the model
actually tend to have higher future returns.

IC > 0:
    Positive ranking relationship.

IC ≈ 0:
    Little useful ranking information.

IC < 0:
    Model ranking tends to be wrong.

Top-N test:
Measures the average future return of the stocks
that the model ranks highest.
"""
)

print("\nCross-sectional ranking evaluation completed.")