from pathlib import Path
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
# LOAD
# ============================================================

print("=" * 80)
print("ETF 2Y RISK EXTREMES AUDIT")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"\nLoaded rows: {len(df):,}")
print(f"ETFs: {df['Ticker'].nunique()}")
print(
    f"Date range: "
    f"{df['Date'].min().date()} → {df['Date'].max().date()}"
)


# ============================================================
# VALID 2Y OBSERVATIONS
# ============================================================

valid = df[
    df["future_date_2y"].notna()
    & df["annualized_return_2y"].notna()
    & df["volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
    & df["risk_adjusted_return_2y"].notna()
].copy()

print(f"\nValid 2Y observations: {len(valid):,}")
print(f"Missing 2Y observations: {len(df) - len(valid):,}")


# ============================================================
# DISPLAY COLUMNS
# ============================================================

cols = [
    "Ticker",
    "Date",
    "future_date_2y",
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",
]


# ============================================================
# 1. LOWEST VOLATILITY
# ============================================================

print("\n" + "=" * 80)
print("1. LOWEST 2Y VOLATILITY")
print("=" * 80)

lowest_vol = valid.sort_values(
    "volatility_2y"
).head(20)

print(
    lowest_vol[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 2. HIGHEST VOLATILITY
# ============================================================

print("\n" + "=" * 80)
print("2. HIGHEST 2Y VOLATILITY")
print("=" * 80)

highest_vol = valid.sort_values(
    "volatility_2y",
    ascending=False
).head(20)

print(
    highest_vol[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 3. NEAR-ZERO MAX DRAWDOWN
# ============================================================

print("\n" + "=" * 80)
print("3. NEAR-ZERO 2Y MAX DRAWDOWN")
print("=" * 80)

# Drawdown is negative.
# Values close to zero mean the price remained close
# to its running peak during the 2-year window.

near_zero_dd = valid[
    valid["max_drawdown_2y"] >= -0.001
].sort_values(
    "max_drawdown_2y",
    ascending=False
).head(30)

print(
    near_zero_dd[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print(
    f"\nObservations with MDD >= -0.1%: "
    f"{(valid['max_drawdown_2y'] >= -0.001).sum():,}"
)

print(
    f"Observations with MDD >= -1%: "
    f"{(valid['max_drawdown_2y'] >= -0.01).sum():,}"
)


# ============================================================
# 4. WORST MAX DRAWDOWN
# ============================================================

print("\n" + "=" * 80)
print("4. WORST 2Y MAX DRAWDOWN")
print("=" * 80)

worst_dd = valid.sort_values(
    "max_drawdown_2y"
).head(20)

print(
    worst_dd[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 5. LOW-VOLATILITY CONCENTRATION
# ============================================================

print("\n" + "=" * 80)
print("5. LOW-VOLATILITY CONCENTRATION BY ETF")
print("=" * 80)

low_vol_thresholds = [0.01, 0.02, 0.05, 0.10]

for threshold in low_vol_thresholds:

    subset = valid[
        valid["volatility_2y"] < threshold
    ]

    print(
        f"\nVolatility < {threshold:.2%}: "
        f"{len(subset):,} observations"
    )

    if len(subset) > 0:

        counts = (
            subset["Ticker"]
            .value_counts()
            .head(20)
        )

        print(counts.to_string())


# ============================================================
# 6. NEAR-ZERO DRAWDOWN CONCENTRATION
# ============================================================

print("\n" + "=" * 80)
print("6. NEAR-ZERO DRAWDOWN CONCENTRATION BY ETF")
print("=" * 80)

dd_thresholds = [-0.001, -0.01, -0.05]

for threshold in dd_thresholds:

    subset = valid[
        valid["max_drawdown_2y"] >= threshold
    ]

    print(
        f"\nMDD >= {threshold:.2%}: "
        f"{len(subset):,} observations"
    )

    if len(subset) > 0:

        counts = (
            subset["Ticker"]
            .value_counts()
            .head(20)
        )

        print(counts.to_string())


# ============================================================
# 7. PER-ETF RISK SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("7. PER-ETF RISK SUMMARY")
print("=" * 80)

summary = (
    valid
    .groupby("Ticker")
    .agg(
        observations=("Ticker", "size"),

        median_volatility=(
            "volatility_2y",
            "median"
        ),

        min_volatility=(
            "volatility_2y",
            "min"
        ),

        max_volatility=(
            "volatility_2y",
            "max"
        ),

        median_drawdown=(
            "max_drawdown_2y",
            "median"
        ),

        best_drawdown=(
            "max_drawdown_2y",
            "max"
        ),

        worst_drawdown=(
            "max_drawdown_2y",
            "min"
        ),

        median_return=(
            "annualized_return_2y",
            "median"
        ),

        median_risk_adjusted=(
            "risk_adjusted_return_2y",
            "median"
        ),
    )
    .sort_values("median_volatility")
)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 8. POSSIBLE FLAT-PRICE WINDOWS
# ============================================================

print("\n" + "=" * 80)
print("8. POSSIBLE FLAT-PRICE WINDOWS")
print("=" * 80)

flat_candidates = valid[
    valid["volatility_2y"] < 0.005
].sort_values(
    "volatility_2y"
)

print(
    f"Observations with volatility < 0.5%: "
    f"{len(flat_candidates):,}"
)

if len(flat_candidates) > 0:

    print(
        flat_candidates[cols].head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 9. EXTREME RISK OBSERVATIONS + RETURN
# ============================================================

print("\n" + "=" * 80)
print("9. EXTREME RISK OBSERVATIONS + RETURN")
print("=" * 80)

print("\nLowest volatility:")

print(
    lowest_vol[
        [
            "Ticker",
            "Date",
            "annualized_return_2y",
            "volatility_2y",
            "max_drawdown_2y",
            "risk_adjusted_return_2y",
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nWorst drawdown:")

print(
    worst_dd[
        [
            "Ticker",
            "Date",
            "annualized_return_2y",
            "volatility_2y",
            "max_drawdown_2y",
            "risk_adjusted_return_2y",
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 10. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(
    f"\nTotal valid 2Y observations : {len(valid):,}"
)

print(
    f"Volatility < 1%             : "
    f"{(valid['volatility_2y'] < 0.01).sum():,}"
)

print(
    f"Volatility < 2%             : "
    f"{(valid['volatility_2y'] < 0.02).sum():,}"
)

print(
    f"Volatility < 5%             : "
    f"{(valid['volatility_2y'] < 0.05).sum():,}"
)

print(
    f"MDD >= -0.1%                : "
    f"{(valid['max_drawdown_2y'] >= -0.001).sum():,}"
)

print(
    f"MDD >= -1%                  : "
    f"{(valid['max_drawdown_2y'] >= -0.01).sum():,}"
)

print(
    f"MDD <= -50%                 : "
    f"{(valid['max_drawdown_2y'] <= -0.50).sum():,}"
)

print("\nAudit complete.")
print("=" * 80)