from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "stock_features.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "features"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "stock_feature_correlation_matrix.parquet"
)

HIGH_CORRELATION_FILE = (
    OUTPUT_DIR
    / "stock_high_correlation_pairs.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CORRELATION_THRESHOLD = 0.90

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-12


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",

    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",

    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",

    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_sma_200",

    "volume_ratio_20d",
    "volume_ratio_60d",
    "volume_change_20d",

    "range_20d",
    "range_60d",

    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",

    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOCK FEATURE REDUNDANCY AUDIT")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_parquet(FEATURE_FILE)

    print(
        f"Rows loaded       : {len(df):,}"
    )

    print(
        f"Stocks             : {df['ticker'].nunique()}"
    )

    print(
        f"Date range         : "
        f"{df['date'].min()} → {df['date'].max()}"
    )

    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    missing = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing feature columns:\n"
            + "\n".join(missing)
        )

    print(
        f"Feature columns    : {len(FEATURE_COLUMNS)}"
    )

    # ========================================================
    # NUMERIC DATA
    # ========================================================

    X = df[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ========================================================
    # BASIC FEATURE STATISTICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("1. FEATURE BASIC STATISTICS")
    print("=" * 70)

    stats = pd.DataFrame(index=FEATURE_COLUMNS)

    stats["count"] = X.count()

    stats["missing"] = X.isna().sum()

    stats["missing_pct"] = (
        stats["missing"]
        / len(X)
        * 100
    )

    stats["unique"] = X.nunique(
        dropna=True
    )

    stats["mean"] = X.mean()

    stats["std"] = X.std()

    stats["min"] = X.min()

    stats["median"] = X.median()

    stats["max"] = X.max()

    stats["skew"] = X.skew()

    stats["zero_count"] = (
        (X == 0).sum()
    )

    stats["zero_pct"] = (
        stats["zero_count"]
        / len(X)
        * 100
    )

    print(
        stats.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ========================================================
    # NEAR-ZERO VARIANCE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("2. NEAR-ZERO VARIANCE FEATURES")
    print("=" * 70)

    near_zero = stats[
        stats["std"]
        <= NEAR_ZERO_VARIANCE_THRESHOLD
    ]

    if len(near_zero) == 0:

        print(
            "No near-zero variance features."
        )

    else:

        print(
            near_zero.to_string()
        )

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("3. PEARSON CORRELATION")
    print("=" * 70)

    pearson = X.corr(
        method="pearson"
    )

    pearson.to_parquet(
        CORRELATION_FILE
    )

    print(
        f"Saved Pearson matrix:"
        f"\n{CORRELATION_FILE}"
    )

    # ========================================================
    # SPEARMAN CORRELATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("4. SPEARMAN CORRELATION")
    print("=" * 70)

    spearman = X.corr(
        method="spearman"
    )

    # ========================================================
    # HIGH CORRELATION PAIRS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        f"5. HIGH CORRELATION PAIRS "
        f"| ABS(CORR) >= {CORRELATION_THRESHOLD}"
    )
    print("=" * 70)

    pairs = []

    for i in range(
        len(FEATURE_COLUMNS)
    ):

        feature_a = FEATURE_COLUMNS[i]

        for j in range(
            i + 1,
            len(FEATURE_COLUMNS)
        ):

            feature_b = FEATURE_COLUMNS[j]

            pearson_value = pearson.loc[
                feature_a,
                feature_b
            ]

            spearman_value = spearman.loc[
                feature_a,
                feature_b
            ]

            max_abs_corr = max(
                abs(pearson_value),
                abs(spearman_value)
            )

            if (
                max_abs_corr
                >= CORRELATION_THRESHOLD
            ):

                pairs.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                    "max_abs_correlation":
                        max_abs_corr,
                })

    high_corr = pd.DataFrame(
        pairs
    )

    if len(high_corr) > 0:

        high_corr = high_corr.sort_values(
            "max_abs_correlation",
            ascending=False
        ).reset_index(drop=True)

        print(
            high_corr.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}"
            )
        )

    else:

        print(
            "No feature pairs exceed "
            f"|correlation| >= "
            f"{CORRELATION_THRESHOLD}."
        )

    high_corr.to_csv(
        HIGH_CORRELATION_FILE,
        index=False
    )

    print(
        f"\nSaved high-correlation pairs:"
        f"\n{HIGH_CORRELATION_FILE}"
    )

    # ========================================================
    # TOP CORRELATIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("6. TOP 20 ABSOLUTE PEARSON CORRELATIONS")
    print("=" * 70)

    top_pairs = []

    for i in range(
        len(FEATURE_COLUMNS)
    ):

        feature_a = FEATURE_COLUMNS[i]

        for j in range(
            i + 1,
            len(FEATURE_COLUMNS)
        ):

            feature_b = FEATURE_COLUMNS[j]

            value = pearson.loc[
                feature_a,
                feature_b
            ]

            top_pairs.append({
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": value,
                "abs_correlation":
                    abs(value),
            })

    top_pairs_df = pd.DataFrame(
        top_pairs
    )

    top_pairs_df = (
        top_pairs_df
        .sort_values(
            "abs_correlation",
            ascending=False
        )
        .head(20)
        .reset_index(drop=True)
    )

    print(
        top_pairs_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # PERCENTILE FEATURE CORRELATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("7. PERCENTILE FEATURE CORRELATION")
    print("=" * 70)

    percentile_features = [
        col
        for col in FEATURE_COLUMNS
        if col.endswith("_pct")
    ]

    percentile_corr = X[
        percentile_features
    ].corr(
        method="spearman"
    )

    print(
        percentile_corr.to_string(
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # RAW VS PERCENTILE RELATIONSHIPS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("8. RAW FEATURE vs PERCENTILE FEATURE RELATIONSHIPS")
    print("=" * 70)

    raw_percentile_pairs = [
        (
            "return_20d",
            "return_20d_pct"
        ),
        (
            "return_60d",
            "return_60d_pct"
        ),
        (
            "return_120d",
            "return_120d_pct"
        ),
        (
            "return_252d",
            "return_252d_pct"
        ),
        (
            "volatility_60d",
            "volatility_60d_pct"
        ),
        (
            "drawdown_120d",
            "drawdown_120d_pct"
        ),
        (
            "volume_ratio_20d",
            "volume_ratio_20d_pct"
        ),
    ]

    for raw_feature, pct_feature in raw_percentile_pairs:

        pair_df = X[
            [
                raw_feature,
                pct_feature,
            ]
        ].dropna()

        if len(pair_df) == 0:

            print(
                f"{raw_feature:<25} "
                f"{pct_feature:<30} "
                f"NO VALID DATA"
            )

            continue

        pearson_value = (
            pair_df[raw_feature]
            .corr(
                pair_df[pct_feature],
                method="pearson"
            )
        )

        spearman_value = (
            pair_df[raw_feature]
            .corr(
                pair_df[pct_feature],
                method="spearman"
            )
        )

        print(
            f"{raw_feature:<25} "
            f"{pct_feature:<30} "
            f"Pearson={pearson_value:.6f} "
            f"Spearman={spearman_value:.6f}"
        )

    # ========================================================
    # DISTRIBUTION EXTREMES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("9. EXTREME DISTRIBUTION CHECK")
    print("=" * 70)

    distribution = pd.DataFrame(
        index=FEATURE_COLUMNS
    )

    distribution["p01"] = (
        X.quantile(0.01)
    )

    distribution["p05"] = (
        X.quantile(0.05)
    )

    distribution["p25"] = (
        X.quantile(0.25)
    )

    distribution["p50"] = (
        X.quantile(0.50)
    )

    distribution["p75"] = (
        X.quantile(0.75)
    )

    distribution["p95"] = (
        X.quantile(0.95)
    )

    distribution["p99"] = (
        X.quantile(0.99)
    )

    print(
        distribution.to_string(
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("10. FINAL AUDIT STATUS")
    print("=" * 70)

    infinite_values = np.isinf(
        X.to_numpy(
            dtype=float
        )
    ).sum()

    duplicate_keys = df.duplicated(
        ["ticker", "date"]
    ).sum()

    print(
        f"Feature columns        : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Infinite values        : "
        f"{infinite_values}"
    )

    print(
        f"Duplicate ticker/date  : "
        f"{duplicate_keys}"
    )

    print(
        f"High-correlation pairs : "
        f"{len(high_corr)}"
    )

    print(
        f"Near-zero variance     : "
        f"{len(near_zero)}"
    )

    if infinite_values != 0:
        raise RuntimeError(
            "Infinite values detected."
        )

    if duplicate_keys != 0:
        raise RuntimeError(
            "Duplicate ticker/date keys detected."
        )

    print("\nAUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()