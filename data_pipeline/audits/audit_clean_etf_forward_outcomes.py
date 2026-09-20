from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RAW_TARGET_FILE = Path(
    "data/targets/etf_forward_outcomes.parquet"
)

CLEAN_TARGET_FILE = Path(
    "data/targets/etf_forward_outcomes_clean.parquet"
)

AUDIT_OUTPUT_FILE = Path(
    "data_pipeline/audits/clean_etf_target_extremes.csv"
)

EXTREME_ROWS = 20


# ============================================================
# LOAD RAW TARGET
# ============================================================

def load_raw_target():

    if not RAW_TARGET_FILE.exists():
        raise FileNotFoundError(
            f"RAW target file not found:\n{RAW_TARGET_FILE}"
        )

    df = pd.read_parquet(RAW_TARGET_FILE)

    required = {
        "ticker",
        "observation_date",
        "observation_price",
        "future_date_6m",
        "future_price_6m",
        "forward_return_6m",
        "future_date_1y",
        "future_price_1y",
        "forward_return_1y",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "RAW target file is missing columns: "
            f"{sorted(missing)}"
        )

    # Normalize into the same schema used by the clean target.
    raw = pd.DataFrame()

    raw["Ticker"] = df["ticker"]
    raw["Date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce"
    )

    raw["Adj Close"] = pd.to_numeric(
        df["observation_price"],
        errors="coerce"
    )

    raw["future_date_6m"] = pd.to_datetime(
        df["future_date_6m"],
        errors="coerce"
    )

    raw["future_adj_close_6m"] = pd.to_numeric(
        df["future_price_6m"],
        errors="coerce"
    )

    raw["forward_return_6m"] = pd.to_numeric(
        df["forward_return_6m"],
        errors="coerce"
    )

    raw["future_date_1y"] = pd.to_datetime(
        df["future_date_1y"],
        errors="coerce"
    )

    raw["future_adj_close_1y"] = pd.to_numeric(
        df["future_price_1y"],
        errors="coerce"
    )

    raw["forward_return_1y"] = pd.to_numeric(
        df["forward_return_1y"],
        errors="coerce"
    )

    return raw


# ============================================================
# LOAD CLEAN TARGET
# ============================================================

def load_clean_target():

    if not CLEAN_TARGET_FILE.exists():
        raise FileNotFoundError(
            f"CLEAN target file not found:\n{CLEAN_TARGET_FILE}"
        )

    df = pd.read_parquet(CLEAN_TARGET_FILE)

    required = {
        "Ticker",
        "Date",
        "Adj Close",
        "future_date_6m",
        "future_adj_close_6m",
        "forward_return_6m",
        "future_date_1y",
        "future_adj_close_1y",
        "forward_return_1y",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "CLEAN target file is missing columns: "
            f"{sorted(missing)}"
        )

    clean = df.copy()

    clean["Date"] = pd.to_datetime(
        clean["Date"],
        errors="coerce"
    )

    clean["future_date_6m"] = pd.to_datetime(
        clean["future_date_6m"],
        errors="coerce"
    )

    clean["future_date_1y"] = pd.to_datetime(
        clean["future_date_1y"],
        errors="coerce"
    )

    return clean


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(
    df,
    return_column,
    title
):

    values = df[return_column].dropna()

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Valid observations: {len(values):,}")
    print(
        f"Missing:           "
        f"{df[return_column].isna().sum():,}"
    )

    if values.empty:
        return

    q = values.quantile(
        [
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

    print(f"Mean:              {values.mean():.6f}")
    print(f"Std:               {values.std():.6f}")
    print(f"Min:               {values.min():.6f}")
    print(f"1%:                {q.loc[0.01]:.6f}")
    print(f"5%:                {q.loc[0.05]:.6f}")
    print(f"10%:               {q.loc[0.10]:.6f}")
    print(f"25%:               {q.loc[0.25]:.6f}")
    print(f"Median:            {q.loc[0.50]:.6f}")
    print(f"75%:               {q.loc[0.75]:.6f}")
    print(f"90%:               {q.loc[0.90]:.6f}")
    print(f"95%:               {q.loc[0.95]:.6f}")
    print(f"99%:               {q.loc[0.99]:.6f}")
    print(f"Max:               {values.max():.6f}")

    print("\nExtreme-return counts:")

    checks = [
        ("> +100%", values > 1.0),
        ("> +200%", values > 2.0),
        ("> +500%", values > 5.0),
        ("> +1000%", values > 10.0),
        ("< -25%", values < -0.25),
        ("< -50%", values < -0.50),
        ("< -75%", values < -0.75),
        ("< -90%", values < -0.90),
    ]

    for label, condition in checks:
        print(f"{label:<12} {condition.sum():,}")


# ============================================================
# EXTREMES
# ============================================================

def print_extremes(
    df,
    return_column,
    horizon
):

    future_date_column = (
        "future_date_6m"
        if horizon == "6M"
        else "future_date_1y"
    )

    future_price_column = (
        "future_adj_close_6m"
        if horizon == "6M"
        else "future_adj_close_1y"
    )

    print("\n" + "=" * 70)
    print(f"TOP {EXTREME_ROWS} {horizon} RETURNS")
    print("=" * 70)

    columns = [
        "Ticker",
        "Date",
        "Adj Close",
        future_date_column,
        future_price_column,
        return_column,
    ]

    top = (
        df.dropna(subset=[return_column])
        .nlargest(
            EXTREME_ROWS,
            return_column
        )[columns]
    )

    print(top.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"BOTTOM {EXTREME_ROWS} {horizon} RETURNS")
    print("=" * 70)

    bottom = (
        df.dropna(subset=[return_column])
        .nsmallest(
            EXTREME_ROWS,
            return_column
        )[columns]
    )

    print(bottom.to_string(index=False))


# ============================================================
# RAW VS CLEAN COMPARISON
# ============================================================

def compare_distributions(
    raw,
    clean,
    return_column,
    horizon
):

    raw_values = raw[return_column].dropna()
    clean_values = clean[return_column].dropna()

    metrics = [
        "Valid",
        "Mean",
        "Std",
        "Min",
        "1%",
        "5%",
        "10%",
        "25%",
        "Median",
        "75%",
        "90%",
        "95%",
        "99%",
        "Max",
    ]

    def calculate(values):

        return [
            len(values),
            values.mean(),
            values.std(),
            values.min(),
            values.quantile(0.01),
            values.quantile(0.05),
            values.quantile(0.10),
            values.quantile(0.25),
            values.quantile(0.50),
            values.quantile(0.75),
            values.quantile(0.90),
            values.quantile(0.95),
            values.quantile(0.99),
            values.max(),
        ]

    comparison = pd.DataFrame(
        {
            "Metric": metrics,
            "Raw": calculate(raw_values),
            "Clean": calculate(clean_values),
        }
    )

    print("\n" + "=" * 70)
    print(f"RAW vs CLEAN — {horizon}")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# EXTREME AUDIT FILE
# ============================================================

def build_extreme_audit(clean):

    frames = []

    for horizon, return_column in [
        ("6M", "forward_return_6m"),
        ("1Y", "forward_return_1y"),
    ]:

        subset = clean.dropna(
            subset=[return_column]
        ).copy()

        extreme = subset[
            (subset[return_column] > 1.0)
            | (subset[return_column] < -0.50)
        ].copy()

        if extreme.empty:
            continue

        extreme["Horizon"] = horizon

        extreme = extreme[
            [
                "Horizon",
                "Ticker",
                "Date",
                "Adj Close",
                return_column,
            ]
        ].rename(
            columns={
                return_column: "Forward_Return"
            }
        )

        frames.append(extreme)

    if not frames:

        return pd.DataFrame(
            columns=[
                "Horizon",
                "Ticker",
                "Date",
                "Adj Close",
                "Forward_Return",
            ]
        )

    result = pd.concat(
        frames,
        ignore_index=True
    )

    return result.sort_values(
        ["Horizon", "Forward_Return"],
        ascending=[True, False]
    )


# ============================================================
# TICKER EXTREME COUNTS
# ============================================================

def print_ticker_extremes(
    clean,
    return_column,
    horizon
):

    print("\n" + "=" * 70)
    print(f"CLEAN ETF EXTREME COUNTS BY TICKER — {horizon}")
    print("=" * 70)

    summary = (
        clean
        .dropna(subset=[return_column])
        .groupby("Ticker")[return_column]
        .agg(
            observations="count",
            maximum="max",
            minimum="min",
            over_100_pct=lambda x: (x > 1.0).sum(),
            over_200_pct=lambda x: (x > 2.0).sum(),
            over_500_pct=lambda x: (x > 5.0).sum(),
            under_minus_50_pct=lambda x: (x < -0.50).sum(),
        )
        .sort_values(
            "maximum",
            ascending=False
        )
    )

    print(
        summary.head(15).to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN ETF FORWARD OUTCOME AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading raw ETF targets...")

    raw = load_raw_target()

    print(
        f"Raw rows:      {len(raw):,}"
    )

    print(
        f"Raw tickers:   {raw['Ticker'].nunique():,}"
    )

    print("\nLoading clean ETF targets...")

    clean = load_clean_target()

    print(
        f"Clean rows:    {len(clean):,}"
    )

    print(
        f"Clean tickers: {clean['Ticker'].nunique():,}"
    )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRUCTURAL COMPARISON")
    print("=" * 70)

    print(
        f"Raw row count:       {len(raw):,}"
    )

    print(
        f"Clean row count:     {len(clean):,}"
    )

    print(
        f"Raw ticker count:    {raw['Ticker'].nunique()}"
    )

    print(
        f"Clean ticker count:  {clean['Ticker'].nunique()}"
    )

    print(
        f"Raw date range:      "
        f"{raw['Date'].min().date()} → "
        f"{raw['Date'].max().date()}"
    )

    print(
        f"Clean date range:    "
        f"{clean['Date'].min().date()} → "
        f"{clean['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Clean distributions
    # --------------------------------------------------------

    print_distribution(
        clean,
        "forward_return_6m",
        "CLEAN 6-MONTH FORWARD RETURNS"
    )

    print_distribution(
        clean,
        "forward_return_1y",
        "CLEAN 1-YEAR FORWARD RETURNS"
    )

    # --------------------------------------------------------
    # Extremes
    # --------------------------------------------------------

    print_extremes(
        clean,
        "forward_return_6m",
        "6M"
    )

    print_extremes(
        clean,
        "forward_return_1y",
        "1Y"
    )

    # --------------------------------------------------------
    # Raw vs clean
    # --------------------------------------------------------

    compare_distributions(
        raw,
        clean,
        "forward_return_6m",
        "6-MONTH"
    )

    compare_distributions(
        raw,
        clean,
        "forward_return_1y",
        "1-YEAR"
    )

    # --------------------------------------------------------
    # Per ETF
    # --------------------------------------------------------

    print_ticker_extremes(
        clean,
        "forward_return_6m",
        "6M"
    )

    print_ticker_extremes(
        clean,
        "forward_return_1y",
        "1Y"
    )

    # --------------------------------------------------------
    # Audit CSV
    # --------------------------------------------------------

    extreme_audit = build_extreme_audit(clean)

    AUDIT_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extreme_audit.to_csv(
        AUDIT_OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("AUDIT FILE")
    print("=" * 70)

    print(
        f"Extreme observations saved to:\n"
        f"{AUDIT_OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "No returns were deleted, clipped, capped, or modified."
    )

    print(
        "PostgreSQL was not modified."
    )

    print(
        "The original raw ETF target file was not modified."
    )

    print(
        "This audit is for investigation before freezing "
        "the clean ETF target dataset."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()