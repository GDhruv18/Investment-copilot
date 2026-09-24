from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TARGET_FILE = Path(
    "data/targets/etf_forward_outcomes_clean.parquet"
)

EXPECTED_ETF_COUNT = 41


# ============================================================
# HELPERS
# ============================================================

def print_distribution(series, name):
    series = series.dropna()

    print(f"\n{name}")
    print("-" * 70)

    print(f"Count:              {len(series):,}")
    print(f"Mean:               {series.mean():.6f}")
    print(f"Std:                {series.std():.6f}")
    print(f"Min:                {series.min():.6f}")

    percentiles = {
        "1%": 0.01,
        "5%": 0.05,
        "10%": 0.10,
        "25%": 0.25,
        "Median": 0.50,
        "75%": 0.75,
        "90%": 0.90,
        "95%": 0.95,
        "99%": 0.99,
    }

    for label, q in percentiles.items():
        print(
            f"{label:<20}"
            f"{series.quantile(q):.6f}"
        )

    print(f"Max:                {series.max():.6f}")

    print("\nExtreme counts:")
    print(
        f"> +100%:            {(series > 1.0).sum():,}"
    )
    print(
        f"> +200%:            {(series > 2.0).sum():,}"
    )
    print(
        f"> +500%:            {(series > 5.0).sum():,}"
    )
    print(
        f"> +1000%:           {(series > 10.0).sum():,}"
    )

    print(
        f"< -25%:             {(series < -0.25).sum():,}"
    )
    print(
        f"< -50%:             {(series < -0.50).sum():,}"
    )
    print(
        f"< -75%:             {(series < -0.75).sum():,}"
    )
    print(
        f"< -90%:             {(series < -0.90).sum():,}"
    )


def print_extremes(df, column, name, ascending=False, n=15):
    print(f"\n{name}")
    print("-" * 70)

    cols = [
        "Ticker",
        "Date",
        column,
    ]

    available_cols = [
        col for col in cols
        if col in df.columns
    ]

    output = (
        df.dropna(subset=[column])
        .sort_values(
            column,
            ascending=ascending
        )
        .head(n)
    )

    print(
        output[available_cols].to_string(
            index=False
        )
    )


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("FINAL CLEAN ETF FORWARD OUTCOME AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"Target file not found:\n{TARGET_FILE}"
        )

    df = pd.read_parquet(TARGET_FILE)

    print("\nDATASET")
    print("-" * 70)

    print(f"File:               {TARGET_FILE}")
    print(f"Rows:               {len(df):,}")
    print(
        f"Tickers:            "
        f"{df['Ticker'].nunique():,}"
    )

    print(
        f"Date range:         "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Universe validation
    # --------------------------------------------------------

    tickers = set(df["Ticker"].unique())

    print("\nUNIVERSE VALIDATION")
    print("-" * 70)

    print(
        f"Expected ETFs:      {EXPECTED_ETF_COUNT}"
    )
    print(
        f"Actual ETFs:        {len(tickers)}"
    )

    if len(tickers) != EXPECTED_ETF_COUNT:
        raise ValueError(
            "ETF universe size mismatch."
        )

    if "DYNAMIC.NS" in tickers:
        raise ValueError(
            "DYNAMIC.NS is present in the final ETF target dataset."
        )

    print("DYNAMIC.NS present: False")
    print("Universe check:     PASS")

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    required_columns = [
        "Ticker",
        "Date",
        "Adj Close",
        "target_date_6m",
        "future_date_6m",
        "future_adj_close_6m",
        "forward_return_6m",
        "target_date_1y",
        "future_date_1y",
        "future_adj_close_1y",
        "forward_return_1y",
    ]

    print("\nSCHEMA VALIDATION")
    print("-" * 70)

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns:   PASS")

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    print(
        f"Duplicate rows:     {duplicate_count}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate ticker/date observations found."
        )

    # --------------------------------------------------------
    # Target-date validation
    # --------------------------------------------------------

    invalid_6m = (
        df["future_date_6m"].notna()
        &
        (
            df["future_date_6m"]
            < df["target_date_6m"]
        )
    ).sum()

    invalid_1y = (
        df["future_date_1y"].notna()
        &
        (
            df["future_date_1y"]
            < df["target_date_1y"]
        )
    ).sum()

    print("\nTARGET-DATE VALIDATION")
    print("-" * 70)

    print(
        f"6M violations:      {invalid_6m}"
    )

    print(
        f"1Y violations:      {invalid_1y}"
    )

    if invalid_6m != 0 or invalid_1y != 0:
        raise ValueError(
            "Forward target-date violations found."
        )

    print("Target-date checks:  PASS")

    # --------------------------------------------------------
    # Return sanity
    # --------------------------------------------------------

    invalid_6m_returns = (
        df["forward_return_6m"].notna()
        &
        (
            df["forward_return_6m"] <= -1
        )
    ).sum()

    invalid_1y_returns = (
        df["forward_return_1y"].notna()
        &
        (
            df["forward_return_1y"] <= -1
        )
    ).sum()

    print("\nRETURN SANITY")
    print("-" * 70)

    print(
        f"Invalid 6M returns: {invalid_6m_returns}"
    )

    print(
        f"Invalid 1Y returns: {invalid_1y_returns}"
    )

    if (
        invalid_6m_returns != 0
        or invalid_1y_returns != 0
    ):
        raise ValueError(
            "Found forward returns <= -100%."
        )

    print("Return sanity:      PASS")

    # --------------------------------------------------------
    # Distribution audit
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETURN DISTRIBUTIONS")
    print("=" * 70)

    print_distribution(
        df["forward_return_6m"],
        "6-MONTH FORWARD RETURNS"
    )

    print_distribution(
        df["forward_return_1y"],
        "1-YEAR FORWARD RETURNS"
    )

    # --------------------------------------------------------
    # Extreme observations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXTREME OBSERVATIONS")
    print("=" * 70)

    print_extremes(
        df,
        "forward_return_6m",
        "TOP 15 6-MONTH RETURNS",
        ascending=False
    )

    print_extremes(
        df,
        "forward_return_6m",
        "BOTTOM 15 6-MONTH RETURNS",
        ascending=True
    )

    print_extremes(
        df,
        "forward_return_1y",
        "TOP 15 1-YEAR RETURNS",
        ascending=False
    )

    print_extremes(
        df,
        "forward_return_1y",
        "BOTTOM 15 1-YEAR RETURNS",
        ascending=True
    )

    # --------------------------------------------------------
    # Per-ETF coverage
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-ETF TARGET COVERAGE")
    print("=" * 70)

    coverage = (
        df.groupby("Ticker")
        .agg(
            observations=("Date", "count"),
            valid_6m=(
                "forward_return_6m",
                "count"
            ),
            valid_1y=(
                "forward_return_1y",
                "count"
            ),
        )
        .reset_index()
    )

    coverage["6m_coverage"] = (
        coverage["valid_6m"]
        / coverage["observations"]
    )

    coverage["1y_coverage"] = (
        coverage["valid_1y"]
        / coverage["observations"]
    )

    print(
        coverage.to_string(
            index=False,
            formatters={
                "6m_coverage": "{:.2%}".format,
                "1y_coverage": "{:.2%}".format,
            }
        )
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        "Final 41-ETF target dataset passed "
        "structural and return sanity checks."
    )

    print(
        "\nNO LABELS WERE CREATED."
    )

    print(
        "NO DATA WAS MODIFIED."
    )


if __name__ == "__main__":
    main()