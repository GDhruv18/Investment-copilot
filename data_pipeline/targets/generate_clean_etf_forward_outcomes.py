from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/clean/etfs")
OUTPUT_FILE = Path("data/targets/etf_forward_outcomes_clean.parquet")

HORIZONS = {
    "6m": relativedelta(months=6),
    "1y": relativedelta(years=1),
}


# ============================================================
# HELPERS
# ============================================================

def ticker_from_filename(path: Path) -> str:
    """
    Convert:
        BANKBEES_NS.parquet
    into:
        BANKBEES.NS
    """
    stem = path.stem

    if stem.endswith("_NS"):
        return stem[:-3] + ".NS"

    return stem


def load_clean_etf_data():
    """Load all cleaned ETF parquet files.

    Clean ETF Parquet files store trading dates in the DataFrame index,
    while price fields such as ``Adj Close`` are columns.
    """
    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ETF parquet files found in {INPUT_DIR}")

    frames = []
    for file in files:
        ticker = ticker_from_filename(file)
        if ticker == "DYNAMIC.NS":
            continue

        df = pd.read_parquet(file)
        if "Adj Close" not in df.columns:
            raise ValueError(
                f"{file} does not contain 'Adj Close'. "
                f"Columns found: {list(df.columns)}"
            )

        dates = pd.to_datetime(df.index, errors="coerce").normalize()
        frame = pd.DataFrame({
            "Date": dates,
            "Adj Close": pd.to_numeric(df["Adj Close"], errors="coerce"),
        })
        frame["Ticker"] = ticker
        frames.append(frame)

    if not frames:
        raise ValueError("No clean ETF files were loaded.")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["Date", "Adj Close"])

    invalid_price_count = (data["Adj Close"] <= 0).sum()
    if invalid_price_count:
        raise ValueError(f"Found {invalid_price_count} non-positive Adj Close values.")

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    duplicate_count = data.duplicated(subset=["Ticker", "Date"]).sum()
    if duplicate_count:
        print(f"Warning: removing {duplicate_count} duplicate ticker/date observations.")
        data = data.drop_duplicates(
            subset=["Ticker", "Date"], keep="last"
        ).reset_index(drop=True)

    actual_tickers = set(data["Ticker"].unique())
    if len(actual_tickers) != 41:
        raise ValueError(
            f"Expected 41 clean ETF tickers, but loaded {len(actual_tickers)}."
        )

    if "DYNAMIC.NS" in actual_tickers:
        raise ValueError("DYNAMIC.NS must not be present in the clean ETF target dataset.")

    return data


# ============================================================
# FORWARD OUTCOME CALCULATION
# ============================================================

def calculate_forward_outcomes(df):
    """
    For each observation date t:

        6M target = first available trading observation
                    on or after t + 6 calendar months

        1Y target = first available trading observation
                   on or after t + 1 calendar year

    Forward return:

        future_adj_close / current_adj_close - 1
    """

    all_results = []

    for ticker, group in df.groupby("Ticker", sort=False):

        group = group.sort_values("Date").reset_index(drop=True)

        dates = group["Date"].to_numpy()
        prices = group["Adj Close"].to_numpy()

        result = group.copy()

        # ----------------------------------------------------
        # Calendar target dates
        # ----------------------------------------------------

        result["target_date_6m"] = result["Date"].apply(
            lambda x: x + HORIZONS["6m"]
        )

        result["target_date_1y"] = result["Date"].apply(
            lambda x: x + HORIZONS["1y"]
        )

        # ----------------------------------------------------
        # Find first available trading date >= target date
        # ----------------------------------------------------

        date_series = pd.Series(
            dates,
            index=range(len(dates))
        )

        # searchsorted requires sorted dates
        date_values = date_series.to_numpy(
            dtype="datetime64[ns]"
        )

        def find_future_index(target_date):
            target = target_date.to_datetime64()

            idx = date_values.searchsorted(
                target,
                side="left"
            )

            if idx >= len(date_values):
                return None

            return idx

        future_dates_6m = []
        future_prices_6m = []

        future_dates_1y = []
        future_prices_1y = []

        for target_date in result["target_date_6m"]:
            idx = find_future_index(target_date)

            if idx is None:
                future_dates_6m.append(pd.NaT)
                future_prices_6m.append(float("nan"))
            else:
                future_dates_6m.append(
                    pd.Timestamp(date_values[idx])
                )
                future_prices_6m.append(
                    float(prices[idx])
                )

        for target_date in result["target_date_1y"]:
            idx = find_future_index(target_date)

            if idx is None:
                future_dates_1y.append(pd.NaT)
                future_prices_1y.append(float("nan"))
            else:
                future_dates_1y.append(
                    pd.Timestamp(date_values[idx])
                )
                future_prices_1y.append(
                    float(prices[idx])
                )

        result["future_date_6m"] = future_dates_6m
        result["future_adj_close_6m"] = future_prices_6m

        result["future_date_1y"] = future_dates_1y
        result["future_adj_close_1y"] = future_prices_1y

        # ----------------------------------------------------
        # Forward returns
        # ----------------------------------------------------

        result["forward_return_6m"] = (
            result["future_adj_close_6m"]
            / result["Adj Close"]
            - 1.0
        )

        result["forward_return_1y"] = (
            result["future_adj_close_1y"]
            / result["Adj Close"]
            - 1.0
        )

        all_results.append(result)

    return pd.concat(
        all_results,
        ignore_index=True
    )


# ============================================================
# VALIDATION / SUMMARY
# ============================================================

def print_summary(result):

    print("\n" + "=" * 70)
    print("CLEAN ETF FORWARD OUTCOME GENERATION")
    print("=" * 70)

    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file:     {OUTPUT_FILE}")

    print("\nDATASET")
    print("-" * 70)

    print(f"Rows:             {len(result):,}")
    print(
        f"Tickers:          {result['Ticker'].nunique():,}"
    )
    print(
        f"Date range:       "
        f"{result['Date'].min().date()} → "
        f"{result['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # 6M
    # --------------------------------------------------------

    valid_6m = result["forward_return_6m"].dropna()

    print("\n6-MONTH FORWARD RETURN")
    print("-" * 70)

    print(f"Valid:             {len(valid_6m):,}")
    print(
        f"Missing:           "
        f"{result['forward_return_6m'].isna().sum():,}"
    )

    if len(valid_6m) > 0:
        print(
            f"Min:               {valid_6m.min():.6f}"
        )
        print(
            f"Max:               {valid_6m.max():.6f}"
        )
        print(
            f"Median:            {valid_6m.median():.6f}"
        )
        print(
            f"Mean:              {valid_6m.mean():.6f}"
        )

    # --------------------------------------------------------
    # 1Y
    # --------------------------------------------------------

    valid_1y = result["forward_return_1y"].dropna()

    print("\n1-YEAR FORWARD RETURN")
    print("-" * 70)

    print(f"Valid:             {len(valid_1y):,}")
    print(
        f"Missing:           "
        f"{result['forward_return_1y'].isna().sum():,}"
    )

    if len(valid_1y) > 0:
        print(
            f"Min:               {valid_1y.min():.6f}"
        )
        print(
            f"Max:               {valid_1y.max():.6f}"
        )
        print(
            f"Median:            {valid_1y.median():.6f}"
        )
        print(
            f"Mean:              {valid_1y.mean():.6f}"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print("\nSANITY CHECKS")
    print("-" * 70)

    duplicate_count = result.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    print(
        f"Duplicate ticker/date rows: {duplicate_count}"
    )

    invalid_6m = (
        result["future_date_6m"].notna()
        &
        (
            result["future_date_6m"]
            < result["target_date_6m"]
        )
    ).sum()

    invalid_1y = (
        result["future_date_1y"].notna()
        &
        (
            result["future_date_1y"]
            < result["target_date_1y"]
        )
    ).sum()

    print(
        f"6M target-date violations:   {invalid_6m}"
    )

    print(
        f"1Y target-date violations:   {invalid_1y}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATING CLEAN ETF FORWARD OUTCOMES — 41 ETF UNIVERSE")
    print("=" * 70)

    print("\nLoading cleaned ETF prices...")

    df = load_clean_etf_data()

    print(
        f"Loaded {len(df):,} clean ETF price rows."
    )

    print(
        f"Loaded {df['Ticker'].nunique()} ETF tickers."
    )

    print(
        f"DYNAMIC.NS present: {'DYNAMIC.NS' in set(df['Ticker'].unique())}"
    )

    print("\nCalculating 6M and 1Y forward outcomes...")

    result = calculate_forward_outcomes(df)

    # Ensure datetime columns use consistent precision
    datetime_columns = [
        "Date",
        "target_date_6m",
        "target_date_1y",
        "future_date_6m",
        "future_date_1y",
    ]

    for column in datetime_columns:
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce"
        ).astype("datetime64[ns]")

    # Final column ordering
    result = result[
        [
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
    ]

    # Sort
    result = result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save
    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print_summary(result)

    print(
        f"\nSaved clean ETF forward outcomes to:"
        f"\n{OUTPUT_FILE}"
    )

    print("\nRAW DATA WAS NOT MODIFIED.")
    print("POSTGRESQL WAS NOT MODIFIED.")
    print("OLD ETF TARGET FILE WAS NOT MODIFIED.")


if __name__ == "__main__":
    main()