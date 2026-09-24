from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/clean/etfs")

OUTPUT_FILE = Path(
    "data/targets/etf_2y_diagnostics.parquet"
)

HORIZON = relativedelta(years=2)

TRADING_DAYS_PER_YEAR = 252

EXPECTED_ETF_COUNT = 41


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
    """
    Load the final clean 41-ETF price universe.

    Dates are stored as the Parquet index.
    """

    files = sorted(INPUT_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No ETF parquet files found in {INPUT_DIR}"
        )

    frames = []

    for file in files:

        ticker = ticker_from_filename(file)

        # DYNAMIC was removed from the locked ETF universe.
        if ticker == "DYNAMIC.NS":
            continue

        df = pd.read_parquet(file)

        if "Adj Close" not in df.columns:
            raise ValueError(
                f"{file} does not contain 'Adj Close'. "
                f"Columns found: {list(df.columns)}"
            )

        # Date is stored as the index.
        dates = pd.to_datetime(
            df.index,
            errors="coerce"
        ).normalize()

        prices = pd.to_numeric(
            df["Adj Close"],
            errors="coerce"
        )

        frame = pd.DataFrame(
            {
                "Ticker": ticker,
                "Date": dates,
                "Adj Close": prices,
            }
        )

        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    data = data.dropna(
        subset=["Date", "Adj Close"]
    )

    # Prices must be positive.
    invalid_prices = (
        data["Adj Close"] <= 0
    ).sum()

    if invalid_prices:
        raise ValueError(
            f"Found {invalid_prices} non-positive "
            f"Adj Close values."
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    duplicate_count = data.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"ticker/date observations."
        )

    tickers = set(
        data["Ticker"].unique()
    )

    if len(tickers) != EXPECTED_ETF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ETF_COUNT} ETFs, "
            f"but found {len(tickers)}."
        )

    if "DYNAMIC.NS" in tickers:
        raise ValueError(
            "DYNAMIC.NS is present in the clean ETF universe."
        )

    return data


# ============================================================
# 2Y+ METRIC CALCULATION
# ============================================================

def calculate_metrics_for_observation(
    dates,
    prices,
    observation_index
):
    """
    Calculate all 2Y diagnostics for one observation.

    The future endpoint is the first available trading
    observation on or after observation_date + 2 years.

    Metrics are calculated only using information between
    the observation date and the 2Y endpoint.
    """

    observation_date = dates[observation_index]

    target_date = (
        observation_date + HORIZON
    )

    date_values = dates.to_numpy(
        dtype="datetime64[ns]"
    )

    target_value = target_date.to_datetime64()

    future_index = date_values.searchsorted(
        target_value,
        side="left"
    )

    if future_index >= len(date_values):
        return {
            "target_date_2y": target_date,
            "future_date_2y": pd.NaT,
            "future_adj_close_2y": np.nan,
            "annualized_return_2y": np.nan,
            "volatility_2y": np.nan,
            "max_drawdown_2y": np.nan,
            "risk_adjusted_return_2y": np.nan,
        }

    future_date = pd.Timestamp(
        date_values[future_index]
    )

    future_price = float(
        prices.iloc[future_index]
    )

    starting_price = float(
        prices.iloc[observation_index]
    )

    if starting_price <= 0 or future_price <= 0:
        raise ValueError(
            f"Invalid price for 2Y calculation on "
            f"{observation_date}."
        )

    # --------------------------------------------------------
    # 2Y annualized return
    # --------------------------------------------------------

    actual_days = (
        future_date - observation_date
    ).days

    years = actual_days / 365.25

    if years <= 0:
        raise ValueError(
            "Future period is not positive."
        )

    annualized_return = (
        (future_price / starting_price)
        ** (1.0 / years)
        - 1.0
    )

    # --------------------------------------------------------
    # Evaluation window
    # --------------------------------------------------------

    window_prices = prices.iloc[
        observation_index:
        future_index + 1
    ].astype(float)

    if len(window_prices) < 2:
        return {
            "target_date_2y": target_date,
            "future_date_2y": future_date,
            "future_adj_close_2y": future_price,
            "annualized_return_2y": annualized_return,
            "volatility_2y": np.nan,
            "max_drawdown_2y": np.nan,
            "risk_adjusted_return_2y": np.nan,
        }

    # --------------------------------------------------------
    # Daily returns
    # --------------------------------------------------------

    daily_returns = (
        window_prices
        .pct_change()
        .dropna()
    )

    if len(daily_returns) >= 2:
        volatility = (
            daily_returns.std(
                ddof=1
            )
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
    else:
        volatility = np.nan

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_peak = (
        window_prices
        .cummax()
    )

    drawdowns = (
        window_prices / running_peak
        - 1.0
    )

    max_drawdown = drawdowns.min()

    # --------------------------------------------------------
    # Risk-adjusted return
    # --------------------------------------------------------

    if (
        pd.notna(volatility)
        and volatility > 0
    ):
        risk_adjusted_return = (
            annualized_return
            / volatility
        )
    else:
        risk_adjusted_return = np.nan

    return {
        "target_date_2y": target_date,
        "future_date_2y": future_date,
        "future_adj_close_2y": future_price,
        "annualized_return_2y": annualized_return,
        "volatility_2y": volatility,
        "max_drawdown_2y": max_drawdown,
        "risk_adjusted_return_2y": risk_adjusted_return,
    }


# ============================================================
# GENERATE 2Y DIAGNOSTICS
# ============================================================

def generate_diagnostics(df):

    results = []

    for ticker, group in df.groupby(
        "Ticker",
        sort=False
    ):

        group = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

        dates = group["Date"]

        prices = group["Adj Close"]

        print(
            f"Processing {ticker} "
            f"({len(group):,} observations)..."
        )

        for i in range(len(group)):

            metrics = (
                calculate_metrics_for_observation(
                    dates,
                    prices,
                    i
                )
            )

            results.append(
                {
                    "Ticker": ticker,
                    "Date": dates.iloc[i],
                    "Adj Close": float(
                        prices.iloc[i]
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# AUDIT SUMMARY
# ============================================================

def print_summary(result):

    print("\n" + "=" * 70)
    print("2Y+ ETF DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"\nRows:               {len(result):,}"
    )

    print(
        f"Tickers:            "
        f"{result['Ticker'].nunique():,}"
    )

    print(
        f"Date range:         "
        f"{result['Date'].min().date()} → "
        f"{result['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    print("\n2Y COVERAGE")
    print("-" * 70)

    # IMPORTANT:
    # Count the boolean Series itself.
    # Do NOT create a DataFrame and call .sum(),
    # because the DataFrame contains datetime columns.
    valid_count = (
        result["future_date_2y"]
        .notna()
        .sum()
    )

    missing_count = (
        result["future_date_2y"]
        .isna()
        .sum()
    )

    print(
        f"Valid 2Y outcomes:  {valid_count:,}"
    )

    print(
        f"Missing 2Y outcomes: {missing_count:,}"
    )

    # --------------------------------------------------------
    # Annualized return
    # --------------------------------------------------------

    print(
        "\n2Y ANNUALIZED RETURN"
    )
    print("-" * 70)

    series = (
        result[
            "annualized_return_2y"
        ]
        .dropna()
    )

    if len(series):

        print(
            f"Mean:               "
            f"{series.mean():.6f}"
        )

        print(
            f"Std:                "
            f"{series.std():.6f}"
        )

        print(
            f"Min:                "
            f"{series.min():.6f}"
        )

        for label, q in [
            ("1%", .01),
            ("5%", .05),
            ("10%", .10),
            ("25%", .25),
            ("Median", .50),
            ("75%", .75),
            ("90%", .90),
            ("95%", .95),
            ("99%", .99),
        ]:
            print(
                f"{label:<20}"
                f"{series.quantile(q):.6f}"
            )

        print(
            f"Max:                "
            f"{series.max():.6f}"
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    print(
        "\n2Y ANNUALIZED VOLATILITY"
    )
    print("-" * 70)

    series = (
        result[
            "volatility_2y"
        ]
        .dropna()
    )

    if len(series):

        print(
            f"Mean:               "
            f"{series.mean():.6f}"
        )

        print(
            f"Median:             "
            f"{series.median():.6f}"
        )

        print(
            f"Min:                "
            f"{series.min():.6f}"
        )

        print(
            f"Max:                "
            f"{series.max():.6f}"
        )

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    print(
        "\n2Y MAXIMUM DRAWDOWN"
    )
    print("-" * 70)

    series = (
        result[
            "max_drawdown_2y"
        ]
        .dropna()
    )

    if len(series):

        print(
            f"Mean:               "
            f"{series.mean():.6f}"
        )

        print(
            f"Median:             "
            f"{series.median():.6f}"
        )

        print(
            f"Min:                "
            f"{series.min():.6f}"
        )

        print(
            f"Max:                "
            f"{series.max():.6f}"
        )

    # --------------------------------------------------------
    # Risk-adjusted return
    # --------------------------------------------------------

    print(
        "\n2Y RISK-ADJUSTED RETURN"
    )
    print("-" * 70)

    series = (
        result[
            "risk_adjusted_return_2y"
        ]
        .dropna()
    )

    if len(series):

        print(
            f"Mean:               "
            f"{series.mean():.6f}"
        )

        print(
            f"Median:             "
            f"{series.median():.6f}"
        )

        print(
            f"Min:                "
            f"{series.min():.6f}"
        )

        print(
            f"Max:                "
            f"{series.max():.6f}"
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_result(result):

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    duplicate_count = result.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    print(
        f"Duplicate ticker/date: "
        f"{duplicate_count}"
    )

    if duplicate_count:
        raise ValueError(
            "Duplicate ticker/date observations found."
        )

    invalid_future_dates = (
        result["future_date_2y"].notna()
        &
        (
            result["future_date_2y"]
            < result["target_date_2y"]
        )
    ).sum()

    print(
        f"2Y target-date violations: "
        f"{invalid_future_dates}"
    )

    if invalid_future_dates:
        raise ValueError(
            "2Y target-date violations found."
        )

    invalid_return = (
        result["annualized_return_2y"].notna()
        &
        (
            result["annualized_return_2y"]
            <= -1
        )
    ).sum()

    print(
        f"Invalid annualized returns: "
        f"{invalid_return}"
    )

    if invalid_return:
        raise ValueError(
            "Invalid annualized return found."
        )

    if (
        result["Ticker"].nunique()
        != EXPECTED_ETF_COUNT
    ):
        raise ValueError(
            "ETF universe size mismatch."
        )

    if "DYNAMIC.NS" in set(
        result["Ticker"].unique()
    ):
        raise ValueError(
            "DYNAMIC.NS found in results."
        )

    print("Validation:         PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATING ETF 2Y+ DIAGNOSTICS")
    print("=" * 70)

    print("\nLoading cleaned ETF prices...")

    df = load_clean_etf_data()

    print(
        f"Loaded {len(df):,} clean ETF rows."
    )

    print(
        f"Loaded {df['Ticker'].nunique()} ETFs."
    )

    print(
        "\nCalculating 2Y diagnostics..."
    )

    result = generate_diagnostics(df)

    # --------------------------------------------------------
    # Datetime normalization
    # --------------------------------------------------------

    datetime_columns = [
        "Date",
        "target_date_2y",
        "future_date_2y",
    ]

    for column in datetime_columns:

        result[column] = (
            pd.to_datetime(
                result[column],
                errors="coerce"
            )
            .astype("datetime64[ns]")
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_result(result)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(result)

    print(
        "\nSaved 2Y diagnostics to:"
    )

    print(OUTPUT_FILE)

    print(
        "\nRAW ETF DATA WAS NOT MODIFIED."
    )

    print(
        "CLEAN ETF PRICE DATA WAS NOT MODIFIED."
    )

    print(
        "NO LABELS WERE CREATED."
    )


if __name__ == "__main__":
    main()