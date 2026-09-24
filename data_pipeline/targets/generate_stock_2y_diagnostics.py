"""
Generate Stock 2Y Diagnostics

Methodology:
- Fixed 2-calendar-year forward horizon
- Future endpoint = first available trading observation on or after t + 2 calendar years
- Annualized return uses actual elapsed years
- Annualized volatility = daily Adj Close return std * sqrt(252)
- Maximum drawdown = minimum price/running peak - 1
- Raw risk-adjusted return = annualized return / annualized volatility
  (diagnostic only; NOT used directly as a composite score)

IMPORTANT:
- Date normalization preserves the original calendar date.
- Timezones are stripped WITHOUT converting through UTC.
- Raw stock files are never modified.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_STOCK_DIR = BASE_DIR / "data" / "raw" / "stocks"
OUTPUT_FILE = BASE_DIR / "data" / "targets" / "stock_2y_diagnostics.parquet"


# ============================================================
# EXPECTED DATA RANGE
# ============================================================

EXPECTED_MIN_DATE = pd.Timestamp("2014-01-01")
EXPECTED_MAX_DATE = pd.Timestamp("2026-09-18")

HORIZON_YEARS = 2
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# DATE NORMALIZATION
# ============================================================

def strip_timezone_preserve_calendar(value):
    """
    Convert a date/time value to a timezone-naive Timestamp
    WITHOUT changing its calendar date.

    Example:
        2014-01-01 00:00+05:30
            -> 2014-01-01 00:00

    NOT:
        2014-01-01 00:00+05:30
            -> 2013-12-31 18:30
    """

    if pd.isna(value):
        return pd.NaT

    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


# ============================================================
# TICKER FROM FILE NAME
# ============================================================

def ticker_from_filename(path):
    """
    Convert filenames such as:

        RELIANCE_NS.parquet
        HDFCBANK_NS.parquet

    into:

        RELIANCE.NS
        HDFCBANK.NS
    """

    return path.stem.replace("_NS", ".NS")


# ============================================================
# LOAD ONE STOCK FILE
# ============================================================

def load_stock_file(path):
    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Empty stock file: {path.name}")

    # --------------------------------------------------------
    # Case 1: Date exists as a normal column
    # --------------------------------------------------------

    date_column = None

    for candidate in ["Date", "date", "Datetime", "datetime"]:
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is not None:
        dates = df[date_column]
        working = df.copy()

    # --------------------------------------------------------
    # Case 2: Date is the index
    # --------------------------------------------------------

    else:
        working = df.copy()
        dates = working.index

    # --------------------------------------------------------
    # Find Adj Close
    # --------------------------------------------------------

    adj_close_column = None

    for candidate in [
        "Adj Close",
        "Adj_Close",
        "adj_close",
        "adjclose",
    ]:
        if candidate in working.columns:
            adj_close_column = candidate
            break

    if adj_close_column is None:

        # Handle possible MultiIndex columns
        if isinstance(working.columns, pd.MultiIndex):

            for col in working.columns:
                pieces = [str(x) for x in col]

                if any(
                    x.lower().replace(" ", "_") == "adj_close"
                    for x in pieces
                ):
                    adj_close_column = col
                    break

        if adj_close_column is None:
            raise ValueError(
                f"Could not find Adj Close in {path.name}. "
                f"Columns: {list(working.columns)}"
            )

    # --------------------------------------------------------
    # Build normalized dataframe
    # --------------------------------------------------------

    normalized = pd.DataFrame(
        {
            "date": dates,
            "adj_close": working[adj_close_column].values,
        }
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Strip timezone while preserving calendar date.
    #
    # DO NOT use:
    # pd.to_datetime(..., utc=True)
    # --------------------------------------------------------

    normalized["date"] = normalized["date"].map(
        strip_timezone_preserve_calendar
    )

    normalized["adj_close"] = pd.to_numeric(
        normalized["adj_close"],
        errors="coerce",
    )

    normalized["ticker"] = ticker_from_filename(path)

    normalized = normalized.dropna(
        subset=["date", "adj_close"]
    )

    normalized = normalized[
        normalized["adj_close"] > 0
    ]

    normalized = normalized[
        ["ticker", "date", "adj_close"]
    ]

    return normalized


# ============================================================
# LOAD ALL STOCKS
# ============================================================

def load_all_stocks():

    files = sorted(RAW_STOCK_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {RAW_STOCK_DIR}"
        )

    frames = []

    print("=" * 70)
    print("LOADING STOCK RAW DATA")
    print("=" * 70)

    print(f"Files found: {len(files)}")

    for i, path in enumerate(files, start=1):

        df = load_stock_file(path)
        frames.append(df)

        print(
            f"[{i:3d}/{len(files)}] "
            f"{path.name:<35} "
            f"Rows: {len(df):>6}"
        )

    prices = pd.concat(
        frames,
        ignore_index=True
    )

    prices = prices.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    duplicate_count = prices.duplicated(
        ["ticker", "date"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate ticker/date rows found: "
            f"{duplicate_count}"
        )

    if prices["date"].isna().any():
        raise ValueError("NaT dates found.")

    if prices["adj_close"].isna().any():
        raise ValueError("NaN Adj Close values found.")

    if (prices["adj_close"] <= 0).any():
        raise ValueError("Non-positive Adj Close values found.")

    # --------------------------------------------------------
    # CRITICAL DATE-RANGE CHECK
    # --------------------------------------------------------

    actual_min = prices["date"].min()
    actual_max = prices["date"].max()

    print()
    print("DATE RANGE VALIDATION")
    print("-" * 70)
    print(f"Actual minimum date: {actual_min}")
    print(f"Expected minimum:    {EXPECTED_MIN_DATE}")
    print(f"Actual maximum date: {actual_max}")
    print(f"Expected maximum:    {EXPECTED_MAX_DATE}")

    if actual_min != EXPECTED_MIN_DATE:
        raise ValueError(
            "CRITICAL DATE ERROR: "
            f"Expected minimum date {EXPECTED_MIN_DATE.date()}, "
            f"but loaded data starts at {actual_min.date()}."
        )

    if actual_max != EXPECTED_MAX_DATE:
        raise ValueError(
            "CRITICAL DATE ERROR: "
            f"Expected maximum date {EXPECTED_MAX_DATE.date()}, "
            f"but loaded data ends at {actual_max.date()}."
        )

    print("Date range validation: PASS")

    return prices


# ============================================================
# CALCULATE 2Y DIAGNOSTICS FOR ONE STOCK
# ============================================================

def calculate_stock_2y_metrics(group):

    group = group.sort_values("date").reset_index(drop=True)

    dates = group["date"].values
    prices = group["adj_close"].astype(float).values

    n = len(group)

    # --------------------------------------------------------
    # Daily returns
    # --------------------------------------------------------

    daily_returns = pd.Series(prices).pct_change()

    # --------------------------------------------------------
    # Running peak and drawdown
    # --------------------------------------------------------

    running_peak = pd.Series(prices).cummax()

    drawdowns = (
        pd.Series(prices) / running_peak
    ) - 1.0

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    future_date = [pd.NaT] * n
    future_adj_close = [np.nan] * n
    actual_years = [np.nan] * n

    annualized_return = [np.nan] * n
    annualized_volatility = [np.nan] * n
    max_drawdown = [np.nan] * n
    risk_adjusted_return = [np.nan] * n

    # --------------------------------------------------------
    # For every observation, find first trading day
    # on or after t + 2 calendar years.
    # --------------------------------------------------------

    date_series = pd.Series(group["date"])

    for i in range(n):

        current_date = pd.Timestamp(dates[i])

        target_date = current_date + pd.DateOffset(
            years=HORIZON_YEARS
        )

        # First available observation >= target date
        future_position = date_series.searchsorted(
            target_date,
            side="left"
        )

        if future_position >= n:
            continue

        future_dt = pd.Timestamp(
            dates[future_position]
        )

        current_price = prices[i]
        future_price = prices[future_position]

        if (
            not np.isfinite(current_price)
            or current_price <= 0
            or not np.isfinite(future_price)
            or future_price <= 0
        ):
            continue

        # ----------------------------------------------------
        # Actual elapsed time
        # ----------------------------------------------------

        elapsed_days = (
            future_dt - current_date
        ).days

        years = elapsed_days / 365.25

        if years <= 0:
            continue

        # ----------------------------------------------------
        # Annualized return
        # ----------------------------------------------------

        ann_return = (
            future_price / current_price
        ) ** (1.0 / years) - 1.0

        # ----------------------------------------------------
        # Risk metrics inside the 2Y window
        #
        # Window includes current observation and future
        # endpoint.
        # ----------------------------------------------------

        window_prices = prices[
            i : future_position + 1
        ]

        if len(window_prices) < 2:
            continue

        window_returns = pd.Series(
            window_prices
        ).pct_change().dropna()

        if len(window_returns) == 0:
            continue

        volatility = (
            window_returns.std(
                ddof=1
            )
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

        window_running_peak = (
            pd.Series(window_prices)
            .cummax()
        )

        window_drawdowns = (
            pd.Series(window_prices)
            / window_running_peak
        ) - 1.0

        mdd = window_drawdowns.min()

        # ----------------------------------------------------
        # Raw risk-adjusted return
        #
        # Diagnostic only.
        # ----------------------------------------------------

        if (
            np.isfinite(volatility)
            and volatility > 0
        ):
            raw_risk_adjusted = (
                ann_return / volatility
            )
        else:
            raw_risk_adjusted = np.nan

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        future_date[i] = future_dt
        future_adj_close[i] = future_price
        actual_years[i] = years

        annualized_return[i] = ann_return
        annualized_volatility[i] = volatility
        max_drawdown[i] = mdd
        risk_adjusted_return[i] = raw_risk_adjusted

    result = group.copy()

    result["target_date_2y"] = (
        result["date"]
        + pd.DateOffset(years=HORIZON_YEARS)
    )

    result["future_date_2y"] = future_date
    result["future_adj_close_2y"] = future_adj_close
    result["actual_years_2y"] = actual_years

    result["annualized_return_2y"] = annualized_return
    result["annualized_volatility_2y"] = (
        annualized_volatility
    )
    result["max_drawdown_2y"] = max_drawdown
    result["risk_adjusted_return_2y"] = (
        risk_adjusted_return
    )

    return result


# ============================================================
# GENERATE DIAGNOSTICS
# ============================================================

def generate_diagnostics(prices):

    print()
    print("=" * 70)
    print("GENERATING STOCK 2Y DIAGNOSTICS")
    print("=" * 70)

    results = []

    tickers = sorted(
        prices["ticker"].unique()
    )

    for i, ticker in enumerate(tickers, start=1):

        group = prices[
            prices["ticker"] == ticker
        ].copy()

        result = calculate_stock_2y_metrics(
            group
        )

        results.append(result)

        print(
            f"[{i:3d}/{len(tickers)}] "
            f"{ticker:<20} "
            f"Rows: {len(result):>6}"
        )

    diagnostics = pd.concat(
        results,
        ignore_index=True
    )

    diagnostics = diagnostics.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return diagnostics


# ============================================================
# VALIDATION
# ============================================================

def validate_diagnostics(
    diagnostics,
    prices
):

    print()
    print("=" * 70)
    print("VALIDATING STOCK 2Y DIAGNOSTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    print(
        f"Diagnostic rows: {len(diagnostics):,}"
    )

    print(
        f"Stock count: "
        f"{diagnostics['ticker'].nunique()}"
    )

    print(
        f"Expected stocks: "
        f"{prices['ticker'].nunique()}"
    )

    if len(diagnostics) != len(prices):
        raise ValueError(
            "Diagnostic row count does not match "
            "source row count."
        )

    if (
        diagnostics["ticker"].nunique()
        != prices["ticker"].nunique()
    ):
        raise ValueError(
            "Diagnostic stock count does not match "
            "source stock count."
        )

    # --------------------------------------------------------
    # Duplicate ticker/date
    # --------------------------------------------------------

    duplicate_count = diagnostics.duplicated(
        ["ticker", "date"]
    ).sum()

    print(
        f"Duplicate ticker/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate ticker/date rows found."
        )

    # --------------------------------------------------------
    # DYNAMIC should never appear
    # --------------------------------------------------------

    dynamic_count = (
        diagnostics["ticker"]
        .eq("DYNAMIC.NS")
        .sum()
    )

    print(
        f"DYNAMIC rows: {dynamic_count}"
    )

    if dynamic_count != 0:
        raise ValueError(
            "DYNAMIC.NS unexpectedly present."
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    min_date = diagnostics["date"].min()
    max_date = diagnostics["date"].max()

    print(
        f"Diagnostic date range: "
        f"{min_date.date()} -> {max_date.date()}"
    )

    if min_date != EXPECTED_MIN_DATE:
        raise ValueError(
            f"Diagnostic minimum date is {min_date.date()}, "
            f"expected {EXPECTED_MIN_DATE.date()}."
        )

    if max_date != EXPECTED_MAX_DATE:
        raise ValueError(
            f"Diagnostic maximum date is {max_date.date()}, "
            f"expected {EXPECTED_MAX_DATE.date()}."
        )

    # --------------------------------------------------------
    # Price validity
    # --------------------------------------------------------

    invalid_prices = (
        diagnostics["adj_close"].isna()
        | (diagnostics["adj_close"] <= 0)
    ).sum()

    print(
        f"Invalid Adj Close values: "
        f"{invalid_prices}"
    )

    if invalid_prices != 0:
        raise ValueError(
            "Invalid Adj Close values found."
        )

    # --------------------------------------------------------
    # Valid 2Y outcomes
    # --------------------------------------------------------

    valid_2y = diagnostics[
        diagnostics["future_date_2y"].notna()
    ].copy()

    missing_2y = (
        len(diagnostics) - len(valid_2y)
    )

    print(
        f"Valid 2Y outcomes: "
        f"{len(valid_2y):,}"
    )

    print(
        f"Missing 2Y outcomes: "
        f"{missing_2y:,}"
    )

    # --------------------------------------------------------
    # Target-date validation
    # --------------------------------------------------------

    violations = (
        valid_2y["future_date_2y"]
        < valid_2y["target_date_2y"]
    ).sum()

    print(
        f"2Y target-date violations: "
        f"{violations}"
    )

    if violations != 0:
        raise ValueError(
            "Future date occurs before target date."
        )

    # --------------------------------------------------------
    # Actual elapsed years
    # --------------------------------------------------------

    invalid_years = (
        valid_2y["actual_years_2y"].isna()
        | (
            valid_2y["actual_years_2y"]
            <= 0
        )
    ).sum()

    print(
        f"Invalid elapsed years: "
        f"{invalid_years}"
    )

    if invalid_years != 0:
        raise ValueError(
            "Invalid elapsed years found."
        )

    # --------------------------------------------------------
    # Annualized return validity
    # --------------------------------------------------------

    invalid_return = (
        valid_2y["annualized_return_2y"]
        .isna()
        | ~np.isfinite(
            valid_2y["annualized_return_2y"]
        )
    ).sum()

    print(
        f"Invalid annualized returns: "
        f"{invalid_return}"
    )

    if invalid_return != 0:
        raise ValueError(
            "Invalid annualized returns found."
        )

    # --------------------------------------------------------
    # Volatility validity
    # --------------------------------------------------------

    invalid_volatility = (
        valid_2y[
            "annualized_volatility_2y"
        ].isna()
        | ~np.isfinite(
            valid_2y[
                "annualized_volatility_2y"
            ]
        )
        | (
            valid_2y[
                "annualized_volatility_2y"
            ] < 0
        )
    ).sum()

    print(
        f"Invalid volatility: "
        f"{invalid_volatility}"
    )

    if invalid_volatility != 0:
        raise ValueError(
            "Invalid volatility values found."
        )

    # --------------------------------------------------------
    # Drawdown validity
    # --------------------------------------------------------

    invalid_drawdown = (
        valid_2y[
            "max_drawdown_2y"
        ].isna()
        | ~np.isfinite(
            valid_2y[
                "max_drawdown_2y"
            ]
        )
        | (
            valid_2y[
                "max_drawdown_2y"
            ] > 0
        )
    ).sum()

    print(
        f"Invalid drawdown: "
        f"{invalid_drawdown}"
    )

    if invalid_drawdown != 0:
        raise ValueError(
            "Invalid drawdown values found."
        )

    # --------------------------------------------------------
    # Duplicate output validation
    # --------------------------------------------------------

    duplicate_output = diagnostics.duplicated(
        ["ticker", "date"]
    ).sum()

    print(
        f"Duplicate output rows: "
        f"{duplicate_output}"
    )

    if duplicate_output != 0:
        raise ValueError(
            "Duplicate output rows found."
        )

    print()
    print("VALIDATION STATUS: PASS")


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def print_summary(diagnostics):

    valid = diagnostics[
        diagnostics["future_date_2y"].notna()
    ].copy()

    print()
    print("=" * 70)
    print("STOCK 2Y DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"Rows: {len(diagnostics):,}"
    )

    print(
        f"Stocks: "
        f"{diagnostics['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{diagnostics['date'].min().date()} "
        f"-> "
        f"{diagnostics['date'].max().date()}"
    )

    print(
        f"Valid 2Y outcomes: "
        f"{len(valid):,}"
    )

    print(
        f"Missing 2Y outcomes: "
        f"{len(diagnostics) - len(valid):,}"
    )

    # --------------------------------------------------------
    # Annualized return
    # --------------------------------------------------------

    print()
    print("ANNUALIZED RETURN")
    print("-" * 70)

    print(
        valid[
            "annualized_return_2y"
        ].describe(
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

    # --------------------------------------------------------
    # Annualized volatility
    # --------------------------------------------------------

    print()
    print("ANNUALIZED VOLATILITY")
    print("-" * 70)

    print(
        valid[
            "annualized_volatility_2y"
        ].describe(
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

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    print()
    print("MAXIMUM DRAWDOWN")
    print("-" * 70)

    print(
        valid[
            "max_drawdown_2y"
        ].describe(
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

    # --------------------------------------------------------
    # Raw risk-adjusted return
    # --------------------------------------------------------

    print()
    print("RAW RISK-ADJUSTED RETURN")
    print("-" * 70)

    print(
        valid[
            "risk_adjusted_return_2y"
        ].describe(
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

    # --------------------------------------------------------
    # Extreme-return counts
    # --------------------------------------------------------

    print()
    print("RETURN EXTREMES")
    print("-" * 70)

    print(
        f"Return > 50%: "
        f"{(valid['annualized_return_2y'] > 0.50).sum():,}"
    )

    print(
        f"Return > 100%: "
        f"{(valid['annualized_return_2y'] > 1.00).sum():,}"
    )

    print(
        f"Return > 200%: "
        f"{(valid['annualized_return_2y'] > 2.00).sum():,}"
    )

    print(
        f"Return > 500%: "
        f"{(valid['annualized_return_2y'] > 5.00).sum():,}"
    )

    print(
        f"Return < -25%: "
        f"{(valid['annualized_return_2y'] < -0.25).sum():,}"
    )

    print(
        f"Return < -50%: "
        f"{(valid['annualized_return_2y'] < -0.50).sum():,}"
    )

    print(
        f"Return < -75%: "
        f"{(valid['annualized_return_2y'] < -0.75).sum():,}"
    )

    # --------------------------------------------------------
    # Risk extremes
    # --------------------------------------------------------

    print()
    print("RISK EXTREMES")
    print("-" * 70)

    print(
        f"Volatility < 1%: "
        f"{(valid['annualized_volatility_2y'] < 0.01).sum():,}"
    )

    print(
        f"Volatility < 2%: "
        f"{(valid['annualized_volatility_2y'] < 0.02).sum():,}"
    )

    print(
        f"Volatility < 5%: "
        f"{(valid['annualized_volatility_2y'] < 0.05).sum():,}"
    )

    print(
        f"Drawdown > -1%: "
        f"{(valid['max_drawdown_2y'] > -0.01).sum():,}"
    )

    print(
        f"Drawdown < -50%: "
        f"{(valid['max_drawdown_2y'] < -0.50).sum():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOCK 2Y DIAGNOSTICS")
    print("=" * 70)

    print(
        "Method: fixed 2-calendar-year horizon"
    )

    print(
        "Date handling: preserve calendar date; "
        "strip timezone without UTC conversion"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    prices = load_all_stocks()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    diagnostics = generate_diagnostics(
        prices
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_diagnostics(
        diagnostics,
        prices
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        diagnostics
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    diagnostics.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Reload validation
    # --------------------------------------------------------

    reloaded = pd.read_parquet(
        OUTPUT_FILE
    )

    print()
    print(
        f"Reloaded rows: "
        f"{len(reloaded):,}"
    )

    print(
        f"Reloaded stocks: "
        f"{reloaded['ticker'].nunique()}"
    )

    if len(reloaded) != len(diagnostics):
        raise ValueError(
            "Reloaded row count mismatch."
        )

    if (
        reloaded["ticker"].nunique()
        != diagnostics["ticker"].nunique()
    ):
        raise ValueError(
            "Reloaded stock count mismatch."
        )

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()