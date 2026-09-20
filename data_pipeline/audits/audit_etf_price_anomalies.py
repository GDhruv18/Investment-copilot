import os
import math
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

OUTPUT_FILE = "data/targets/etf_price_anomaly_audit.csv"

# Known suspicious ETFs from the forward-return audit
FOCUS_ETFS = [
    "GOLDBEES.NS",
    "SETFGOLD.NS",
    "HDFCNEXT50.NS",
    "PSUBANK.NS",
]

# A price movement larger than this is suspicious.
# This does NOT mean it is necessarily wrong.
EXTREME_RATIO = 3.0

# Number of rows shown around known suspicious dates
CONTEXT_ROWS = 15


# ============================================================
# DATABASE CONNECTION
# ============================================================

def create_db_engine():
    missing = []

    if not DB_NAME:
        missing.append("DB_NAME")
    if not DB_USER:
        missing.append("DB_USER")
    if not DB_PASSWORD:
        missing.append("DB_PASSWORD")
    if not DB_HOST:
        missing.append("DB_HOST")
    if not DB_PORT:
        missing.append("DB_PORT")

    if missing:
        raise ValueError(
            "Missing database environment variables: "
            + ", ".join(missing)
        )

    connection_string = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(connection_string)


# ============================================================
# LOAD ETF DATA
# ============================================================

def load_etf_data(engine):
    query = """
        SELECT
            ticker,
            date,
            open,
            high,
            low,
            close,
            adj_close,
            volume
        FROM etf_prices
        ORDER BY ticker, date
    """

    df = pd.read_sql_query(text(query), engine)

    if df.empty:
        raise ValueError("No ETF data found in etf_prices table.")

    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# CALCULATE DAY-TO-DAY PRICE RATIOS
# ============================================================

def calculate_ratios(df):
    df = df.sort_values(["ticker", "date"]).copy()

    df["previous_close"] = (
        df.groupby("ticker")["close"]
        .shift(1)
    )

    df["previous_adj_close"] = (
        df.groupby("ticker")["adj_close"]
        .shift(1)
    )

    df["close_ratio"] = (
        df["close"] / df["previous_close"]
    )

    df["adj_close_ratio"] = (
        df["adj_close"] / df["previous_adj_close"]
    )

    df["close_return"] = (
        df["close"] / df["previous_close"] - 1
    )

    df["adj_close_return"] = (
        df["adj_close"] / df["previous_adj_close"] - 1
    )

    return df


# ============================================================
# FIND EXTREME DAILY MOVEMENTS
# ============================================================

def find_extreme_movements(df):
    valid = df[
        df["close_ratio"].notna()
        & df["adj_close_ratio"].notna()
        & (df["close"] > 0)
        & (df["previous_close"] > 0)
        & (df["adj_close"] > 0)
        & (df["previous_adj_close"] > 0)
    ].copy()

    # Distance from 1.0 on logarithmic scale.
    # This treats 10x and 0.1x symmetrically.
    valid["close_log_move"] = valid["close_ratio"].apply(
        lambda x: abs(math.log(x))
    )

    valid["adj_log_move"] = valid["adj_close_ratio"].apply(
        lambda x: abs(math.log(x))
    )

    # Flag either raw Close or Adj Close as suspicious.
    extreme = valid[
        (valid["close_ratio"] >= EXTREME_RATIO)
        | (valid["close_ratio"] <= 1 / EXTREME_RATIO)
        | (valid["adj_close_ratio"] >= EXTREME_RATIO)
        | (valid["adj_close_ratio"] <= 1 / EXTREME_RATIO)
    ].copy()

    extreme["largest_move"] = extreme[
        ["close_log_move", "adj_log_move"]
    ].max(axis=1)

    extreme = extreme.sort_values(
        "largest_move",
        ascending=False
    )

    return extreme


# ============================================================
# PRINT EXTREME MOVEMENTS
# ============================================================

def print_extreme_movements(extreme):
    print()
    print("=" * 100)
    print("TOP ETF DAILY PRICE DISCONTINUITIES")
    print("=" * 100)

    if extreme.empty:
        print("No extreme daily movements detected.")
        return

    display_columns = [
        "ticker",
        "date",
        "previous_close",
        "close",
        "close_ratio",
        "previous_adj_close",
        "adj_close",
        "adj_close_ratio",
    ]

    top = extreme.head(50)

    print(
        top[display_columns]
        .to_string(index=False)
    )


# ============================================================
# PRINT FOCUSED ETF HISTORY
# ============================================================

def print_focus_etf(df, ticker):
    ticker_df = df[
        df["ticker"] == ticker
    ].copy()

    if ticker_df.empty:
        print()
        print(f"{ticker}: NO DATA FOUND")
        return

    # Find the largest adjusted-price discontinuity
    valid = ticker_df[
        ticker_df["adj_close_ratio"].notna()
    ].copy()

    if valid.empty:
        print(f"{ticker}: insufficient data")
        return

    valid["adj_move"] = valid[
        "adj_close_ratio"
    ].apply(lambda x: abs(math.log(x)))

    suspicious_row = valid.loc[
        valid["adj_move"].idxmax()
    ]

    suspicious_date = suspicious_row["date"]

    # Find corresponding row position
    positions = ticker_df.index[
        ticker_df["date"] == suspicious_date
    ]

    if len(positions) == 0:
        return

    position = ticker_df.index.get_loc(positions[0])

    start_position = max(0, position - CONTEXT_ROWS)
    end_position = min(
        len(ticker_df),
        position + CONTEXT_ROWS + 1
    )

    context = ticker_df.iloc[
        start_position:end_position
    ].copy()

    print()
    print("=" * 100)
    print(f"FOCUSED AUDIT: {ticker}")
    print("=" * 100)

    print(
        f"Largest adjusted-price discontinuity: "
        f"{suspicious_date.date()}"
    )

    print()

    display_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "previous_close",
        "close_ratio",
        "previous_adj_close",
        "adj_close_ratio",
    ]

    print(
        context[display_columns]
        .to_string(index=False)
    )


# ============================================================
# COMPARE CLOSE VS ADJUSTED CLOSE
# ============================================================

def print_adjustment_relationship(df):
    print()
    print("=" * 100)
    print("CLOSE vs ADJUSTED CLOSE RELATIONSHIP")
    print("=" * 100)

    summary = []

    for ticker, group in df.groupby("ticker"):
        group = group.copy()

        valid = group[
            group["close"].notna()
            & group["adj_close"].notna()
            & (group["close"] > 0)
        ]

        if valid.empty:
            continue

        valid["adjustment_factor"] = (
            valid["adj_close"] / valid["close"]
        )

        summary.append({
            "ticker": ticker,
            "rows": len(valid),
            "first_date": valid["date"].min(),
            "last_date": valid["date"].max(),
            "min_adj_close_to_close": valid[
                "adjustment_factor"
            ].min(),
            "max_adj_close_to_close": valid[
                "adjustment_factor"
            ].max(),
        })

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        "max_adj_close_to_close",
        ascending=False
    )

    print(
        summary_df.to_string(index=False)
    )


# ============================================================
# SAVE AUDIT
# ============================================================

def save_audit(extreme):
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    extreme.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Full anomaly audit saved to:")
    print(OUTPUT_FILE)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("ETF PRICE ANOMALY AUDIT")
    print("=" * 100)

    print()
    print("Purpose:")
    print(
        "Investigate extreme ETF price movements before "
        "using ETF forward returns for ML targets."
    )

    print()
    print("This script DOES NOT modify any database data.")

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print()
    print("Connecting to PostgreSQL...")

    engine = create_db_engine()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("Loading ETF data...")

    df = load_etf_data(engine)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"ETF tickers: {df['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Calculate ratios
    # --------------------------------------------------------

    print()
    print("Calculating daily price ratios...")

    df = calculate_ratios(df)

    # --------------------------------------------------------
    # Extreme movements
    # --------------------------------------------------------

    extreme = find_extreme_movements(df)

    print(
        f"Extreme daily movements detected: "
        f"{len(extreme):,}"
    )

    print_extreme_movements(extreme)

    # --------------------------------------------------------
    # Focused ETFs
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FOCUSED ETF INVESTIGATION")
    print("=" * 100)

    for ticker in FOCUS_ETFS:
        print_focus_etf(df, ticker)

    # --------------------------------------------------------
    # Adjustment relationship
    # --------------------------------------------------------

    print_adjustment_relationship(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_audit(extreme)

    # --------------------------------------------------------
    # Final instructions
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

    print()
    print("IMPORTANT:")
    print("1. No ETF data was modified.")
    print("2. Do NOT clip extreme returns yet.")
    print("3. Do NOT generate ETF labels yet.")
    print("4. Do NOT train ETF models yet.")
    print()
    print(
        "Next step is to determine whether the extreme "
        "movements are genuine market events, corporate-action "
        "adjustments, or data discontinuities."
    )


if __name__ == "__main__":
    main()