
import yfinance as yf
import pandas as pd


def normalize_ticker(ticker: str) -> str:
    """Convert internal recommendation tickers to Yahoo Finance format."""
    ticker = ticker.strip().upper()

    if ticker.endswith("_NS"):
        return ticker[:-3] + ".NS"

    return ticker


def fetch_intraday_data(
    ticker: str,
    period: str = "1d",
    interval: str = "5m",
) -> dict:
    """Fetch recent intraday OHLCV data from Yahoo Finance."""

    yahoo_ticker = normalize_ticker(ticker)

    df = yf.Ticker(yahoo_ticker).history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(
            f"No intraday market data available for {yahoo_ticker}"
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        raise ValueError(
            f"No valid intraday candles available for {yahoo_ticker}"
        )

    candles = []

    for timestamp, row in df.iterrows():
        candles.append({
            "timestamp": timestamp.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
        })

    return {
        "ticker": yahoo_ticker,
        "source": "Yahoo Finance",
        "interval": interval,
        "latest": candles[-1],
        "candles": candles,
    }


def fetch_daily_data(
    ticker: str,
    period: str = "1y",
) -> dict:
    """Fetch daily historical OHLCV data from Yahoo Finance."""

    yahoo_ticker = normalize_ticker(ticker)

    df = yf.Ticker(yahoo_ticker).history(
        period=period,
        interval="1d",
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(
            f"No daily market data available for {yahoo_ticker}"
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    records = []

    for timestamp, row in df.iterrows():
        records.append({
            "date": timestamp.date().isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
        })

    return {
        "ticker": yahoo_ticker,
        "source": "Yahoo Finance",
        "history": records,
        "latest": records[-1],
    }
