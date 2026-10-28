
from datetime import date
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.market_data_service import fetch_intraday_data


def _serialize_row(row):
    """Convert database values into JSON-compatible values."""
    if row is None:
        return None

    result = {}

    for key, value in dict(row).items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (date,)):
            result[key] = value.isoformat()
        else:
            result[key] = value

    return result


def _period_start(period: str):
    """Calculate the earliest date for a historical chart period."""
    from datetime import timedelta

    today = date.today()

    period_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }

    if period == "max":
        return None

    return today - timedelta(days=period_days[period])


def get_etf_details(
    db: Session,
    ticker: str,
    period: str = "1y",
) -> dict:
    """Retrieve ETF metadata, technical features and historical prices."""

    ticker = ticker.strip().upper()

    # 1. Fetch ETF metadata.
    metadata_query = text("""
        SELECT *
        FROM etfs
        WHERE UPPER(ticker) = :ticker
        LIMIT 1
    """)

    metadata_row = db.execute(
        metadata_query,
        {"ticker": ticker},
    ).mappings().first()

    if metadata_row is None:
        raise ValueError("ETF not found.")

    # 2. Fetch the latest available technical features for this ETF.
    features_query = text("""
        SELECT *
        FROM etf_features
        WHERE UPPER(ticker) = :ticker
        ORDER BY date DESC
        LIMIT 1
    """)

    features_row = db.execute(
        features_query,
        {"ticker": ticker},
    ).mappings().first()

    # 3. Fetch historical prices from PostgreSQL.
    start_date = _period_start(period)

    if start_date is None:
        history_query = text("""
            SELECT date, open, high, low, close, volume, adj_close
            FROM etf_prices
            WHERE UPPER(ticker) = :ticker
            ORDER BY date ASC
        """)
        params = {"ticker": ticker}
    else:
        history_query = text("""
            SELECT date, open, high, low, close, volume, adj_close
            FROM etf_prices
            WHERE UPPER(ticker) = :ticker
              AND date >= :start_date
            ORDER BY date ASC
        """)
        params = {
            "ticker": ticker,
            "start_date": start_date,
        }

    history_rows = db.execute(
        history_query,
        params,
    ).mappings().all()

    history = [
        _serialize_row(row)
        for row in history_rows
    ]

    # 4. Fetch latest available intraday data.
    try:
        intraday = fetch_intraday_data(
            ticker=ticker,
            period="1d",
            interval="5m",
        )

        market_data = {
            "available": True,
            "source": intraday["source"],
            "interval": intraday["interval"],
            "latest": intraday["latest"],
        }

    except Exception:
        market_data = {
            "available": False,
            "source": "Yahoo Finance",
            "message": "Market data is currently unavailable.",
        }

    return {
        "etf": _serialize_row(metadata_row),
        "market_data": market_data,
        "technical_features": _serialize_row(features_row),
        "historical_data": {
            "source": "PostgreSQL",
            "period": period,
            "total_records": len(history),
            "history": history,
            "latest": history[-1] if history else None,
        },
    }


def get_etf_history(
    db: Session,
    ticker: str,
    period: str = "1y",
) -> dict:
    """Retrieve historical ETF price data from PostgreSQL."""

    ticker = ticker.strip().upper()

    # Verify that the ETF exists.
    exists_query = text("""
        SELECT ticker
        FROM etfs
        WHERE UPPER(ticker) = :ticker
        LIMIT 1
    """)

    exists = db.execute(
        exists_query,
        {"ticker": ticker},
    ).first()

    if exists is None:
        raise ValueError("ETF not found.")

    # Calculate the historical date range.
    start_date = _period_start(period)

    if start_date is None:
        history_query = text("""
            SELECT date, open, high, low, close, volume, adj_close
            FROM etf_prices
            WHERE UPPER(ticker) = :ticker
            ORDER BY date ASC
        """)
        params = {"ticker": ticker}

    else:
        history_query = text("""
            SELECT date, open, high, low, close, volume, adj_close
            FROM etf_prices
            WHERE UPPER(ticker) = :ticker
              AND date >= :start_date
            ORDER BY date ASC
        """)
        params = {
            "ticker": ticker,
            "start_date": start_date,
        }

    rows = db.execute(
        history_query,
        params,
    ).mappings().all()

    history = [
        _serialize_row(row)
        for row in rows
    ]

    return {
        "ticker": ticker,
        "source": "PostgreSQL",
        "period": period,
        "total_records": len(history),
        "history": history,
        "latest": history[-1] if history else None,
    }


def get_etf_features(
    db: Session,
    ticker: str,
) -> dict:
    """Retrieve the latest available technical features for an ETF."""

    ticker = ticker.strip().upper()

    # Verify that the ETF exists.
    exists_query = text("""
        SELECT ticker
        FROM etfs
        WHERE UPPER(ticker) = :ticker
        LIMIT 1
    """)

    exists = db.execute(
        exists_query,
        {"ticker": ticker},
    ).first()

    if exists is None:
        raise ValueError("ETF not found.")

    # Fetch the latest available feature record.
    features_query = text("""
        SELECT *
        FROM etf_features
        WHERE UPPER(ticker) = :ticker
        ORDER BY date DESC
        LIMIT 1
    """)

    feature_row = db.execute(
        features_query,
        {"ticker": ticker},
    ).mappings().first()

    if feature_row is None:
        return {
            "ticker": ticker,
            "available": False,
            "message": "No technical features available for this ETF.",
            "features": None,
        }

    return {
        "ticker": ticker,
        "available": True,
        "source": "PostgreSQL",
        "features": _serialize_row(feature_row),
    }


def resolve_etf_ticker(db: Session, identifier: str) -> str:
    """
    Resolve an ETF name or ticker into its database ticker.
    """

    identifier = identifier.strip().casefold()

    if not identifier:
        raise ValueError("ETF identifier cannot be empty.")

    query = text("""
        SELECT ticker, etf_name
        FROM etfs
    """)

    rows = db.execute(query).mappings().all()

    # First, check exact matches.
    exact_matches = [
        row for row in rows
        if (
            row["ticker"].casefold() == identifier
            or row["etf_name"].casefold() == identifier
        )
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]["ticker"]

    if len(exact_matches) > 1:
        raise ValueError("Multiple ETFs match this identifier.")

    # Then, check partial matches.
    partial_matches = [
        row for row in rows
        if (
            identifier in row["ticker"].casefold()
            or identifier in row["etf_name"].casefold()
        )
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]["ticker"]

    if len(partial_matches) > 1:
        raise ValueError(
            "Multiple ETFs match this name. "
            "Please use a more specific name or ticker."
        )

    raise ValueError("ETF not found.")