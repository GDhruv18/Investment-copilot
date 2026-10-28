
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.market_data_service import (
    normalize_ticker,
    fetch_intraday_data,
    fetch_daily_data,
)
from backend.app.services.stock_search_service import load_stock_master


def _serialize_row(row):
    if row is None:
        return None

    result = {}

    for key, value in dict(row).items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        else:
            result[key] = value

    return result


def get_stock_database_details(
    db: Session,
    ticker: str,
    period: str = "1y",
) -> dict:

    yahoo_ticker = normalize_ticker(ticker)

    # Find company information in the stock master.
    stocks = load_stock_master()

    stock = next(
        (
            item for item in stocks
            if item["yahoo_ticker"].upper() == yahoo_ticker.upper()
            or item["internal_ticker"].upper() == ticker.upper()
        ),
        None,
    )

    if stock is None:
        raise ValueError("Stock not found in the stock master.")

    # Fetch the latest technical features.
    feature_query = text("""
        SELECT *
        FROM stock_features
        WHERE ticker = :ticker
        ORDER BY date DESC
        LIMIT 1
    """)

    feature_row = db.execute(
        feature_query,
        {"ticker": yahoo_ticker},
    ).mappings().first()

    # Fetch the latest available fundamentals.
    fundamentals_query = text("""
        SELECT *
        FROM company_fundamentals
        WHERE ticker = :ticker
        ORDER BY report_date DESC
        LIMIT 1
    """)

    fundamentals_row = db.execute(
        fundamentals_query,
        {"ticker": yahoo_ticker},
    ).mappings().first()

    # Fetch the latest available intraday market data.
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

    # Fetch daily historical prices for the chart.
    try:
        historical_data = fetch_daily_data(
            ticker=ticker,
            period=period,
        )

    except Exception:
        historical_data = {
            "available": False,
            "source": "Yahoo Finance",
            "message": "Historical price data is currently unavailable.",
            "history": [],
        }

    return {
        "company": stock,
        "market_data": market_data,
        "technical_features": _serialize_row(feature_row),
        "fundamentals": _serialize_row(fundamentals_row),
        "historical_data": historical_data,
    }
