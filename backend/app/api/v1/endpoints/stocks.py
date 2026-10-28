
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.services.stock_service import get_stock_database_details
from backend.app.services.stock_search_service import (
    search_stocks,
    resolve_stock_ticker,
)
from backend.app.services.market_data_service import (
    fetch_intraday_data,
    fetch_daily_data,
)


router = APIRouter(
    prefix="/stocks",
    tags=["Stocks"],
)


def resolve_identifier(identifier: str) -> str:
    """Resolve a common company name or ticker."""
    try:
        return resolve_stock_ticker(identifier)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@router.get("/search")
def search_stock_endpoint(
    q: str = Query(..., min_length=1, description="Company name or ticker"),
    limit: int = Query(default=10, ge=1, le=20),
):
    results = search_stocks(q, limit)

    return {
        "query": q,
        "total": len(results),
        "results": results,
    }


@router.get("/{ticker}/intraday")
def get_intraday_data(
    ticker: str,
    period: Literal["1d", "5d"] = "1d",
    interval: Literal["1m", "5m", "15m", "30m", "60m"] = "5m",
):
    resolved_ticker = resolve_identifier(ticker)

    try:
        return fetch_intraday_data(
            ticker=resolved_ticker,
            period=period,
            interval=interval,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="No intraday market data available for this stock.",
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve intraday market data from the provider.",
        )


@router.get("/{ticker}/daily")
def get_daily_data(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = Query(
        default="1y"
    ),
):
    resolved_ticker = resolve_identifier(ticker)

    try:
        return fetch_daily_data(
            ticker=resolved_ticker,
            period=period,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="No daily market data available for this stock.",
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve daily market data from the provider.",
        )


@router.get("/{ticker}/details")
def get_stock_details(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = "1y",
    db: Session = Depends(get_db),
):
    resolved_ticker = resolve_identifier(ticker)

    try:
        return get_stock_database_details(
            db=db,
            ticker=resolved_ticker,
            period=period,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Stock not found.",
        )
