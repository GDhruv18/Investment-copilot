
from typing import Literal
from sqlalchemy import text
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.services.etf_service import (
    get_etf_details,
    get_etf_history,
    get_etf_features,
    resolve_etf_ticker,
)

from backend.app.db.session import get_db
from backend.app.services.etf_service import (
    get_etf_details,
    resolve_etf_ticker,
)
from backend.app.services.market_data_service import fetch_intraday_data

router = APIRouter(prefix="/etfs", tags=["ETFs"])

@router.get("/search")
def search_etfs(
    q: str = Query(
        ...,
        min_length=1,
        description="ETF name or ticker",
    ),
    limit: int = Query(default=10, ge=1, le=20),
    db: Session = Depends(get_db),
):
    query = q.strip().casefold()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Search query cannot be empty.",
        )

    search_query = text("""
        SELECT ticker, etf_name
        FROM etfs
        WHERE
            LOWER(ticker) LIKE :query
            OR LOWER(etf_name) LIKE :query
        ORDER BY
            CASE
                WHEN LOWER(ticker) LIKE :prefix THEN 0
                WHEN LOWER(etf_name) LIKE :prefix THEN 1
                ELSE 2
            END,
            etf_name
        LIMIT :limit
    """)

    results = db.execute(
        search_query,
        {
            "query": f"%{query}%",
            "prefix": f"{query}%",
            "limit": limit,
        },
    ).mappings().all()

    return {
        "query": q,
        "total": len(results),
        "results": [
            {
                "ticker": row["ticker"],
                "etf_name": row["etf_name"],
            }
            for row in results
        ],
    }


@router.get("/")
def list_etfs(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    count_query = text("""
        SELECT COUNT(*)
        FROM etfs
    """)

    total = db.execute(count_query).scalar()

    list_query = text("""
        SELECT
            ticker,
            etf_name,
            issuer,
            underlying_index,
            expense_ratio,
            aum,
            tracking_error
        FROM etfs
        ORDER BY etf_name
        LIMIT :limit
        OFFSET :offset
    """)

    results = db.execute(
        list_query,
        {
            "limit": limit,
            "offset": offset,
        },
    ).mappings().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "ticker": row["ticker"],
                "etf_name": row["etf_name"],
                "issuer": row["issuer"],
                "underlying_index": row["underlying_index"],
                "expense_ratio": row["expense_ratio"],
                "aum": row["aum"],
                "tracking_error": row["tracking_error"],
            }
            for row in results
        ],
    }


def resolve_identifier(db: Session, identifier: str) -> str:
    try:
        return resolve_etf_ticker(db, identifier)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{ticker}/intraday")
def get_etf_intraday(
    ticker: str,
    period: Literal["1d", "5d"] = "1d",
    interval: Literal["1m", "5m", "15m", "30m", "60m"] = "5m",
    db: Session = Depends(get_db),
):
    resolved_ticker = resolve_identifier(db, ticker)

    try:
        return fetch_intraday_data(
            ticker=resolved_ticker,
            period=period,
            interval=interval,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="No intraday market data available for this ETF.",
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve intraday market data from the provider.",
        )


@router.get("/{ticker}/features")
def get_etf_features_endpoint(
    ticker: str,
    db: Session = Depends(get_db),
):
    resolved_ticker = resolve_identifier(db, ticker)

    try:
        return get_etf_features(
            db=db,
            ticker=resolved_ticker,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="ETF not found.",
        )


@router.get("/{ticker}/details")
def get_etf_details_endpoint(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = Query(default="1y"),
    db: Session = Depends(get_db),
):
    resolved_ticker = resolve_identifier(db, ticker)

    try:
        return get_etf_details(
            db=db,
            ticker=resolved_ticker,
            period=period,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="ETF not found.")


@router.get("/{ticker}/history")
def get_etf_history_endpoint(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = Query(default="1y"),
    db: Session = Depends(get_db),
):
    resolved_ticker = resolve_identifier(db, ticker)

    try:
        return get_etf_history(
            db=db,
            ticker=resolved_ticker,
            period=period,
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="ETF not found.",
        )
