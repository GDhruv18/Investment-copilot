
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_CANDIDATES = [
    PROJECT_ROOT / "data_pipeline" / "config" / "stocks_100.json",
    PROJECT_ROOT / "data_pipeline" / "config" / "stocks.json",
]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fundamentals"
    / "yfinance"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fundamentals"
)

REQUEST_DELAY_SECONDS = 0.5

STATEMENT_TYPES = {
    "annual_income_statement": "income_stmt",
    "quarterly_income_statement": "quarterly_income_stmt",
    "annual_balance_sheet": "balance_sheet",
    "quarterly_balance_sheet": "quarterly_balance_sheet",
    "annual_cash_flow": "cashflow",
    "quarterly_cash_flow": "quarterly_cashflow",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    """Convert common pandas/numpy/date values to JSON-safe values."""
    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return str(value)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_stock_universe() -> list[str]:
    config_path = next(
        (path for path in CONFIG_CANDIDATES if path.exists()),
        None,
    )

    if config_path is None:
        expected = "\n".join(str(p) for p in CONFIG_CANDIDATES)
        raise FileNotFoundError(
            "Could not find the stock-universe JSON file. "
            "Checked:\n" + expected
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if isinstance(config, dict):
        raw_stocks = config.get("stocks")
    elif isinstance(config, list):
        raw_stocks = config
    else:
        raw_stocks = None

    if not isinstance(raw_stocks, list) or not raw_stocks:
        raise ValueError(
            f"Unexpected stock-universe structure in {config_path}. "
            "Expected a list or an object containing a 'stocks' list."
        )

    tickers = []

    for item in raw_stocks:
        if isinstance(item, str):
            ticker = item
        elif isinstance(item, dict):
            ticker = (
                item.get("ticker")
                or item.get("symbol")
                or item.get("yf_ticker")
            )
        else:
            ticker = None

        if ticker:
            ticker = str(ticker).strip().upper()
            if ticker:
                tickers.append(ticker)

    # Preserve order while removing duplicate tickers.
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        raise ValueError("No usable tickers found in the stock universe.")

    logger.info("Universe file: %s", config_path)
    logger.info("Unique tickers found: %d", len(tickers))

    return tickers


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def save_statement(
    statement: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    """
    Save a statement with line items as rows and statement
    periods as columns.

    Column dates are financial period-end dates, not necessarily
    the dates on which the reports became publicly available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if statement is None or statement.empty:
        return {
            "available": False,
            "rows": 0,
            "periods": 0,
            "latest_period_end": None,
        }

    frame = statement.copy()
    frame.index = frame.index.astype(str)
    frame.columns = [str(column) for column in frame.columns]
    frame.index.name = "financial_item"

    frame.to_csv(path, encoding="utf-8")

    period_dates = pd.to_datetime(
        frame.columns,
        errors="coerce",
    ).dropna()

    latest_period = (
        period_dates.max().date().isoformat()
        if len(period_dates) > 0
        else None
    )

    return {
        "available": True,
        "rows": int(frame.shape[0]),
        "periods": int(frame.shape[1]),
        "latest_period_end": latest_period,
        "file": str(path.relative_to(PROJECT_ROOT)),
    }


# ---------------------------------------------------------
# Download one company
# ---------------------------------------------------------

def collect_one_ticker(ticker_symbol: str) -> dict[str, Any]:
    retrieved_at = utc_now()

    ticker_dir = OUTPUT_ROOT / safe_filename(ticker_symbol)
    ticker_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "ticker": ticker_symbol,
        "retrieved_at_utc": retrieved_at,
        "source": "Yahoo Finance via yfinance",
        "status": "started",
        "info_available": False,
        "info_error": None,
        "statement_errors": [],
        "statements": {},
    }

    logger.info("Downloading %s", ticker_symbol)

    ticker = yf.Ticker(ticker_symbol)

    # Company profile and current key metrics.
    try:
        info = ticker.get_info()

        save_json(
            ticker_dir / "info.json",
            {
                "ticker": ticker_symbol,
                "source": "Yahoo Finance via yfinance",
                "retrieved_at_utc": retrieved_at,
                "data": info,
            },
        )

        result["info_available"] = bool(info)
        result["info_fields_count"] = len(info)
        result["info_fields_present"] = sorted(info.keys())

    except Exception as exc:
        result["info_error"] = str(exc)
        logger.warning(
            "%s: profile download failed: %s",
            ticker_symbol,
            exc,
        )

    # Annual and quarterly financial statements.
    for output_name, attribute_name in STATEMENT_TYPES.items():
        try:
            statement = getattr(ticker, attribute_name)

            details = save_statement(
                statement,
                ticker_dir / f"{output_name}.csv",
            )

            result["statements"][output_name] = details

            if not details["available"]:
                logger.info(
                    "%s: %s unavailable or empty",
                    ticker_symbol,
                    output_name,
                )

        except Exception as exc:
            result["statements"][output_name] = {
                "available": False,
                "error": str(exc),
            }

            result["statement_errors"].append(
                {
                    "statement": output_name,
                    "error": str(exc),
                }
            )

            logger.warning(
                "%s: %s failed: %s",
                ticker_symbol,
                output_name,
                exc,
            )

    result["completed_at_utc"] = utc_now()

    if result["info_available"] or any(
        details.get("available", False)
        for details in result["statements"].values()
    ):
        result["status"] = "success"
    else:
        result["status"] = "failed"

    save_json(ticker_dir / "collection_metadata.json", result)

    return result


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    tickers = load_stock_universe()
    results = []

    for index, ticker_symbol in enumerate(tickers, start=1):
        logger.info(
            "Processing %d/%d: %s",
            index,
            len(tickers),
            ticker_symbol,
        )

        try:
            result = collect_one_ticker(ticker_symbol)
        except Exception as exc:
            logger.exception(
                "Unexpected error for %s",
                ticker_symbol,
            )

            result = {
                "ticker": ticker_symbol,
                "status": "failed",
                "unexpected_error": str(exc),
                "retrieved_at_utc": utc_now(),
            }

        results.append(result)

        # Save progress after each ticker so a partial run
        # still leaves a useful coverage report.
        progress_path = REPORT_ROOT / "yfinance_fundamentals_coverage.csv"
        pd.json_normalize(results, sep=".").to_csv(
            progress_path,
            index=False,
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    report = pd.json_normalize(results, sep=".")
    report_path = REPORT_ROOT / "yfinance_fundamentals_coverage.csv"
    report.to_csv(report_path, index=False)

    summary = {
        "source": "Yahoo Finance via yfinance",
        "started_or_completed_at_utc": utc_now(),
        "tickers_requested": len(tickers),
        "successful_tickers": int(
            (report["status"] == "success").sum()
        ),
        "failed_tickers": int(
            (report["status"] == "failed").sum()
        ),
        "coverage_report": str(
            report_path.relative_to(PROJECT_ROOT)
        ),
        "raw_data_directory": str(
            OUTPUT_ROOT.relative_to(PROJECT_ROOT)
        ),
        "database_modified": False,
        "models_modified": False,
    }

    save_json(
        REPORT_ROOT / "yfinance_fundamentals_run_summary.json",
        summary,
    )

    print("\n" + "=" * 65)
    print("YAHOO FINANCE FUNDAMENTALS COLLECTION SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=2))
    print("=" * 65)


if __name__ == "__main__":
    main()
