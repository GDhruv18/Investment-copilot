
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent

RAW_ROOT = ROOT / "data" / "raw" / "fundamentals" / "yfinance"
OUTPUT_ROOT = ROOT / "data" / "clean" / "fundamentals" / "yfinance"
REPORT_ROOT = ROOT / "data" / "processed" / "fundamentals"

STATEMENTS = {
    "annual_income_statement": "annual_income_statement.csv",
    "quarterly_income_statement": "quarterly_income_statement.csv",
    "annual_balance_sheet": "annual_balance_sheet.csv",
    "quarterly_balance_sheet": "quarterly_balance_sheet.csv",
    "annual_cash_flow": "annual_cash_flow.csv",
    "quarterly_cash_flow": "quarterly_cash_flow.csv",
}

# Preserve the original provider fields; do not calculate or replace ratios here.
PROFILE_FIELDS = [
    "longName",
    "sector",
    "industry",
    "country",
    "currency",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "totalRevenue",
    "netIncomeToCommon",
    "totalDebt",
    "totalCash",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
    "operatingCashflow",
    "dividendYield",
    "beta",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "sharesOutstanding",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            return payload

    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)

    return {}


def read_profile(ticker_dir: Path) -> dict[str, Any]:
    payload = read_json(ticker_dir / "info.json")
    data = payload.get("data", {})

    if not isinstance(data, dict):
        data = {}

    metadata = read_json(ticker_dir / "collection_metadata.json")

    record = {
        "ticker": ticker_dir.name,
        "source": "Yahoo Finance via yfinance",
        "retrieved_at_utc": payload.get(
            "retrieved_at_utc",
            metadata.get("retrieved_at_utc"),
        ),
        "profile_available": bool(data),
    }

    for field in PROFILE_FIELDS:
        value = data.get(field)

        # These fields are expected to be scalar values.
        if isinstance(value, (dict, list, tuple, set)):
            value = json.dumps(value, default=str)

        record[field] = value

    # Preserve useful provider context without confusing it with
    # the financial statement period or publication date.
    record["provider_quote_type"] = data.get("quoteType")
    record["provider_exchange"] = data.get("exchange")
    record["provider_symbol"] = data.get("symbol")

    return record


def read_statement(
    ticker_dir: Path,
    statement_name: str,
    filename: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = ticker_dir / filename

    if not path.exists():
        return [], {
            "ticker": ticker_dir.name,
            "statement_type": statement_name,
            "file_available": False,
            "item_count": 0,
            "period_count": 0,
            "observation_count": 0,
            "latest_period_end": None,
            "error": "File missing",
        }

    try:
        frame = pd.read_csv(path, index_col=0)

        if frame.empty:
            return [], {
                "ticker": ticker_dir.name,
                "statement_type": statement_name,
                "file_available": True,
                "item_count": 0,
                "period_count": 0,
                "observation_count": 0,
                "latest_period_end": None,
                "error": None,
            }

        frame.index = frame.index.astype(str)
        frame.index.name = "financial_item"

        parsed_dates = pd.to_datetime(
            frame.columns,
            errors="coerce",
        )

        valid_positions = [
            index
            for index, date in enumerate(parsed_dates)
            if not pd.isna(date)
        ]

        observations = []

        for position in valid_positions:
            original_column = frame.columns[position]
            period_end = parsed_dates[position].date().isoformat()

            values = pd.to_numeric(
                frame.iloc[:, position],
                errors="coerce",
            )

            for financial_item, value in values.items():
                if pd.isna(value):
                    continue

                observations.append({
                    "ticker": ticker_dir.name,
                    "statement_type": statement_name,
                    "financial_item": str(financial_item),
                    "period_end": period_end,
                    "value": float(value),
                    "source": "Yahoo Finance via yfinance",
                })

        valid_dates = [
            parsed_dates[position] for position in valid_positions
        ]

        latest_period_end = (
            max(valid_dates).date().isoformat()
            if valid_dates
            else None
        )

        audit = {
            "ticker": ticker_dir.name,
            "statement_type": statement_name,
            "file_available": True,
            "item_count": int(len(frame.index)),
            "period_count": int(len(valid_positions)),
            "observation_count": int(len(observations)),
            "latest_period_end": latest_period_end,
            "error": None,
        }

        return observations, audit

    except Exception as exc:
        logger.warning(
            "Failed to process %s for %s: %s",
            statement_name,
            ticker_dir.name,
            exc,
        )

        return [], {
            "ticker": ticker_dir.name,
            "statement_type": statement_name,
            "file_available": True,
            "item_count": 0,
            "period_count": 0,
            "observation_count": 0,
            "latest_period_end": None,
            "error": str(exc),
        }


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw fundamentals directory does not exist: {RAW_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    ticker_dirs = sorted(
        path for path in RAW_ROOT.iterdir()
        if path.is_dir()
    )

    if not ticker_dirs:
        raise RuntimeError("No ticker directories found.")

    profiles = []
    observations = []
    statement_audits = []

    for index, ticker_dir in enumerate(ticker_dirs, start=1):
        logger.info(
            "Preparing %d/%d: %s",
            index,
            len(ticker_dirs),
            ticker_dir.name,
        )

        profiles.append(read_profile(ticker_dir))

        for statement_name, filename in STATEMENTS.items():
            rows, audit = read_statement(
                ticker_dir,
                statement_name,
                filename,
            )
            observations.extend(rows)
            statement_audits.append(audit)

    profile_df = pd.DataFrame(profiles)
    statement_df = pd.DataFrame(
        observations,
        columns=[
            "ticker",
            "statement_type",
            "financial_item",
            "period_end",
            "value",
            "source",
        ],
    )
    audit_df = pd.DataFrame(statement_audits)

    # Current profile fields are a snapshot. They are NOT assigned
    # historical financial period dates.
    profile_path = OUTPUT_ROOT / "yfinance_company_snapshot.parquet"
    statement_path = OUTPUT_ROOT / "yfinance_statement_observations.parquet"
    audit_path = REPORT_ROOT / "yfinance_statement_staging_audit.csv"

    profile_df.to_parquet(profile_path, index=False)
    statement_df.to_parquet(statement_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    # Detect duplicate statement keys for investigation.
    key_columns = [
        "ticker",
        "statement_type",
        "financial_item",
        "period_end",
    ]

    duplicate_count = (
        int(statement_df.duplicated(key_columns, keep=False).sum())
        if not statement_df.empty
        else 0
    )

    summary = {
        "ticker_directories": len(ticker_dirs),
        "profile_rows": len(profile_df),
        "statement_observations": len(statement_df),
        "statement_types": int(
            statement_df["statement_type"].nunique()
        ) if not statement_df.empty else 0,
        "unique_tickers_in_statements": int(
            statement_df["ticker"].nunique()
        ) if not statement_df.empty else 0,
        "duplicate_statement_key_rows": duplicate_count,
        "profile_output": str(profile_path.relative_to(ROOT)),
        "statement_output": str(statement_path.relative_to(ROOT)),
        "audit_output": str(audit_path.relative_to(ROOT)),
        "database_modified": False,
        "raw_data_modified": False,
        "models_modified": False,
    }

    print("\n" + "=" * 65)
    print("FUNDAMENTALS STAGING SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=2))

    print("\nStatement audit by type:")
    print(
        audit_df.groupby("statement_type").agg(
            companies=("ticker", "nunique"),
            periods=("period_count", "sum"),
            observations=("observation_count", "sum"),
            errors=("error", lambda col: col.notna().sum()),
        ).to_string()
    )

    print("\nProfile preview:")
    preview_columns = [
        column for column in [
            "ticker",
            "longName",
            "sector",
            "marketCap",
            "trailingPE",
            "priceToBook",
            "returnOnEquity",
            "debtToEquity",
        ]
        if column in profile_df.columns
    ]
    print(profile_df[preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
