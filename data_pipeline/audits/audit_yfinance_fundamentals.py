
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent

RAW_ROOT = ROOT / "data" / "raw" / "fundamentals" / "yfinance"
REPORT_ROOT = ROOT / "data" / "processed" / "fundamentals"

INFO_FIELDS = [
    "longName",
    "sector",
    "industry",
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
]

STATEMENTS = [
    "annual_income_statement",
    "quarterly_income_statement",
    "annual_balance_sheet",
    "quarterly_balance_sheet",
    "annual_cash_flow",
    "quarterly_cash_flow",
]


def read_info(ticker_dir: Path) -> dict:
    path = ticker_dir / "info.json"

    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("data", {})
    except (OSError, json.JSONDecodeError):
        return {}


def statement_summary(ticker_dir: Path, name: str) -> dict:
    path = ticker_dir / f"{name}.csv"

    if not path.exists():
        return {
            f"{name}_file_exists": False,
            f"{name}_rows": 0,
            f"{name}_periods": 0,
            f"{name}_latest_period_end": None,
        }

    try:
        df = pd.read_csv(path, index_col=0)
        df = df.dropna(how="all")

        periods = pd.to_datetime(
            df.columns,
            errors="coerce",
        ).dropna()

        latest_period = (
            periods.max().date().isoformat()
            if len(periods)
            else None
        )

        return {
            f"{name}_file_exists": True,
            f"{name}_rows": len(df),
            f"{name}_periods": len(df.columns),
            f"{name}_latest_period_end": latest_period,
        }

    except Exception as exc:
        return {
            f"{name}_file_exists": True,
            f"{name}_rows": 0,
            f"{name}_periods": 0,
            f"{name}_latest_period_end": None,
            f"{name}_read_error": str(exc),
        }


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_ROOT}"
        )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    ticker_dirs = sorted(
        path for path in RAW_ROOT.iterdir()
        if path.is_dir()
    )

    rows = []

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        info = read_info(ticker_dir)

        row = {
            "ticker": ticker,
            "info_json_available": bool(info),
            "info_field_count": len(info),
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

        for field in INFO_FIELDS:
            value = info.get(field)

            row[f"{field}_available"] = (
                value is not None
                and not (
                    isinstance(value, float)
                    and pd.isna(value)
                )
            )

            # Retain selected values in the audit for inspection.
            if field in {
                "marketCap",
                "trailingPE",
                "forwardPE",
                "priceToBook",
                "returnOnEquity",
                "debtToEquity",
                "totalRevenue",
                "netIncomeToCommon",
                "totalDebt",
            }:
                row[f"info_{field}"] = value

        for statement in STATEMENTS:
            row.update(
                statement_summary(ticker_dir, statement)
            )

        metadata_path = ticker_dir / "collection_metadata.json"

        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                row["collection_status"] = metadata.get("status")
                row["retrieved_at_utc"] = metadata.get(
                    "retrieved_at_utc"
                )
                row["statement_error_count"] = len(
                    metadata.get("statement_errors", [])
                )
            except (OSError, json.JSONDecodeError):
                row["collection_status"] = "metadata_read_error"
                row["statement_error_count"] = None
        else:
            row["collection_status"] = "metadata_missing"
            row["statement_error_count"] = None

        rows.append(row)

    report = pd.DataFrame(rows)

    output_csv = REPORT_ROOT / "yfinance_fundamentals_field_audit.csv"
    report.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("YAHOO FINANCE FIELD-LEVEL AUDIT")
    print("=" * 70)
    print(f"Ticker directories: {len(report)}")
    print(f"Report: {output_csv.relative_to(ROOT)}")

    if report.empty:
        print("No ticker directories found.")
        return

    print("\nCollection status:")
    print(report["collection_status"].value_counts(dropna=False))

    print("\nInfo field coverage:")
    for field in INFO_FIELDS:
        column = f"{field}_available"
        if column in report.columns:
            available = int(report[column].sum())
            print(
                f"{field:25s} {available:3d}/{len(report)} "
                f"({available / len(report):6.1%})"
            )

    print("\nStatement availability:")
    for statement in STATEMENTS:
        column = f"{statement}_periods"
        available = int((report[column] > 0).sum())
        print(
            f"{statement:30s} "
            f"{available:3d}/{len(report)} companies"
        )

    print("\nLatest annual income-statement periods:")
    column = "annual_income_statement_latest_period_end"
    if column in report.columns:
        print(
            report[column]
            .fillna("Unavailable")
            .value_counts()
            .head(15)
            .to_string()
        )

    print("\nFirst 10 audit rows:")
    print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
