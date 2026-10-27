
from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_access_test"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "INFOSYS": "INFY",
    "HDFC_BANK": "HDFCBANK",
    "ICICI_BANK": "ICICIBANK",
}

BASE_URL = "https://www.screener.in"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
    )
})


def extract_tables(html: str, company: str) -> list[dict]:
    """Extract HTML tables while preserving their original values."""

    tables = pd.read_html(
        StringIO(html),  # FIX: parse HTML text, not a filename
        displayed_only=False,
    )

    extracted = []

    for index, table in enumerate(tables):
        csv_path = OUTPUT_DIR / f"{company}_table_{index}.csv"
        table.to_csv(csv_path, index=False)

        columns = [str(column) for column in table.columns]
        preview = (
            table.head(3)
            .fillna("")
            .astype(str)
            .values.tolist()
        )

        extracted.append({
            "table_index": index,
            "rows": len(table),
            "columns": columns,
            "csv_file": str(csv_path),
            "preview": preview,
        })

    return extracted


def inspect_company(company: str, slug: str) -> dict:
    url = f"{BASE_URL}/company/{slug}/"

    result = {
        "company": company,
        "url": url,
        "http_status": None,
        "page_title": None,
        "table_count": 0,
        "financial_tables": [],
        "error": None,
    }

    try:
        response = session.get(url, timeout=30)
        result["http_status"] = response.status_code
        response.raise_for_status()

        html = response.text

        # Save the HTML page for troubleshooting.
        (OUTPUT_DIR / f"{company.lower()}.html").write_text(
            html, encoding="utf-8"
        )

        soup = BeautifulSoup(html, "lxml")
        if soup.title:
            result["page_title"] = soup.title.get_text(
                " ", strip=True
            )

        tables = extract_tables(html, company)
        result["table_count"] = len(tables)

        # Identify tables using their row labels and date columns.
        for table_info in tables:
            preview_text = json.dumps(
                table_info["preview"], ensure_ascii=False
            ).lower()
            columns_text = " ".join(
                table_info["columns"]
            ).lower()
            combined = preview_text + " " + columns_text

            if any(term in combined for term in (
                "sales", "revenue", "expenses",
                "net profit", "profit before tax",
                "mar 2025", "jun 2025", "sep 2025",
            )):
                result["financial_tables"].append({
                    "table_index": table_info["table_index"],
                    "rows": table_info["rows"],
                    "columns": table_info["columns"],
                    "csv_file": table_info["csv_file"],
                    "preview": table_info["preview"],
                })

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def main() -> None:
    results = []

    for company, slug in COMPANIES.items():
        print(f"\nChecking {company} ...")

        result = inspect_company(company, slug)
        results.append(result)

        print("HTTP status:", result["http_status"])
        print("Page title:", result["page_title"])
        print("Tables extracted:", result["table_count"])
        print(
            "Potential financial tables:",
            len(result["financial_tables"]),
        )

        if result["error"]:
            print("ERROR:", result["error"])

        time.sleep(2)

    json_path = OUTPUT_DIR / "access_test_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = pd.DataFrame([
        {
            "company": r["company"],
            "url": r["url"],
            "http_status": r["http_status"],
            "page_title": r["page_title"],
            "table_count": r["table_count"],
            "potential_financial_tables": len(
                r["financial_tables"]
            ),
            "error": r["error"],
        }
        for r in results
    ])

    summary_path = OUTPUT_DIR / "access_test_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 55)
    print("SCREENER TEST FINISHED")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("\nSummary CSV:", summary_path)
    print("Detailed JSON:", json_path)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
