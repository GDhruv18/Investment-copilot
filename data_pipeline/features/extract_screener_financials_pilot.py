
from __future__ import annotations

import json
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_pilot"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "INFOSYS": "INFY",
    "HDFC_BANK": "HDFCBANK",
    "ICICI_BANK": "ICICIBANK",
}

# These are the actual section IDs observed on Screener pages.
SECTIONS = {
    "quarters": "Quarterly Results",
    "profit-loss": "Annual Profit and Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

BASE_URL = "https://www.screener.in"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
    )
})


def clean_column_name(column) -> str:
    """Convert a possibly multi-level table heading to text."""

    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if str(part).strip()
            and not str(part).startswith("Unnamed:")
        ]
        return " ".join(parts)

    return str(column).strip()


def clean_cell(value) -> str:
    """Preserve financial values as text without changing units."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def extract_section_tables(
    html: str,
    company: str,
    section_id: str,
    section_name: str,
) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    section = soup.find(id=section_id)

    if section is None:
        return []

    extracted = []

    for table_index, html_table in enumerate(
        section.find_all("table")
    ):
        try:
            frames = pd.read_html(
                StringIO(str(html_table)),
                displayed_only=False,
            )
        except ValueError:
            continue

        if not frames:
            continue

        df = frames[0].copy()
        df.columns = [
            clean_column_name(column)
            for column in df.columns
        ]

        # Save a wide copy so the original table layout
        # remains available for inspection.
        wide_path = (
            OUTPUT_DIR
            / f"{company}_{section_id}_table_{table_index}.csv"
        )

        df.to_csv(wide_path, index=False)

        # Convert the table to long format:
        # company, section, metric, period, raw value.
        if len(df.columns) < 2:
            continue

        metric_column = df.columns[0]

        for _, row in df.iterrows():
            metric = clean_cell(row[metric_column])

            if not metric:
                continue

            for period_column in df.columns[1:]:
                period = clean_column_name(period_column)
                value = clean_cell(row[period_column])

                if not period or not value:
                    continue

                extracted.append({
                    "company": company,
                    "screener_slug": COMPANIES[company],
                    "section": section_name,
                    "section_id": section_id,
                    "metric": metric,
                    "period": period,
                    "raw_value": value,
                    "source_url": (
                        f"{BASE_URL}/company/"
                        f"{COMPANIES[company]}/"
                    ),
                    "source": "Screener.in",
                    "unit": "As displayed by Screener; verify",
                    "table_index": table_index,
                })

    return extracted


def main() -> None:
    all_rows = []
    audit_rows = []

    for company, slug in COMPANIES.items():
        url = f"{BASE_URL}/company/{slug}/"

        print(f"\nCollecting {company} ...")

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:
            print(f"Request failed for {company}: {exc}")
            audit_rows.append({
                "company": company,
                "section": "ALL",
                "tables_found": 0,
                "rows_extracted": 0,
                "status": f"REQUEST_ERROR: {exc}",
            })
            continue

        for section_id, section_name in SECTIONS.items():
            section_rows = extract_section_tables(
                html,
                company,
                section_id,
                section_name,
            )

            all_rows.extend(section_rows)

            distinct_metrics = len({
                row["metric"] for row in section_rows
            })

            distinct_periods = len({
                row["period"] for row in section_rows
            })

            status = (
                "EXTRACTED"
                if section_rows
                else "SECTION_OR_TABLE_NOT_FOUND"
            )

            audit_rows.append({
                "company": company,
                "section": section_name,
                "rows_extracted": len(section_rows),
                "distinct_metrics": distinct_metrics,
                "distinct_period_labels": distinct_periods,
                "status": status,
            })

            print(
                f"  {section_name}: "
                f"{len(section_rows)} values | "
                f"{distinct_metrics} metrics | "
                f"{distinct_periods} period labels"
            )

        # Conservative pacing between company pages.
        time.sleep(2)

    # Write long-format output.
    if all_rows:
        long_df = pd.DataFrame(all_rows)

        long_path = OUTPUT_DIR / "screener_financials_pilot_long.csv"
        long_df.to_csv(long_path, index=False)

        print("\nExtracted values:", len(long_df))
        print("Companies:", long_df["company"].nunique())
        print("Long-format CSV:", long_path)
    else:
        print("\nNo financial values were extracted.")

    # Write audit even if extraction was incomplete.
    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_DIR / "screener_financials_pilot_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    json_path = OUTPUT_DIR / "screener_financials_pilot_audit.json"
    json_path.write_text(
        json.dumps(
            audit_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("PILOT EXTRACTION COMPLETE")
    print("=" * 55)
    print("Audit CSV:", audit_path)
    print("Audit JSON:", json_path)
    print("Output folder:", OUTPUT_DIR)

    if not audit_df.empty:
        print("\nSection audit:")
        print(audit_df.to_string(index=False))


if __name__ == "__main__":
    main()
