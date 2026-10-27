
from pathlib import Path
from io import StringIO
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_consolidated_pilot"
)

COMPANIES = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "INFOSYS": "INFY",
    "HDFC_BANK": "HDFCBANK",
    "ICICI_BANK": "ICICIBANK",
}

SECTIONS = {
    "quarters": "Quarterly Results",
    "profit-loss": "Annual Profit and Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
    )
}


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_period(value):
    """Parse labels like 'Jun 2026'; preserve non-date labels separately."""
    value = clean_text(value)
    match = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{4})", value)

    if not match:
        return None

    try:
        period = pd.to_datetime(
            f"{match.group(1)} {match.group(2)}",
            format="%b %Y",
            errors="raise",
        )
        return (
            period + pd.offsets.MonthEnd(0)
        ).date().isoformat()
    except (ValueError, TypeError):
        return None


def parse_numeric(value):
    """Convert numeric-looking values without changing their scale."""
    value = clean_text(value)

    if value in {"", "-", "--", "N/A", "NA"}:
        return None

    value = value.replace(",", "").replace("₹", "")
    value = value.replace("Rs.", "").replace("Rs", "").strip()

    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip()

    value = value.replace("%", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


def infer_unit(metric, period_end):
    """
    Infer only common display units. This is a classification,
    not independent verification of the reported value.
    """
    metric_lower = metric.lower()

    if "%" in metric or "percentage" in metric_lower:
        return "percent_as_displayed"

    if "eps" in metric_lower:
        return "INR_per_share"

    if period_end is None:
        return "summary_metric_unit_unverified"

    return "INR_crore_assumed_from_page_display"


def flatten_column(value):
    if isinstance(value, tuple):
        parts = [
            clean_text(part)
            for part in value
            if clean_text(part)
            and not clean_text(part).startswith("Unnamed:")
        ]
        return " ".join(parts).strip()

    return clean_text(value)


def extract_table_records(
    table,
    company,
    slug,
    section_id,
    section_name,
    source_url,
    table_index,
):
    records = []

    try:
        frames = pd.read_html(StringIO(str(table)))
    except (ValueError, ImportError) as exc:
        print(
            f"  Warning: could not parse table {table_index} "
            f"for {company}/{section_id}: {exc}"
        )
        return records

    if not frames:
        return records

    frame = frames[0]

    if frame.empty or len(frame.columns) < 2:
        return records

    columns = [flatten_column(col) for col in frame.columns]
    frame.columns = columns

    # First column should contain the financial metric names.
    metric_column = frame.columns[0]

    for _, row in frame.iterrows():
        metric = clean_text(row.get(metric_column, ""))

        if not metric or metric.lower() in {
            "particulars", "metric", "items", "",
        }:
            continue

        for period_label in frame.columns[1:]:
            period = clean_text(period_label)

            if not period or period.lower().startswith("unnamed:"):
                continue

            raw_value = clean_text(row.get(period_label, ""))

            if raw_value in {"", "-", "--", "nan", "None"}:
                continue

            period_end = parse_period(period)

            records.append({
                "company": company,
                "screener_slug": slug,
                "reporting_basis": "consolidated",
                "section": section_name,
                "section_id": section_id,
                "metric": metric,
                "period_original": period,
                "period_end": period_end,
                "raw_value": raw_value,
                "numeric_value": parse_numeric(raw_value),
                "unit": infer_unit(metric, period_end),
                "source": "Screener.in",
                "source_url": source_url,
                "table_index": table_index,
            })

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records = []
    page_audit = []

    for company, slug in COMPANIES.items():
        url = f"https://www.screener.in/company/{slug}/consolidated/"

        print(f"\nCollecting consolidated data: {company}")

        audit = {
            "company": company,
            "slug": slug,
            "source_url": url,
            "http_status": None,
            "final_url": None,
            "sections_found": 0,
            "tables_found": 0,
            "records_extracted": 0,
            "status": "FAILED",
            "error": None,
        }

        try:
            response = session.get(url, timeout=30)
            audit["http_status"] = response.status_code
            audit["final_url"] = response.url

            if response.status_code != 200:
                audit["status"] = "HTTP_ERROR"
                page_audit.append(audit)
                print(f"  HTTP status: {response.status_code}")
                time.sleep(2)
                continue

            final_path = requests.utils.urlparse(
                response.url
            ).path.rstrip("/").lower()

            expected_path = (
                f"/company/{slug}/consolidated"
            ).lower()

            if final_path != expected_path:
                audit["status"] = "FINAL_URL_MISMATCH"
                audit["error"] = (
                    f"Expected {expected_path}; got {final_path}"
                )
                page_audit.append(audit)
                print("  Stopping: final URL does not match.")
                time.sleep(2)
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for section_id, section_name in SECTIONS.items():
                section = soup.find(id=section_id)

                if section is None:
                    print(f"  Missing section: {section_name}")
                    continue

                audit["sections_found"] += 1
                tables = section.find_all("table")
                audit["tables_found"] += len(tables)

                for table_index, table in enumerate(tables):
                    records = extract_table_records(
                        table=table,
                        company=company,
                        slug=slug,
                        section_id=section_id,
                        section_name=section_name,
                        source_url=response.url,
                        table_index=table_index,
                    )
                    all_records.extend(records)
                    audit["records_extracted"] += len(records)

            if audit["sections_found"] == len(SECTIONS):
                audit["status"] = "EXTRACTED_ALL_SECTIONS"
            else:
                audit["status"] = "PARTIAL_SECTIONS"

            print(
                f"  Sections: {audit['sections_found']}/"
                f"{len(SECTIONS)} | "
                f"Tables: {audit['tables_found']} | "
                f"Records: {audit['records_extracted']}"
            )

        except requests.RequestException as exc:
            audit["error"] = str(exc)
            audit["status"] = "REQUEST_FAILED"
            print(f"  Request failed: {exc}")

        page_audit.append(audit)

        # Conservative delay between company requests.
        time.sleep(2)

    data = pd.DataFrame(all_records)
    audit_df = pd.DataFrame(page_audit)

    data_path = OUTPUT_DIR / "consolidated_financials_long.csv"
    audit_path = OUTPUT_DIR / "consolidated_extraction_audit.csv"
    summary_path = OUTPUT_DIR / "consolidated_extraction_summary.json"

    data.to_csv(data_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    if data.empty:
        section_summary = []
        duplicate_rows = 0
        date_rows = 0
        non_date_rows = 0
        missing_numeric = 0
    else:
        section_summary = (
            data.groupby(
                ["company", "section"],
                dropna=False,
            )
            .agg(
                records=("metric", "size"),
                distinct_metrics=("metric", "nunique"),
                date_period_rows=("period_end", "count"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

        duplicate_rows = int(
            data.duplicated(
                subset=[
                    "company",
                    "reporting_basis",
                    "section_id",
                    "metric",
                    "period_original",
                    "table_index",
                ],
                keep=False,
            ).sum()
        )

        date_rows = int(data["period_end"].notna().sum())
        non_date_rows = int(data["period_end"].isna().sum())
        missing_numeric = int(data["numeric_value"].isna().sum())

    summary = {
        "companies_requested": len(COMPANIES),
        "companies_with_all_sections": int(
            (
                audit_df["status"]
                == "EXTRACTED_ALL_SECTIONS"
            ).sum()
        ) if not audit_df.empty else 0,
        "total_records": len(data),
        "date_period_rows": date_rows,
        "non_date_period_rows": non_date_rows,
        "missing_numeric_values": missing_numeric,
        "duplicate_key_rows": duplicate_rows,
        "reporting_basis": "consolidated",
        "unit_status": "INFERRED_FROM_DISPLAY; VALIDATION_REQUIRED",
        "financial_values_independently_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "section_summary": section_summary,
        "data_file": str(data_path),
        "audit_file": str(audit_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nExtraction summary:")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nData: {data_path}")
    print(f"Audit: {audit_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
