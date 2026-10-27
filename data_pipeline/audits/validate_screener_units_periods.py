
from pathlib import Path
from io import StringIO
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_pilot"
    / "consolidated_financials_long.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_pilot" / "unit_period_validation"
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

EPS_PATTERN = re.compile(r"\beps\b|earnings per share", re.I)
PERCENT_PATTERN = re.compile(
    r"%|margin|dividend payout|growth|cfo/ebitda",
    re.I,
)
DERIVED_PATTERN = re.compile(
    r"cfo/ebitda|share of operating profit converting into cash",
    re.I,
)


def metric_unit_class(metric):
    metric = str(metric).strip()

    if EPS_PATTERN.search(metric):
        return "INR_PER_SHARE"

    if DERIVED_PATTERN.search(metric):
        return "PERCENT_OR_RATIO_REQUIRES_REVIEW"

    if PERCENT_PATTERN.search(metric):
        return "PERCENT_AS_DISPLAYED"

    return "AMOUNT_UNIT_REQUIRES_SOURCE_CONFIRMATION"


def fetch_page_unit_evidence(session, company, slug):
    url = f"https://www.screener.in/company/{slug}/consolidated/"
    result = {
        "company": company,
        "url": url,
        "http_status": None,
        "final_url": None,
        "consolidated_path_confirmed": False,
        "sections_found": 0,
        "rupees_crores_text_found": False,
        "status": "FAILED",
        "error": None,
    }

    try:
        response = session.get(url, timeout=30)
        result["http_status"] = response.status_code
        result["final_url"] = response.url

        if response.status_code != 200:
            result["status"] = "HTTP_ERROR"
            return result

        path = requests.utils.urlparse(response.url).path.rstrip("/")
        result["consolidated_path_confirmed"] = (
            path.lower() == f"/company/{slug}/consolidated".lower()
        )

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(" ", strip=True).lower()

        result["rupees_crores_text_found"] = any(
            phrase in page_text
            for phrase in [
                "figures in rs. crores",
                "figures in rs crores",
                "figures in ₹ crores",
            ]
        )

        result["sections_found"] = sum(
            soup.find(id=section_id) is not None
            for section_id in SECTIONS
        )

        if (
            result["consolidated_path_confirmed"]
            and result["sections_found"] == len(SECTIONS)
            and result["rupees_crores_text_found"]
        ):
            result["status"] = "PAGE_EVIDENCE_FOUND"
        else:
            result["status"] = "PAGE_EVIDENCE_INCOMPLETE"

    except requests.RequestException as exc:
        result["error"] = str(exc)

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required = {
        "company", "reporting_basis", "section", "metric",
        "period_original", "period_end", "raw_value", "source_url",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["period_date"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["unit_class"] = df["metric"].apply(metric_unit_class)

    df["expected_period_semantics"] = df["section"].map({
        "Quarterly Results": "QUARTERLY_FLOW_OR_PER_SHARE",
        "Annual Profit and Loss": "ANNUAL_FLOW_OR_PER_SHARE",
        "Balance Sheet": "POINT_IN_TIME_BALANCE",
        "Cash Flow": "ANNUAL_CASH_FLOW_OR_DERIVED_METRIC",
    }).fillna("UNKNOWN_SECTION")

    df["basis_check"] = (
        df["reporting_basis"].astype(str).str.lower() == "consolidated"
    )

    # Date-pattern checks. These flag unusual dates; they do not
    # automatically reject records.
    df["period_check"] = "REVIEW"

    quarterly = df["section"].eq("Quarterly Results")
    annual_pl = df["section"].eq("Annual Profit and Loss")
    balance_sheet = df["section"].eq("Balance Sheet")
    cash_flow = df["section"].eq("Cash Flow")

    df.loc[df["period_date"].isna(), "period_check"] = "NON_DATE_PERIOD"

    df.loc[
        quarterly & df["period_date"].dt.month.isin([3, 6, 9, 12]),
        "period_check",
    ] = "QUARTER_END_MONTH"

    df.loc[
        annual_pl & df["period_date"].dt.month.eq(3),
        "period_check",
    ] = "MARCH_YEAR_END"

    df.loc[
        cash_flow & df["period_date"].dt.month.eq(3),
        "period_check",
    ] = "MARCH_YEAR_END_CASH_FLOW"

    df.loc[
        balance_sheet & df["period_date"].notna(),
        "period_check",
    ] = "POINT_IN_TIME_DATE_REQUIRES_REVIEW"

    # Save row-level audit with raw values retained.
    df.to_csv(OUTPUT_DIR / "unit_period_record_audit.csv", index=False)

    # Summarize date coverage by company and section.
    coverage = (
        df.groupby(["company", "section"], dropna=False)
        .agg(
            rows=("metric", "size"),
            dated_rows=("period_date", lambda x: x.notna().sum()),
            earliest_date=("period_date", "min"),
            latest_date=("period_date", "max"),
            distinct_periods=("period_date", "nunique"),
            consolidated_basis_ok=("basis_check", "all"),
        )
        .reset_index()
    )
    coverage.to_csv(OUTPUT_DIR / "period_coverage_audit.csv", index=False)

    # Source-page evidence check, one request per company.
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
        )
    })

    page_results = []
    for company, slug in COMPANIES.items():
        page_results.append(fetch_page_unit_evidence(session, company, slug))
        time.sleep(2)

    page_audit = pd.DataFrame(page_results)
    page_audit.to_csv(OUTPUT_DIR / "source_unit_evidence_audit.csv", index=False)

    summary = {
        "input_rows": len(df),
        "companies_in_data": int(df["company"].nunique()),
        "all_rows_tagged_consolidated": bool(df["basis_check"].all()),
        "rows_with_unparsed_dates": int(df["period_date"].isna().sum()),
        "unit_class_counts": df["unit_class"].value_counts().to_dict(),
        "period_check_counts": df["period_check"].value_counts().to_dict(),
        "source_pages_with_unit_text": int(
            page_audit["rupees_crores_text_found"].sum()
        ),
        "source_pages_with_confirmed_consolidated_path": int(
            page_audit["consolidated_path_confirmed"].sum()
        ),
        "note": (
            "Page text is evidence of displayed units, not independent "
            "verification of every extracted value. Confirm that unit "
            "text applies to each section before production use."
        ),
        "cash_flow_quarterly_vs_annual_semantics_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "UNIT_PERIOD_AUDIT_COMPLETE_REVIEW_REQUIRED",
    }

    (OUTPUT_DIR / "unit_period_validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nUNIT / PERIOD VALIDATION SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print("\nPERIOD COVERAGE")
    print(coverage.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
