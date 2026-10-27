
from pathlib import Path
import json
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
    / "screener_consolidated_validation"
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
    "profit-loss": "Profit & Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
    )
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []

    for company, slug in COMPANIES.items():
        url = f"https://www.screener.in/company/{slug}/consolidated/"

        row = {
            "company": company,
            "slug": slug,
            "requested_url": url,
            "final_url": None,
            "http_status": None,
            "page_title": None,
            "consolidated_path_confirmed": False,
            "quarterly_section_found": False,
            "annual_pl_section_found": False,
            "balance_sheet_section_found": False,
            "cash_flow_section_found": False,
            "financial_unit_text_found": False,
            "status": "NOT_CHECKED",
            "error": None,
        }

        try:
            response = session.get(url, timeout=30)
            row["http_status"] = response.status_code
            row["final_url"] = response.url

            if response.status_code != 200:
                row["status"] = "HTTP_ERROR"
                results.append(row)
                print(f"{company}: HTTP {response.status_code}")
                time.sleep(2)
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            if soup.title:
                row["page_title"] = soup.title.get_text(
                    " ", strip=True
                )

            final_path = requests.utils.urlparse(
                response.url
            ).path.rstrip("/")

            expected_path = (
                f"/company/{slug}/consolidated"
            ).lower()

            row["consolidated_path_confirmed"] = (
                final_path.lower() == expected_path
            )

            for section_id, label in SECTIONS.items():
                section = soup.find(id=section_id)
                found = section is not None

                if section_id == "quarters":
                    row["quarterly_section_found"] = found
                elif section_id == "profit-loss":
                    row["annual_pl_section_found"] = found
                elif section_id == "balance-sheet":
                    row["balance_sheet_section_found"] = found
                elif section_id == "cash-flow":
                    row["cash_flow_section_found"] = found

            page_text = soup.get_text(" ", strip=True).lower()

            row["financial_unit_text_found"] = (
                "figures in rs. crores" in page_text
                or "figures in rs crores" in page_text
                or "figures in ₹ crores" in page_text
            )

            all_sections_found = all([
                row["quarterly_section_found"],
                row["annual_pl_section_found"],
                row["balance_sheet_section_found"],
                row["cash_flow_section_found"],
            ])

            if (
                row["consolidated_path_confirmed"]
                and all_sections_found
            ):
                row["status"] = (
                    "CONSOLIDATED_URL_AND_SECTIONS_CONFIRMED"
                )
            elif not row["consolidated_path_confirmed"]:
                row["status"] = "CONSOLIDATED_PATH_NOT_CONFIRMED"
            else:
                row["status"] = "ONE_OR_MORE_SECTIONS_MISSING"

            print(
                f"{company}: {row['status']} | "
                f"unit text found={row['financial_unit_text_found']}"
            )

        except requests.RequestException as exc:
            row["status"] = "REQUEST_FAILED"
            row["error"] = str(exc)
            print(f"{company}: request failed: {exc}")

        results.append(row)

        # Conservative delay between requests.
        time.sleep(2)

    report = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "consolidated_pilot_validation.csv"
    json_path = OUTPUT_DIR / "consolidated_pilot_validation.json"

    report.to_csv(csv_path, index=False)

    summary = {
        "companies_tested": len(results),
        "consolidated_paths_confirmed": int(
            report["consolidated_path_confirmed"].sum()
        ),
        "all_four_sections_found": int(
            report[
                [
                    "quarterly_section_found",
                    "annual_pl_section_found",
                    "balance_sheet_section_found",
                    "cash_flow_section_found",
                ]
            ].all(axis=1).sum()
        ),
        "financial_unit_text_found": int(
            report["financial_unit_text_found"].sum()
        ),
        "results": results,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nValidation summary:")
    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key != "results"
    }, indent=2))

    print(f"\nCSV report: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
