
from pathlib import Path
from datetime import datetime, timezone
import json
import time

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "refetch_diagnostics"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "AUBANK.NS": "AU Small Finance Bank",
    "BANDHANBNK.NS": "Bandhan Bank",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
    )
}

SECTIONS = {
    "quarters": "Quarterly Results",
    "profit-loss": "Annual Profit and Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

def main():
    session = requests.Session()
    session.headers.update(HEADERS)
    results = []

    for ticker, company_name in TARGETS.items():
        slug = ticker.removesuffix(".NS")
        url = f"https://www.screener.in/company/{slug}/consolidated/"

        record = {
            "ticker": ticker,
            "company_name": company_name,
            "url": url,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            response = session.get(url, timeout=30)
            record["http_status"] = response.status_code
            record["final_url"] = response.url
            record["response_bytes"] = len(response.content)

            if response.status_code != 200:
                record["status"] = "HTTP_ERROR"
                record["error"] = f"HTTP {response.status_code}"
                results.append(record)
                print(ticker, record["status"])
                time.sleep(3)
                continue

            safe_name = ticker.replace(".", "_")
            html_path = OUT_DIR / f"{safe_name}_fresh.html"
            html_path.write_text(response.text, encoding="utf-8")
            record["saved_html"] = str(html_path.relative_to(ROOT))

            soup = BeautifulSoup(response.text, "html.parser")
            section_results = {}

            for section_id, section_name in SECTIONS.items():
                section = soup.find(id=section_id)

                if section is None:
                    section_results[section_id] = {
                        "name": section_name,
                        "found": False,
                    }
                    continue

                tables = []
                for index, table in enumerate(section.find_all("table")):
                    rows = []
                    for tr in table.find_all("tr"):
                        cells = [
                            " ".join(cell.stripped_strings)
                            for cell in tr.find_all(["th", "td"])
                        ]
                        if cells:
                            rows.append(cells)

                    tables.append({
                        "table_index": index,
                        "row_count": len(rows),
                        "max_cells_in_row": max(
                            (len(row) for row in rows), default=0
                        ),
                        "first_rows": rows[:5],
                    })

                section_results[section_id] = {
                    "name": section_name,
                    "found": True,
                    "tables": tables,
                }

            record["sections"] = section_results
            record["status"] = "FETCHED_AND_INSPECTED"

        except requests.RequestException as exc:
            record["status"] = "REQUEST_FAILED"
            record["error"] = str(exc)

        results.append(record)
        print(f"{ticker}: {record['status']}")
        time.sleep(3)

    output = OUT_DIR / "refetch_summary.json"
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSummary saved: {output}")
    print("Original raw data and HTML snapshots were not modified.")

if __name__ == "__main__":
    main()
