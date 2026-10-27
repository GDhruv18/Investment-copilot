
from pathlib import Path
import re
import json
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

PAGES_DIR = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100" / "pages"
)
RAW_CSV = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100"
    / "consolidated_financials_long.csv"
)
OUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "problem_stock_inspection"
)

TARGETS = {
    "AUBANK.NS": ["au small finance bank", "aubank"],
    "BANDHANBNK.NS": ["bandhan bank", "bandhanbnk"],
}

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def main():
    if not PAGES_DIR.exists():
        raise FileNotFoundError(f"Missing page snapshots: {PAGES_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_files = list(PAGES_DIR.rglob("*.html")) + list(
        PAGES_DIR.rglob("*.htm")
    )

    if not html_files:
        raise FileNotFoundError(f"No HTML snapshots found in {PAGES_DIR}")

    table_rows = []
    matches = []

    for path in html_files:
        filename = path.name.lower()
        content = path.read_text(encoding="utf-8", errors="replace")
        content_lower = content.lower()

        matched_tickers = []
        for ticker, terms in TARGETS.items():
            if any(
                term in filename or term in content_lower
                for term in terms
            ):
                matched_tickers.append(ticker)

        if not matched_tickers:
            continue

        soup = BeautifulSoup(content, "html.parser")
        matches.append({
            "file": str(path.relative_to(ROOT)),
            "matched_tickers": ", ".join(matched_tickers),
            "tables_found": len(soup.find_all("table")),
            "file_bytes": path.stat().st_size,
        })

        for ticker in matched_tickers:
            for table_index, table in enumerate(soup.find_all("table")):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [
                        normalize(cell.get_text(" ", strip=True))
                        for cell in tr.find_all(["th", "td"])
                    ]
                    if cells:
                        rows.append(cells)

                if not rows:
                    continue

                table_rows.append({
                    "ticker": ticker,
                    "file": str(path.relative_to(ROOT)),
                    "table_index": table_index,
                    "row_count": len(rows),
                    "first_rows_json": json.dumps(
                        rows[:8], ensure_ascii=False
                    ),
                })

    pd.DataFrame(matches).to_csv(
        OUT_DIR / "matched_html_files.csv", index=False
    )
    pd.DataFrame(table_rows).to_csv(
        OUT_DIR / "html_table_inspection.csv", index=False
    )

    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV, low_memory=False)
        raw = raw[raw["ticker"].isin(TARGETS)]
        raw.to_csv(OUT_DIR / "existing_raw_rows.csv", index=False)

    print("INSPECTION COMPLETE")
    print(f"HTML files scanned: {len(html_files)}")
    print(f"Matched snapshots: {len(matches)}")
    print(f"Tables inspected: {len(table_rows)}")
    print(f"Outputs: {OUT_DIR}")

if __name__ == "__main__":
    main()
