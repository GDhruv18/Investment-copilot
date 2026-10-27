
from pathlib import Path
import json

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

PAGES_DIR = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100" / "pages"
)
OUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "parser_diagnostics"
)

TARGETS = ["AUBANK.NS", "BANDHANBNK.NS"]

SECTIONS = {
    "quarters": "Quarterly Results",
    "profit-loss": "Annual Profit and Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = []

    for ticker in TARGETS:
        html_path = PAGES_DIR / f"{ticker}.html"

        if not html_path.exists():
            diagnostics.append({
                "ticker": ticker,
                "section": "",
                "table_index": -1,
                "status": "SAVED_HTML_MISSING",
                "error": str(html_path),
            })
            continue

        soup = BeautifulSoup(
            html_path.read_text(encoding="utf-8", errors="replace"),
            "html.parser",
        )

        for section_id, section_name in SECTIONS.items():
            section = soup.find(id=section_id)

            if section is None:
                diagnostics.append({
                    "ticker": ticker,
                    "section": section_name,
                    "table_index": -1,
                    "status": "SECTION_NOT_FOUND",
                })
                continue

            for table_index, table in enumerate(
                section.find_all("table")
            ):
                dom_rows = []
                for tr in table.find_all("tr"):
                    cells = [
                        " ".join(cell.stripped_strings)
                        for cell in tr.find_all(["th", "td"])
                    ]
                    if cells:
                        dom_rows.append(cells)

                try:
                    frames = pd.read_html(str(table))
                    frame = frames[0] if frames else pd.DataFrame()
                    pandas_error = ""
                except Exception as exc:
                    frame = pd.DataFrame()
                    pandas_error = f"{type(exc).__name__}: {exc}"

                diagnostics.append({
                    "ticker": ticker,
                    "section": section_name,
                    "section_id": section_id,
                    "table_index": table_index,
                    "dom_row_count": len(dom_rows),
                    "dom_max_cells": max(
                        (len(row) for row in dom_rows), default=0
                    ),
                    "dom_first_rows": json.dumps(
                        dom_rows[:6], ensure_ascii=False
                    ),
                    "pandas_row_count": len(frame),
                    "pandas_column_count": len(frame.columns),
                    "pandas_columns": json.dumps(
                        [str(c) for c in frame.columns],
                        ensure_ascii=False,
                    ),
                    "pandas_first_rows": (
                        frame.head(5).to_json(
                            orient="records", force_ascii=False
                        )
                        if not frame.empty else "[]"
                    ),
                    "pandas_error": pandas_error,
                    "status": (
                        "PANDAS_EMPTY_OR_TOO_NARROW"
                        if frame.empty or len(frame.columns) < 2
                        else "PANDAS_TABLE_PARSED"
                    ),
                })

    result = pd.DataFrame(diagnostics)
    output = OUT_DIR / "two_stock_table_diagnostics.csv"
    result.to_csv(output, index=False)

    print("TARGETED DIAGNOSTIC COMPLETE")
    print(f"Tables/sections inspected: {len(result)}")
    print(result.groupby(
        ["ticker", "status"], dropna=False
    ).size().to_string())
    print(f"\nOutput: {output}")

if __name__ == "__main__":
    main()
