
from pathlib import Path
import re
import json
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "refetch_diagnostics"
)
OUT_DIR = INPUT_DIR / "dynamic_data_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "AUBANK.NS": "AUBANK_NS_fresh.html",
    "BANDHANBNK.NS": "BANDHANBNK_NS_fresh.html",
}

KEYWORDS = [
    "quarterly-results",
    "quarterly_results",
    "profit-loss",
    "balance-sheet",
    "cash-flow",
    "api/",
    "fetch(",
    "axios",
    "json",
    "Revenue +",
    "Equity Capital",
    "Cash from Operating Activity",
]

def main():
    summaries = []

    for ticker, filename in FILES.items():
        path = INPUT_DIR / filename
        if not path.exists():
            summaries.append({
                "ticker": ticker,
                "error": f"Missing saved HTML: {path}",
            })
            continue

        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        scripts = []
        for script in soup.find_all("script"):
            src = script.get("src")
            body = script.get_text(" ", strip=True)
            if src or body:
                scripts.append({
                    "src": src,
                    "type": script.get("type"),
                    "id": script.get("id"),
                    "inline_length": len(body),
                    "inline_preview": body[:500] if body else "",
                })

        matches = {}
        lower_html = html.lower()
        for keyword in KEYWORDS:
            positions = []
            start = 0
            needle = keyword.lower()

            while len(positions) < 5:
                pos = lower_html.find(needle, start)
                if pos < 0:
                    break
                positions.append({
                    "position": pos,
                    "context": html[max(0, pos - 180):pos + 320],
                })
                start = pos + len(needle)

            matches[keyword] = positions

        sections = {}
        for section_id in [
            "quarters", "profit-loss", "balance-sheet", "cash-flow"
        ]:
            section = soup.find(id=section_id)
            if section is not None:
                sections[section_id] = {
                    "data_attributes": dict(section.attrs),
                    "classes": section.get("class", []),
                    "html_length": len(str(section)),
                    "table_count": len(section.find_all("table")),
                    "buttons": [
                        {
                            "text": button.get_text(" ", strip=True),
                            "attrs": dict(button.attrs),
                        }
                        for button in section.find_all(
                            ["button", "a"], limit=10
                        )
                    ],
                }

        result = {
            "ticker": ticker,
            "html_bytes": len(html.encode("utf-8")),
            "script_count": len(scripts),
            "scripts": scripts,
            "section_details": sections,
            "keyword_matches": matches,
        }

        out = OUT_DIR / f"{ticker.replace('.', '_')}_inspection.json"
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries.append({
            "ticker": ticker,
            "script_count": len(scripts),
            "section_count": len(sections),
            "output": str(out.relative_to(ROOT)),
        })

    summary_path = OUT_DIR / "inspection_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("DYNAMIC-DATA INSPECTION COMPLETE")
    print(json.dumps(summaries, indent=2))
    print(f"Output folder: {OUT_DIR}")

if __name__ == "__main__":
    main()
