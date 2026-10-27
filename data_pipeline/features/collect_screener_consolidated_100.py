
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]

CONFIG_CANDIDATES = [
    ROOT / "data_pipeline" / "config" / "stocks_100.json",
    ROOT / "data_pipeline" / "config" / "stocks.json",
]

OVERRIDE_FILE = (
    ROOT / "data_pipeline" / "config"
    / "screener_slug_overrides.json"
)

RAW_DIR = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100"
)
OUTPUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100"
)
PAGES_DIR = RAW_DIR / "pages"

SECTIONS = {
    "quarters": "Quarterly Results",
    "profit-loss": "Annual Profit and Loss",
    "balance-sheet": "Balance Sheet",
    "cash-flow": "Cash Flow",
}

DELAY_SECONDS = 3
TIMEOUT_SECONDS = 30

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
    """Convert month-year headings to month-end; retain original label too."""
    value = clean_text(value)
    match = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{4})", value)

    if not match:
        return None

    try:
        parsed = pd.to_datetime(
            f"{match.group(1)} {match.group(2)}",
            format="%b %Y",
            errors="raise",
        )
        return (parsed + pd.offsets.MonthEnd(0)).date().isoformat()
    except (ValueError, TypeError):
        return None


def parse_numeric(value):
    """Parse a displayed number without scaling or converting its units."""
    value = clean_text(value)

    if value.lower() in {"", "-", "--", "n/a", "na", "nan", "none"}:
        return None

    value = value.replace(",", "").replace("₹", "")
    value = value.replace("Rs.", "").replace("Rs", "").strip()

    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip()

    value = value.replace("%", "").strip()

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def infer_unit(metric, period_end):
    """
    Screening classification only.
    Amount units still require verification against the source table.
    """
    metric_lower = metric.lower()

    if "%" in metric or "percentage" in metric_lower:
        return "percent_as_displayed"

    if "eps" in metric_lower or "per share" in metric_lower:
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


def find_config():
    for path in CONFIG_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find stocks_100.json or stocks.json in "
        "data_pipeline/config."
    )


def load_universe(path):
    """
    Accept common JSON layouts:
      ["RELIANCE.NS", ...]
      [{"ticker": "RELIANCE.NS", ...}, ...]
      {"stocks": [...]}
      {"RELIANCE.NS": {...}, ...}

    A record may include screener_slug or slug.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("stocks", "tickers", "universe", "data"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("ticker", key)
            else:
                item = {"ticker": key}
            items.append(item)
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Unsupported stock universe JSON structure.")

    universe = []
    seen = set()

    for item in items:
        if isinstance(item, str):
            ticker = item.strip()
            explicit_slug = None
        elif isinstance(item, dict):
            ticker = (
                item.get("ticker")
                or item.get("symbol")
                or item.get("yfinance_ticker")
                or item.get("stock")
            )
            explicit_slug = (
                item.get("screener_slug")
                or item.get("slug")
            )
            ticker = str(ticker).strip() if ticker else ""
        else:
            continue

        if not ticker:
            continue

        if ticker in seen:
            continue

        seen.add(ticker)
        universe.append({
            "ticker": ticker,
            "explicit_slug": (
                str(explicit_slug).strip()
                if explicit_slug else None
            ),
        })

    if not universe:
        raise ValueError(
            f"No tickers were found in {path}. "
            "Inspect the JSON structure before proceeding."
        )

    return universe


def load_overrides():
    if not OVERRIDE_FILE.exists():
        return {}

    with open(OVERRIDE_FILE, "r", encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            "screener_slug_overrides.json must be a JSON object "
            "mapping ticker to Screener slug."
        )

    return {
        str(ticker).strip(): str(slug).strip()
        for ticker, slug in overrides.items()
        if str(ticker).strip() and str(slug).strip()
    }


def derive_slug(ticker):
    """
    Candidate only, not a guarantee of the correct Screener slug.
    Explicit overrides should be used for symbols that differ.
    """
    slug = ticker.strip()

    if slug.upper().endswith(".NS"):
        slug = slug[:-3]
    elif slug.upper().endswith(".BO"):
        slug = slug[:-3]

    return slug


def extract_table_records(
    table,
    ticker,
    slug,
    section_id,
    section_name,
    source_url,
    table_index,
    collected_at,
):
    records = []

    try:
        frames = pd.read_html(StringIO(str(table)))
    except (ValueError, ImportError) as exc:
        print(
            f"  Warning: cannot parse table {table_index} "
            f"for {ticker}/{section_id}: {exc}"
        )
        return records

    if not frames:
        return records

    frame = frames[0]

    if frame.empty or len(frame.columns) < 2:
        return records

    frame.columns = [flatten_column(c) for c in frame.columns]
    metric_column = frame.columns[0]

    for _, row in frame.iterrows():
        metric = clean_text(row.get(metric_column, ""))

        if not metric or metric.lower() in {
            "particulars", "metric", "items", ""
        }:
            continue

        for period_label in frame.columns[1:]:
            period = clean_text(period_label)

            if not period or period.lower().startswith("unnamed:"):
                continue

            raw_value = clean_text(row.get(period_label, ""))

            if raw_value.lower() in {"", "-", "--", "nan", "none"}:
                continue

            period_end = parse_period(period)

            records.append({
                "ticker": ticker,
                "screener_slug": slug,
                "reporting_basis": "consolidated",
                "section": section_name,
                "section_id": section_id,
                "table_index": table_index,
                "metric": metric,
                "period_original": period,
                "period_end": period_end,
                "raw_value": raw_value,
                "numeric_value": parse_numeric(raw_value),
                "unit": infer_unit(metric, period_end),
                "source": "Screener.in",
                "source_url": source_url,
                "collected_at_utc": collected_at,
                "unit_verified": False,
                "financial_value_independently_verified": False,
            })

    return records


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    config_path = find_config()
    universe = load_universe(config_path)
    overrides = load_overrides()

    print(f"Universe config: {config_path}")
    print(f"Tickers requested: {len(universe)}")
    print(f"Explicit slug overrides: {len(overrides)}")

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records = []
    page_audit = []

    for index, item in enumerate(universe, start=1):
        ticker = item["ticker"]

        # Precedence: explicit override file, then config slug,
        # then a candidate slug derived from the ticker.
        if ticker in overrides:
            slug = overrides[ticker]
            slug_source = "OVERRIDE_FILE"
        elif item["explicit_slug"]:
            slug = item["explicit_slug"]
            slug_source = "UNIVERSE_CONFIG"
        else:
            slug = derive_slug(ticker)
            slug_source = "DERIVED_CANDIDATE"

        url = f"https://www.screener.in/company/{slug}/consolidated/"
        collected_at = datetime.now(timezone.utc).isoformat()

        audit = {
            "ticker": ticker,
            "screener_slug": slug,
            "slug_source": slug_source,
            "source_url": url,
            "http_status": None,
            "final_url": None,
            "sections_found": 0,
            "tables_found": 0,
            "records_extracted": 0,
            "status": "FAILED",
            "error": None,
            "collected_at_utc": collected_at,
        }

        print(f"\n[{index}/{len(universe)}] {ticker} -> {slug}")

        try:
            response = session.get(url, timeout=TIMEOUT_SECONDS)
            audit["http_status"] = response.status_code
            audit["final_url"] = response.url

            if response.status_code != 200:
                audit["status"] = "HTTP_ERROR"
                audit["error"] = f"HTTP {response.status_code}"
                page_audit.append(audit)
                print(f"  HTTP {response.status_code}; skipping.")
                time.sleep(DELAY_SECONDS)
                continue

            final_path = (
                requests.utils.urlparse(response.url).path.rstrip("/").lower()
            )
            expected_path = (
                f"/company/{slug}/consolidated".rstrip("/").lower()
            )

            if final_path != expected_path:
                audit["status"] = "FINAL_URL_MISMATCH"
                audit["error"] = (
                    f"Expected path {expected_path}; received {final_path}"
                )
                page_audit.append(audit)
                print("  URL mismatch; skipping page.")
                time.sleep(DELAY_SECONDS)
                continue

            # Save returned HTML for traceability and later debugging.
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
            html_path = PAGES_DIR / f"{safe_name}.html"
            html_path.write_text(response.text, encoding="utf-8")
            audit["saved_html"] = str(html_path)

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
                        ticker=ticker,
                        slug=slug,
                        section_id=section_id,
                        section_name=section_name,
                        source_url=response.url,
                        table_index=table_index,
                        collected_at=collected_at,
                    )
                    all_records.extend(records)
                    audit["records_extracted"] += len(records)

            if audit["sections_found"] == len(SECTIONS):
                audit["status"] = "EXTRACTED_ALL_SECTIONS"
            elif audit["sections_found"] > 0:
                audit["status"] = "PARTIAL_SECTIONS"
            else:
                audit["status"] = "NO_EXPECTED_SECTIONS"

            print(
                f"  Sections: {audit['sections_found']}/4 | "
                f"Tables: {audit['tables_found']} | "
                f"Records: {audit['records_extracted']} | "
                f"Status: {audit['status']}"
            )

        except requests.RequestException as exc:
            audit["error"] = str(exc)
            audit["status"] = "REQUEST_FAILED"
            print(f"  Request failed: {exc}")

        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            audit["status"] = "EXTRACTION_ERROR"
            print(f"  Extraction error: {exc}")

        page_audit.append(audit)

        # Conservative pacing; do not parallelize requests.
        time.sleep(DELAY_SECONDS)

    data = pd.DataFrame(all_records)
    audit_df = pd.DataFrame(page_audit)

    raw_data_path = RAW_DIR / "consolidated_financials_long.csv"
    audit_path = OUTPUT_DIR / "collection_audit.csv"
    summary_path = OUTPUT_DIR / "collection_summary.json"

    # The raw CSV preserves the displayed values without unit conversion.
    data.to_csv(raw_data_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    if data.empty:
        duplicate_rows = 0
        date_rows = 0
        non_date_rows = 0
        missing_numeric = 0
        section_summary = []
    else:
        duplicate_keys = [
            "ticker", "reporting_basis", "section_id",
            "table_index", "metric", "period_original"
        ]
        duplicate_rows = int(
            data.duplicated(subset=duplicate_keys, keep=False).sum()
        )
        date_rows = int(data["period_end"].notna().sum())
        non_date_rows = int(data["period_end"].isna().sum())
        missing_numeric = int(data["numeric_value"].isna().sum())

        section_summary = (
            data.groupby(["ticker", "section"], dropna=False)
            .agg(
                records=("metric", "size"),
                distinct_metrics=("metric", "nunique"),
                date_period_rows=("period_end", "count"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

    status_counts = (
        audit_df["status"].value_counts().to_dict()
        if not audit_df.empty else {}
    )

    summary = {
        "config_file": str(config_path),
        "companies_requested": len(universe),
        "companies_attempted": int(len(audit_df)),
        "companies_with_all_sections": int(
            (audit_df["status"] == "EXTRACTED_ALL_SECTIONS").sum()
        ) if not audit_df.empty else 0,
        "status_counts": status_counts,
        "total_records": int(len(data)),
        "date_period_rows": date_rows,
        "non_date_period_rows": non_date_rows,
        "missing_numeric_values": missing_numeric,
        "duplicate_key_rows": duplicate_rows,
        "reporting_basis": "consolidated",
        "unit_status": "UNVERIFIED; RAW DISPLAY VALUES PRESERVED",
        "financial_values_independently_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "raw_data_file": str(raw_data_path),
        "audit_file": str(audit_path),
        "section_summary": section_summary,
        "status": "COLLECTION_FINISHED_REVIEW_REQUIRED",
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nCOLLECTION SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nRaw records: {raw_data_path}")
    print(f"Audit: {audit_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
