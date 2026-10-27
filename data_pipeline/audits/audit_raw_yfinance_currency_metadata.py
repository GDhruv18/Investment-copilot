
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "fundamentals" / "yfinance"
OUTPUT_DIR = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
)

TARGET_TICKERS = {"HCLTECH.NS", "INFY.NS"}

CURRENCY_KEYS = {
    "currency",
    "financialcurrency",
    "financial_currency",
    "quote_currency",
    "quotecurrency",
    "financialcurrencycode",
}

METADATA_KEYS = CURRENCY_KEYS | {
    "symbol",
    "longname",
    "shortname",
    "exchange",
    "fullExchangeName".lower(),
    "quoteType".lower(),
}


def normalize_key(key):
    return str(key).replace("_", "").lower()


def extract_metadata(obj, found, prefix=""):
    """Recursively collect relevant metadata keys from nested JSON."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = normalize_key(key)
            if normalized in METADATA_KEYS:
                if not isinstance(value, (dict, list)):
                    found.append({
                        "key": str(key),
                        "path": f"{prefix}.{key}" if prefix else str(key),
                        "value": value,
                    })
            extract_metadata(
                value, found, f"{prefix}.{key}" if prefix else str(key)
            )
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            extract_metadata(value, found, f"{prefix}[{i}]")


def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw Yahoo directory not found: {RAW_ROOT}")

    records = []
    files_scanned = 0
    parse_errors = []

    for path in RAW_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue

        files_scanned += 1
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as exc:
            parse_errors.append({
                "file": str(path.relative_to(ROOT)),
                "error": str(exc),
            })
            continue

        # Infer ticker from the parent folder when possible.
        ticker = path.parent.name
        found = []
        extract_metadata(data, found)

        for item in found:
            records.append({
                "ticker_folder": ticker,
                "file": str(path.relative_to(ROOT)),
                **item,
            })

    columns = ["ticker_folder", "file", "key", "path", "value"]
    metadata = pd.DataFrame(records, columns=columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "raw_yfinance_currency_metadata.csv"
    metadata.to_csv(output_path, index=False)

    print("=" * 72)
    print("RAW YAHOO FINANCE CURRENCY METADATA AUDIT")
    print("=" * 72)
    print("Raw directory:", RAW_ROOT)
    print("JSON files scanned:", files_scanned)
    print("Metadata entries found:", len(metadata))
    print("Files with parse errors:", len(parse_errors))

    if metadata.empty:
        print("\nNo relevant metadata keys found in scanned JSON files.")
    else:
        currency_rows = metadata[
            metadata["key"].map(normalize_key).isin(CURRENCY_KEYS)
        ]

        print("\nCurrency-related keys found:")
        if currency_rows.empty:
            print("None")
        else:
            print(
                currency_rows.groupby(
                    ["key", "value"], dropna=False
                ).size().sort_values(ascending=False).to_string()
            )

        print("\nTarget tickers:")
        targets = metadata[
            metadata["ticker_folder"].isin(TARGET_TICKERS)
            | metadata["value"].astype(str).isin(TARGET_TICKERS)
        ]
        if targets.empty:
            print("No matching metadata entries found for target tickers.")
        else:
            print(targets.to_string(index=False, max_rows=100))

    print("\nOutput:", output_path.relative_to(ROOT))
    print("\nRead-only audit. No data modified and no units approved.")


if __name__ == "__main__":
    main()
