import yfinance as yf
import json
import os

CONFIG_FILE = "data_pipeline/config/etfs_42_validated.json"
OUTPUT_DIR = "data/raw/etfs"

START_DATE = "2014-01-01"
END_DATE = "2026-09-20"


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

if isinstance(config, dict):
    tickers = config["etfs"]
else:
    tickers = config

print("=" * 70)
print("ETF HISTORICAL DATA COLLECTION")
print("=" * 70)
print(f"ETFs: {len(tickers)}")
print(f"Requested start: {START_DATE}")
print(f"Requested end:   {END_DATE}")
print("=" * 70)


success = 0
failed = 0

for ticker in tickers:

    print(f"\nDownloading: {ticker}")

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df.empty:
            print("  ❌ No data returned")
            failed += 1
            continue

        # Handle Yahoo MultiIndex columns
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            print(f"  ❌ Missing columns: {missing}")
            failed += 1
            continue

        df = df[required].dropna()

        if df.empty:
            print("  ❌ Empty after cleaning")
            failed += 1
            continue

        df.index.name = "Date"

        filename = ticker.replace(".", "_") + ".parquet"
        filepath = os.path.join(OUTPUT_DIR, filename)

        df.to_parquet(filepath)

        print(f"  ✓ Rows:  {len(df)}")
        print(f"  ✓ First: {df.index.min().date()}")
        print(f"  ✓ Last:  {df.index.max().date()}")

        success += 1

    except Exception as e:
        print(f"  ❌ Error: {e}")
        failed += 1


print("\n" + "=" * 70)
print("ETF COLLECTION COMPLETE")
print("=" * 70)
print(f"Successful: {success}")
print(f"Failed:     {failed}")
print("=" * 70)