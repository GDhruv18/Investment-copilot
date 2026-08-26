import json
from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "stocks_100.json"
RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"

RAW_FOLDER.mkdir(parents=True, exist_ok=True)


# Read the stock list
with open(CONFIG_FILE, "r") as file:
    config = json.load(file)

stocks = config["stocks"]

print(f"Found {len(stocks)} stocks.")


successful = 0
failed = []


for index, ticker in enumerate(stocks, start=1):

    print(f"\n[{index}/{len(stocks)}] Downloading {ticker}...")

    try:

        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print(f"NO DATA: {ticker}")
            failed.append(ticker)
            continue

        output_file = RAW_FOLDER / f"{ticker.replace('.', '_')}.parquet"

        data.to_parquet(output_file)

        print(f"Saved {len(data)} rows.")

        successful += 1

    except Exception as error:

        print(f"FAILED: {ticker}")
        print(error)

        failed.append(ticker)


print("\n" + "=" * 50)
print("DOWNLOAD COMPLETED")
print("=" * 50)

print(f"Successful: {successful}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed stocks:")
    for ticker in failed:
        print(ticker)