import json
from pathlib import Path

import yfinance as yf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_LIST_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "stocks.json"
RAW_DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"


# Create the output folder if it doesn't exist
RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# Read stock list
with open(STOCK_LIST_FILE, "r") as file:
    stocks = json.load(file)["stocks"]


# Download data
for ticker in stocks:
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(f"No data found for {ticker}")
        continue

    # Save as Parquet
    filename = ticker.replace(".", "_") + ".parquet"
    output_path = RAW_DATA_FOLDER / filename

    data.to_parquet(output_path)

    print(f"Saved: {output_path}")


print("\nStock data collection completed!")