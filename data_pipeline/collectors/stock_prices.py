import json
from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_LIST_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "stocks_100.json"
)

RAW_DATA_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stocks"
)

RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2014-01-01"


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

with open(STOCK_LIST_FILE, "r") as file:
    stocks = json.load(file)["stocks"]

print("=" * 70)
print("STOCK HISTORICAL DATA COLLECTION")
print("=" * 70)
print(f"Stocks: {len(stocks)}")
print(f"Start date: {START_DATE}")
print("End date: present")
print("=" * 70)


# ============================================================
# DOWNLOAD DATA
# ============================================================

successful = 0
failed = 0

for index, ticker in enumerate(stocks, start=1):

    print(f"\n[{index}/{len(stocks)}] Downloading {ticker}...")

    try:

        data = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print(f"NO DATA: {ticker}")
            failed += 1
            continue

        # ----------------------------------------------------
        # Flatten Yahoo Finance MultiIndex columns
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ----------------------------------------------------
        # Keep only the fields required by our raw schema
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            print(
                f"FAILED: {ticker} — "
                f"missing columns: {missing_columns}"
            )
            failed += 1
            continue

        data = data[required_columns].copy()

        # ----------------------------------------------------
        # Clean index
        # ----------------------------------------------------

        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"

        # ----------------------------------------------------
        # Remove completely invalid rows
        # ----------------------------------------------------

        data = data.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close"
            ]
        )

        if data.empty:
            print(f"NO VALID DATA: {ticker}")
            failed += 1
            continue

        # ----------------------------------------------------
        # Save Parquet
        # ----------------------------------------------------

        filename = ticker.replace(".", "_") + ".parquet"
        output_path = RAW_DATA_FOLDER / filename

        data.to_parquet(output_path)

        print(
            f"Saved: {filename} | "
            f"Rows: {len(data):,} | "
            f"{data.index.min().date()} → "
            f"{data.index.max().date()}"
        )

        successful += 1

    except Exception as error:

        failed += 1

        print(f"FAILED: {ticker}")
        print(f"Error: {error}")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STOCK COLLECTION COMPLETED")
print("=" * 70)
print(f"Total stocks: {len(stocks)}")
print(f"Successful:   {successful}")
print(f"Failed:       {failed}")
print("=" * 70)