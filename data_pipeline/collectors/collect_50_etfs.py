import json
from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETF_LIST_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "etfs_50_validated.json"
)

RAW_DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "etfs"

RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# Read ETF list
with open(ETF_LIST_FILE, "r") as file:
    etfs = json.load(file)["etfs"]


print(f"Found {len(etfs)} ETFs to download.\n")


successful = []
failed = []


for index, ticker in enumerate(etfs, start=1):

    print("=" * 60)
    print(f"[{index}/{len(etfs)}] Downloading {ticker}...")

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

        filename = ticker.replace(".", "_") + ".parquet"

        output_path = RAW_DATA_FOLDER / filename

        data.to_parquet(output_path)

        print(f"Rows: {len(data)}")
        print(f"Saved: {output_path}")

        successful.append(ticker)

    except Exception as error:

        print(f"FAILED: {ticker}")
        print(error)

        failed.append(ticker)


print("\n" + "=" * 60)
print("ETF DOWNLOAD COMPLETED")
print("=" * 60)

print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if failed:

    print("\nFailed ETFs:")

    for ticker in failed:
        print(ticker)