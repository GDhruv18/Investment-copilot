import json
from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETF_LIST_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "etfs_50.json"


with open(ETF_LIST_FILE, "r") as file:
    etfs = json.load(file)["etfs"]


print(f"Testing {len(etfs)} ETFs...\n")

valid = []
failed = []


for index, ticker in enumerate(etfs, start=1):

    print(f"[{index}/{len(etfs)}] Testing {ticker}...")

    try:

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print("   FAILED - No data")
            failed.append(ticker)
        else:
            print(f"   VALID - {len(data)} rows")
            valid.append(ticker)

    except Exception as error:

        print(f"   FAILED - {error}")
        failed.append(ticker)


print("\n" + "=" * 50)
print("ETF TEST COMPLETED")
print("=" * 50)

print(f"Valid ETFs: {len(valid)}")
print(f"Failed ETFs: {len(failed)}")

if failed:
    print("\nFailed ETFs:")
    for ticker in failed:
        print(ticker)