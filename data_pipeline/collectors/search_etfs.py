import json
from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETF_LIST_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "etfs_50.json"

with open(ETF_LIST_FILE, "r") as file:
    existing_etfs = set(json.load(file)["etfs"])


candidates = [
    "MONQ50.NS",
    "HNGSNGBEES.NS",
    "MOM50.NS",
    "NIFTYBEES.NS",
    "JUNIORBEES.NS",
    "BANKBEES.NS",
    "ITBEES.NS",
    "PHARMABEES.NS",
    "AUTOBEES.NS",
    "GOLDBEES.NS",
    "SILVERBEES.NS"
]


new_valid = []


print(f"Current ETF list contains: {len(existing_etfs)} entries\n")

for ticker in candidates:

    if ticker in existing_etfs:
        print(f"ALREADY IN LIST: {ticker}")
        continue

    try:

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if not data.empty:
            print(f"NEW VALID: {ticker} ({len(data)} rows)")
            new_valid.append(ticker)
        else:
            print(f"FAILED: {ticker}")

    except Exception:
        print(f"FAILED: {ticker}")


print("\n" + "=" * 50)
print(f"NEW UNIQUE VALID ETFs: {len(new_valid)}")
print("=" * 50)

for ticker in new_valid:
    print(ticker)