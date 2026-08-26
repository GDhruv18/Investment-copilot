import json
from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETF_LIST_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "etfs_50.json"
OUTPUT_FILE = PROJECT_ROOT / "data_pipeline" / "config" / "etfs_50_validated.json"


# Read the original ETF list
with open(ETF_LIST_FILE, "r") as file:
    original_etfs = json.load(file)["etfs"]


# Additional ETFs that we successfully validated
additional_candidates = [
    "NIFTYIETF.NS",
    "NIFTYETF.NS",
    "NIFTY1.NS",

    "NIFTYQLITY.NS",
    "NIFTY100LOWVOL30.NS",
    "MOM30IETF.NS",
    "MOM100.NS",
    "ESG.NS",
    "BSE500IETF.NS",
    "MAFANG.NS",
    "MOMOMENTUM.NS",
    "NIFTY100EW.NS",
    "TATAGOLD.NS",
    "DYNAMIC.NS",
    "EQUAL50.NS",
    "MOMENTUM.NS",

    "NIFTY50EQUALWEIGHT.NS",

    "MONQ50.NS",
    "HNGSNGBEES.NS",
    "MOM50.NS"
]


# Combine everything and remove duplicates
all_candidates = list(dict.fromkeys(
    original_etfs + additional_candidates
))


print(f"Total unique candidates: {len(all_candidates)}")
print("\nValidating ETFs...\n")


valid_etfs = []


for index, ticker in enumerate(all_candidates, start=1):

    print(f"[{index}/{len(all_candidates)}] Testing {ticker}...")

    try:

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if not data.empty:

            print(f"   VALID ({len(data)} rows)")

            if ticker not in valid_etfs:
                valid_etfs.append(ticker)

        else:
            print("   FAILED")

    except Exception as error:

        print(f"   FAILED: {error}")


print("\n" + "=" * 60)
print(f"VALID UNIQUE ETFs FOUND: {len(valid_etfs)}")
print("=" * 60)


# We need exactly 50
if len(valid_etfs) < 50:

    print("\nERROR: Fewer than 50 valid ETFs were found.")
    print("We need more candidates before creating the final universe.")

else:

    final_etfs = valid_etfs[:50]

    output = {
        "etfs": final_etfs
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(output, file, indent=4)

    print("\nFinal ETF universe created.")
    print(f"ETFs selected: {len(final_etfs)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFinal ETFs:")

    for ticker in final_etfs:
        print(ticker)