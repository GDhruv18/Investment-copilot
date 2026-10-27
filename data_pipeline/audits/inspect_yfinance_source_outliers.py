
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/fundamentals/yfinance"
PROFILE_PATH = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_company_snapshot.parquet"
)
COMPARISON_PATH = (
    ROOT / "data/processed/fundamentals/"
    "profile_statement_comparison_all_stocks.csv"
)

# Inspect the largest discrepancies plus the extreme debt examples.
TICKERS = [
    "HCLTECH.NS",
    "HAL.NS",
    "SIEMENS.NS",
    "NATIONALUM.NS",
    "INDIGO.NS",
    "DELHIVERY.NS",
    "TVSMOTOR.NS",
    "ADANIGREEN.NS",
    "ASHOKLEY.NS",
]

FIELDS = [
    "symbol",
    "longName",
    "currency",
    "financialCurrency",
    "quoteType",
    "marketCap",
    "totalDebt",
    "totalRevenue",
    "netIncomeToCommon",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "debtToEquity",
    "lastFiscalYearEnd",
    "mostRecentQuarter",
    "lastSplitDate",
]


def main():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(PROFILE_PATH)

    profile = pd.read_parquet(PROFILE_PATH).set_index("ticker")

    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
    else:
        comparison = pd.DataFrame()

    for ticker in TICKERS:
        print("\n" + "=" * 78)
        print(ticker)
        print("=" * 78)

        info_path = RAW_DIR / ticker / "info.json"

        if not info_path.exists():
            print(f"Raw info file not found: {info_path}")
        else:
            try:
                with info_path.open("r", encoding="utf-8") as file:
                    info = json.load(file)

                print("\nOriginal raw info.json fields:")
                for field in FIELDS:
                    if field in info:
                        print(f"  {field}: {info[field]}")
                    else:
                        print(f"  {field}: <not provided>")

            except (json.JSONDecodeError, OSError) as exc:
                print(f"Could not read raw info: {exc}")

        if ticker in profile.index:
            print("\nClean profile values:")
            row = profile.loc[ticker]
            for field in FIELDS:
                if field in profile.columns:
                    print(f"  {field}: {row[field]}")

        if not comparison.empty:
            rows = comparison[comparison["ticker"].eq(ticker)]
            if not rows.empty:
                print("\nProfile versus statement comparison:")
                cols = [
                    "metric",
                    "profile_value",
                    "statement_value",
                    "statement_period_end",
                    "relative_difference_pct",
                ]
                print(rows[cols].to_string(index=False))

    print("\nInspection complete.")
    print("Source data modified: No")
    print("Database modified: No")
    print("Models modified: No")


if __name__ == "__main__":
    main()
