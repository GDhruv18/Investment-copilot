import yfinance as yf
import os

TICKERS = [
    "MPHASIS.NS",
    "EICHERMOT.NS",
    "CIPLA.NS",
    "TORNTPHARM.NS"
]

START_DATE = "2014-01-01"
END_DATE = "2026-09-20"

OUTPUT_DIR = "data/raw/stocks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("REPLACING OLD STOCK FILES")
print("=" * 70)

for ticker in TICKERS:
    print(f"\nDownloading: {ticker}")

    try:
        stock = yf.Ticker(ticker)

        df = stock.history(
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False
        )

        if df.empty:
            print("  ❌ No data")
            continue

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if "Adj Close" not in df.columns:
            print("  ❌ Adj Close missing")
            continue

        df = df[[
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]]

        df = df.dropna()

        df.index.name = "Date"

        filename = ticker.replace(".", "_") + ".parquet"
        filepath = os.path.join(OUTPUT_DIR, filename)

        df.to_parquet(filepath)

        print(f"  ✓ Saved: {filepath}")
        print(f"  Rows: {len(df)}")
        print(f"  First: {df.index.min().date()}")
        print(f"  Last:  {df.index.max().date()}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)