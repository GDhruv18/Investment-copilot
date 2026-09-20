import yfinance as yf

FAILED_TICKERS = [
    "IDFCFIRSTB.NS",
    "MPHASIS.NS",
    "EICHERMOT.NS",
    "CIPLA.NS",
    "TORNTPHARM.NS"
]

START_DATE = "2014-01-01"
END_DATE = "2026-09-20"


print("=" * 70)
print("FAILED STOCK HISTORY DIAGNOSTIC")
print("=" * 70)

for ticker in FAILED_TICKERS:
    print(f"\nTesting: {ticker}")

    try:
        stock = yf.Ticker(ticker)

        df = stock.history(
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False
        )

        if df.empty:
            print("  ❌ No data returned")
            continue

        print(f"  Rows: {len(df)}")
        print(f"  First date: {df.index.min().date()}")
        print(f"  Last date:  {df.index.max().date()}")

        required = ["Open", "High", "Low", "Close", "Volume"]

        missing_columns = [
            col for col in required
            if col not in df.columns
        ]

        if missing_columns:
            print(f"  ⚠ Missing columns: {missing_columns}")
        else:
            print("  ✓ Required columns present")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)