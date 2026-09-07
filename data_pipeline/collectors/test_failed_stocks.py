import yfinance as yf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"

RAW_FOLDER.mkdir(parents=True, exist_ok=True)


stocks = [
    "ABB.NS",
    "INDIAMART.NS"
]


successful = 0


for ticker in stocks:

    print("\n" + "=" * 50)
    print("Downloading:", ticker)

    try:

        data = yf.download(
            ticker,
            start="2018-01-01",
            end="2026-09-08",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            print("NO DATA:", ticker)
            continue

        output_file = RAW_FOLDER / f"{ticker.replace('.', '_')}.parquet"

        data.to_parquet(output_file)

        print("Rows:", len(data))
        print("First date:", data.index[0])
        print("Last date:", data.index[-1])
        print("Saved:", output_file)

        successful += 1

    except Exception as error:

        print("FAILED:", ticker)
        print(error)


print("\n" + "=" * 50)
print("RETRY DOWNLOAD COMPLETED")
print("=" * 50)
print("Successful:", successful)