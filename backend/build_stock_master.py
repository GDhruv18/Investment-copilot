
import json
import yfinance as yf

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT / "data_pipeline" / "config" / "stocks_100.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "backend" / "app" / "data" / "stock_master.json"
)


def main():
    with open(SOURCE_FILE, "r", encoding="utf-8") as file:
        source = json.load(file)

    stock_master = []

    for yahoo_ticker in source["stocks"]:
        print(f"Fetching: {yahoo_ticker}")

        try:
            info = yf.Ticker(yahoo_ticker).get_info()

            company_name = info.get("longName") or info.get("shortName")

            if not company_name:
                print(f"Skipped: No company name for {yahoo_ticker}")
                continue

            internal_ticker = yahoo_ticker.replace(".", "_")

            stock_master.append({
                "company_name": company_name,
                "display_name": company_name,
                "internal_ticker": internal_ticker,
                "yahoo_ticker": yahoo_ticker,
                "exchange": "NSE",
            })

        except Exception as error:
            print(f"Failed: {yahoo_ticker} - {error}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {"stocks": stock_master},
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nSuccessfully collected {len(stock_master)} companies.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
