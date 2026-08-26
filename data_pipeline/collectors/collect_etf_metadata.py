import json
from pathlib import Path

import yfinance as yf
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETF_LIST_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "etfs_42_validated.json"
)


with open(ETF_LIST_FILE, "r") as file:
    etfs = json.load(file)["etfs"]


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


print(f"Collecting metadata for {len(etfs)} ETFs...\n")


successful = 0
failed = []


for index, ticker in enumerate(etfs, start=1):

    print(f"[{index}/{len(etfs)}] {ticker}")

    try:

        fund = yf.Ticker(ticker)

        info = fund.info

        etf_name = (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )

        issuer = info.get("fundFamily")

        underlying_index = (
            info.get("indexName")
            or info.get("category")
        )

        expense_ratio = (
            info.get("annualReportExpenseRatio")
        )

        aum = (
            info.get("totalAssets")
        )

        tracking_error = None


        cursor.execute(
            """
            INSERT INTO etfs
            (
                ticker,
                etf_name,
                issuer,
                underlying_index,
                expense_ratio,
                aum,
                tracking_error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker)
            DO UPDATE SET
                etf_name = EXCLUDED.etf_name,
                issuer = EXCLUDED.issuer,
                underlying_index = EXCLUDED.underlying_index,
                expense_ratio = EXCLUDED.expense_ratio,
                aum = EXCLUDED.aum,
                tracking_error = EXCLUDED.tracking_error;
            """,
            (
                ticker,
                etf_name,
                issuer,
                underlying_index,
                expense_ratio,
                aum,
                tracking_error
            )
        )

        connection.commit()

        successful += 1

        print(f"   Saved: {etf_name}")


    except Exception as error:

        connection.rollback()

        failed.append(ticker)

        print(f"   FAILED: {error}")


cursor.close()
connection.close()


print("\n" + "=" * 50)
print("ETF METADATA COLLECTION COMPLETED")
print("=" * 50)

print(f"Successful: {successful}")
print(f"Failed: {len(failed)}")

if failed:

    print("\nFailed ETFs:")

    for ticker in failed:
        print(ticker)