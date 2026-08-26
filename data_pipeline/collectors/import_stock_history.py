from pathlib import Path

import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_FOLDER = PROJECT_ROOT / "data" / "processed" / "stocks"


connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


files = list(STOCK_FOLDER.glob("*.parquet"))

print(f"Found {len(files)} stock files.")


total_inserted = 0


for index, file in enumerate(files, start=1):

    ticker = file.stem.replace("_", ".")

    print(f"\n[{index}/{len(files)}] Importing {ticker}...")

    try:

        data = pd.read_parquet(file)

        # Handle Yahoo Finance multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        for _, row in data.iterrows():

            date = pd.to_datetime(row["Date"]).date()

            open_price = row["Open"]
            high_price = row["High"]
            low_price = row["Low"]
            close_price = row["Close"]
            volume = row["Volume"]

            if pd.isna(open_price) or pd.isna(close_price):
                continue

            cursor.execute(
                """
                INSERT INTO stock_prices
                (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date) DO NOTHING;
                """,
                (
                    ticker,
                    date,
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    int(volume)
                )
            )

            if cursor.rowcount > 0:
                total_inserted += 1

        connection.commit()

        print(f"Imported {len(data)} rows.")

    except Exception as error:

        connection.rollback()

        print(f"FAILED: {ticker}")
        print(error)


cursor.close()
connection.close()

print("\n" + "=" * 50)
print("HISTORICAL IMPORT COMPLETED")
print("=" * 50)
print(f"New records inserted: {total_inserted}")
