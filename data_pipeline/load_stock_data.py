from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_FOLDER = PROJECT_ROOT / "data" / "processed" / "stocks"


connection = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


for file in STOCK_FOLDER.glob("*.parquet"):

    ticker = file.stem.replace("_", ".")

    print(f"Loading {ticker}...")

    data = pd.read_parquet(file)

    for index, row in data.iterrows():

        date = index.date()

        # Handle possible multi-level columns
        def get_value(column):
            value = row[column]
            if hasattr(value, "iloc"):
                return value.iloc[0]
            return value

        open_price = get_value("Open")
        high_price = get_value("High")
        low_price = get_value("Low")
        close_price = get_value("Close")
        volume = get_value("Volume")

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

    print(f"Finished {ticker}")


connection.commit()

cursor.close()
connection.close()

print("\nAll stock data loaded successfully!")