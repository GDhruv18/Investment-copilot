from pathlib import Path

import yfinance as yf
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


# Process each stock
for file in STOCK_FOLDER.glob("*.parquet"):

    ticker = file.stem.replace("_", ".")

    print(f"\nChecking {ticker}...")

    # Get the latest date for THIS stock
    cursor.execute(
        """
        SELECT MAX(date)
        FROM stock_prices
        WHERE ticker = %s;
        """,
        (ticker,)
    )

    latest_date = cursor.fetchone()[0]

    print("Latest date in database:", latest_date)

    # Download recent data
    data = yf.download(
        ticker,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(f"No new data found for {ticker}")
        continue

    for index, row in data.iterrows():

        date = index.date()

        # Skip data already in database
        if latest_date is not None and date <= latest_date:
            continue

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

        print(f"Added {ticker} - {date}")


connection.commit()

cursor.close()
connection.close()

print("\nStock database update completed!")