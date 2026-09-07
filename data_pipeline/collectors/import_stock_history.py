from pathlib import Path

import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# New expanded 2018+ raw stock data
STOCK_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"


# ============================================================
# DATABASE CONNECTION
# ============================================================

connection = psycopg2.connect(
    host="localhost",
    port=5432,
    database="investment_copilot",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)

cursor = connection.cursor()


# ============================================================
# FIND STOCK FILES
# ============================================================

files = list(STOCK_FOLDER.glob("*.parquet"))

print(f"Found {len(files)} stock files.")


# ============================================================
# IMPORT DATA
# ============================================================

total_processed = 0
total_failed = 0


for index, file in enumerate(files, start=1):

    ticker = file.stem.replace("_", ".")

    print(f"\n[{index}/{len(files)}] Importing {ticker}...")

    try:

        # ----------------------------------------------------
        # Read Parquet file
        # ----------------------------------------------------

        data = pd.read_parquet(file)


        # ----------------------------------------------------
        # Handle Yahoo Finance multi-level columns
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)


        # ----------------------------------------------------
        # Convert index to Date column
        # ----------------------------------------------------

        data = data.reset_index()


        # ----------------------------------------------------
        # Import each row
        # ----------------------------------------------------

        rows_processed = 0

        for _, row in data.iterrows():

            date = pd.to_datetime(
                row["Date"]
            ).date()

            open_price = row["Open"]
            high_price = row["High"]
            low_price = row["Low"]
            close_price = row["Close"]
            volume = row["Volume"]


            # ------------------------------------------------
            # Skip invalid price records
            # ------------------------------------------------

            if (
                pd.isna(open_price)
                or pd.isna(high_price)
                or pd.isna(low_price)
                or pd.isna(close_price)
            ):
                continue


            # ------------------------------------------------
            # Insert or update
            # ------------------------------------------------

            cursor.execute(
                """
                INSERT INTO stock_prices
                (
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)

                ON CONFLICT (ticker, date)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
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

            rows_processed += 1


        # ----------------------------------------------------
        # Commit this stock
        # ----------------------------------------------------

        connection.commit()

        total_processed += rows_processed

        print(
            f"Processed {rows_processed} valid rows."
        )


    except Exception as error:

        connection.rollback()

        total_failed += 1

        print(f"FAILED: {ticker}")
        print(error)


# ============================================================
# CLOSE DATABASE CONNECTION
# ============================================================

cursor.close()
connection.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("HISTORICAL STOCK IMPORT COMPLETED")
print("=" * 60)

print(f"Stock files found:       {len(files)}")
print(f"Stock files failed:      {total_failed}")
print(f"Valid rows processed:    {total_processed:,}")