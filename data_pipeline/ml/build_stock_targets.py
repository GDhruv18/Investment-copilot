from pathlib import Path

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# DATABASE CONNECTION
# ============================================================

connection = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

print("Connected to PostgreSQL")


# ============================================================
# LOAD VALID STOCK PRICES
# ============================================================

query = """
SELECT
    ticker,
    date,
    close
FROM stock_prices
WHERE close IS NOT NULL
  AND close <> 'NaN'::numeric
  AND close > 0
ORDER BY ticker, date;
"""

df = pd.read_sql(query, connection)

print(f"Loaded {len(df):,} valid price records")


# ============================================================
# PREPARE DATA
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)


# ============================================================
# CALCULATE FUTURE PRICES
# ============================================================

print("Calculating 30D, 90D and 252D future prices...")

df["price_30d"] = (
    df.groupby("ticker")["close"]
    .shift(-30)
)

df["price_90d"] = (
    df.groupby("ticker")["close"]
    .shift(-90)
)

df["price_252d"] = (
    df.groupby("ticker")["close"]
    .shift(-252)
)


# ============================================================
# CALCULATE TARGET RETURNS
# ============================================================

print("Calculating target returns...")

df["target_30d"] = (
    df["price_30d"] / df["close"]
) - 1

df["target_90d"] = (
    df["price_90d"] / df["close"]
) - 1

df["target_252d"] = (
    df["price_252d"] / df["close"]
) - 1


# ============================================================
# SELECT TARGET COLUMNS
# ============================================================

target_columns = [
    "ticker",
    "date",
    "close",
    "price_30d",
    "target_30d",
    "price_90d",
    "target_90d",
    "price_252d",
    "target_252d"
]

df = df[target_columns]


# ============================================================
# CLEAN INVALID VALUES
# ============================================================

numeric_columns = [
    "close",
    "price_30d",
    "target_30d",
    "price_90d",
    "target_90d",
    "price_252d",
    "target_252d"
]

df[numeric_columns] = df[numeric_columns].replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# ============================================================
# REBUILD TARGET TABLE
# ============================================================

cursor = connection.cursor()

print("Clearing existing stock_targets...")

cursor.execute(
    "TRUNCATE TABLE stock_targets RESTART IDENTITY;"
)

connection.commit()


# ============================================================
# PREPARE RECORDS
# ============================================================

records = []

for row in df.itertuples(index=False):

    records.append(
        (
            row.ticker,
            row.date.date(),
            row.close,
            row.price_30d,
            row.target_30d,
            row.price_90d,
            row.target_90d,
            row.price_252d,
            row.target_252d
        )
    )


# ============================================================
# BULK INSERT
# ============================================================

print(
    f"Inserting {len(records):,} target records..."
)

insert_query = """
INSERT INTO stock_targets (
    ticker,
    date,
    close_price,
    price_30d,
    target_30d,
    price_90d,
    target_90d,
    price_252d,
    target_252d
)
VALUES %s
"""

from psycopg2.extras import execute_values

execute_values(
    cursor,
    insert_query,
    records,
    page_size=5000
)


# ============================================================
# COMMIT
# ============================================================

connection.commit()

cursor.close()
connection.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("STOCK TARGET ENGINEERING COMPLETED")
print("=" * 60)

print(f"Total target rows: {len(df):,}")
print(f"Unique stocks:     {df['ticker'].nunique()}")

print(
    f"Date range:        "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print("\nTarget availability:")

print(
    f"30D:  {df['target_30d'].notna().sum():,}"
)

print(
    f"90D:  {df['target_90d'].notna().sum():,}"
)

print(
    f"252D: {df['target_252d'].notna().sum():,}"
)