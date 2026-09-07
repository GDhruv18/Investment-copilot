from pathlib import Path

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


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
# LOAD STOCK PRICES
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

print(f"Loaded {len(df):,} stock price records")


# ============================================================
# PREPARE DATA
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)


# ============================================================
# HISTORICAL RETURNS
# ============================================================

print("Calculating returns...")

df["daily_return"] = (
    df.groupby("ticker")["close"]
    .pct_change()
)

df["return_30d"] = (
    df.groupby("ticker")["close"]
    .pct_change(30)
)

df["return_90d"] = (
    df.groupby("ticker")["close"]
    .pct_change(90)
)

df["return_180d"] = (
    df.groupby("ticker")["close"]
    .pct_change(180)
)

df["return_1y"] = (
    df.groupby("ticker")["close"]
    .pct_change(252)
)


# ============================================================
# ROLLING VOLATILITY
# ============================================================

print("Calculating volatility...")

df["volatility_30d"] = (
    df.groupby("ticker")["daily_return"]
    .rolling(30)
    .std()
    .reset_index(level=0, drop=True)
    * (252 ** 0.5)
)

df["volatility_90d"] = (
    df.groupby("ticker")["daily_return"]
    .rolling(90)
    .std()
    .reset_index(level=0, drop=True)
    * (252 ** 0.5)
)


# ============================================================
# MOVING AVERAGES
# ============================================================

print("Calculating moving averages...")

df["sma_50"] = (
    df.groupby("ticker")["close"]
    .rolling(50)
    .mean()
    .reset_index(level=0, drop=True)
)

df["sma_200"] = (
    df.groupby("ticker")["close"]
    .rolling(200)
    .mean()
    .reset_index(level=0, drop=True)
)


# ============================================================
# DRAWDOWN
# ============================================================

print("Calculating drawdown...")

df["rolling_peak"] = (
    df.groupby("ticker")["close"]
    .cummax()
)

df["drawdown"] = (
    (df["close"] - df["rolling_peak"])
    / df["rolling_peak"]
)


# ============================================================
# 1-YEAR MOMENTUM
# ============================================================

df["momentum_1y"] = df["return_1y"]


# ============================================================
# REMOVE HELPER COLUMN
# ============================================================

df.drop(
    columns=["rolling_peak"],
    inplace=True
)


# ============================================================
# REBUILD STOCK FEATURES TABLE
# ============================================================

cursor = connection.cursor()

print("\nClearing existing stock features...")

cursor.execute(
    "TRUNCATE TABLE stock_features RESTART IDENTITY;"
)

connection.commit()


# ============================================================
# PREPARE DATA FOR BULK INSERT
# ============================================================

feature_columns = [
    "daily_return",
    "return_30d",
    "return_90d",
    "return_180d",
    "return_1y",
    "volatility_30d",
    "volatility_90d",
    "sma_50",
    "sma_200",
    "drawdown",
    "momentum_1y"
]


# Convert NaN and infinite values to None
df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df[feature_columns] = df[feature_columns].where(
    pd.notna(df[feature_columns]),
    None
)


# ============================================================
# CREATE INSERT RECORDS
# ============================================================

records = []

for row in df.itertuples(index=False):

    records.append(
        (
            row.ticker,
            row.date.date(),
            row.daily_return,
            row.return_30d,
            row.return_90d,
            row.return_180d,
            row.return_1y,
            row.volatility_30d,
            row.volatility_90d,
            row.sma_50,
            row.sma_200,
            row.drawdown,
            row.momentum_1y
        )
    )


# ============================================================
# BULK INSERT
# ============================================================

print(
    f"Inserting {len(records):,} feature records..."
)

insert_query = """
INSERT INTO stock_features (
    ticker,
    date,
    daily_return,
    return_30d,
    return_90d,
    return_180d,
    return_1y,
    volatility_30d,
    volatility_90d,
    sma_50,
    sma_200,
    drawdown,
    momentum_1y
)
VALUES %s
"""


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


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("STOCK FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"Features generated: {len(df):,}")
print(f"Unique stocks:      {df['ticker'].nunique()}")
print(
    f"Date range:         "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print("\nFeature columns:")

for column in feature_columns:
    print(f"  - {column}")