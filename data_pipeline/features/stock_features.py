from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os


# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# Database connection
connection = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)


print("Connected to PostgreSQL")


# Load stock prices
query = """
SELECT
    ticker,
    date,
    close
FROM stock_prices
ORDER BY ticker, date;
"""

df = pd.read_sql(query, connection)

print(f"Loaded {len(df)} stock price records")


# Make sure data is correctly sorted
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"])


# Calculate daily return
df["daily_return"] = (
    df.groupby("ticker")["close"]
      .pct_change()
)


# Calculate historical returns
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


# Rolling volatility
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


# Moving averages
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


# Drawdown
df["rolling_peak"] = (
    df.groupby("ticker")["close"]
      .cummax()
)

df["drawdown"] = (
    (df["close"] - df["rolling_peak"])
    / df["rolling_peak"]
)


# 1-year momentum
df["momentum_1y"] = df["return_1y"]


# Remove helper column
df.drop(columns=["rolling_peak"], inplace=True)


# Convert pandas NaN values to Python None
# so PostgreSQL stores them as SQL NULL.
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

for column in feature_columns:
    df[column] = df[column].astype(object)
    df.loc[pd.isna(df[column]), column] = None


# Insert features into PostgreSQL
cursor = connection.cursor()


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
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date)
DO UPDATE SET
    daily_return = EXCLUDED.daily_return,
    return_30d = EXCLUDED.return_30d,
    return_90d = EXCLUDED.return_90d,
    return_180d = EXCLUDED.return_180d,
    return_1y = EXCLUDED.return_1y,
    volatility_30d = EXCLUDED.volatility_30d,
    volatility_90d = EXCLUDED.volatility_90d,
    sma_50 = EXCLUDED.sma_50,
    sma_200 = EXCLUDED.sma_200,
    drawdown = EXCLUDED.drawdown,
    momentum_1y = EXCLUDED.momentum_1y;
"""


for row in df.itertuples(index=False):

    cursor.execute(
        insert_query,
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


connection.commit()

cursor.close()
connection.close()


print("Stock feature engineering completed!")
print(f"Features generated: {len(df)}")