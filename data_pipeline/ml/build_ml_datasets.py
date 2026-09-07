import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


# ============================================================
# 1. LOAD DATABASE CONFIGURATION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Database environment variables are missing.")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# 2. LOAD STOCK DATA
# ============================================================

print("Loading stock prices...")

prices = pd.read_sql(
    """
    SELECT
        ticker,
        date,
        close
    FROM stock_prices
    WHERE close IS NOT NULL
      AND close <> 'NaN'::numeric
      AND close > 0
    ORDER BY ticker, date
    """,
    engine
)

print(f"Stock price rows: {len(prices):,}")


print("Loading stock features...")

features = pd.read_sql(
    """
    SELECT
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
        drawdown
    FROM stock_features
    ORDER BY ticker, date
    """,
    engine
)

print(f"Stock feature rows: {len(features):,}")


print("Loading stock targets...")

targets = pd.read_sql(
    """
    SELECT
        ticker,
        date,
        target_30d,
        target_90d,
        target_252d
    FROM stock_targets
    ORDER BY ticker, date
    """,
    engine
)

print(f"Stock target rows: {len(targets):,}")


# ============================================================
# 3. CALCULATE RELATIVE SMA FEATURES
# ============================================================

print("Calculating SMA ratios...")

prices["sma50_ratio"] = (
    prices["close"] / features["sma_50"]
)

prices["sma200_ratio"] = (
    prices["close"] / features["sma_200"]
)


# We actually need the SMA values associated with the same
# ticker/date, so calculate them directly from the feature table
# after merging.

data = features.merge(
    prices[
        [
            "ticker",
            "date",
            "close"
        ]
    ],
    on=["ticker", "date"],
    how="inner"
)


# ============================================================
# 4. CALCULATE RELATIVE SMA DISTANCES
# ============================================================

data["sma50_ratio"] = (
    data["close"] / data["sma_50"]
) - 1

data["sma200_ratio"] = (
    data["close"] / data["sma_200"]
) - 1


# ============================================================
# 5. MERGE TARGETS
# ============================================================

data = data.merge(
    targets,
    on=["ticker", "date"],
    how="left"
)


# ============================================================
# 6. SELECT FINAL ML FEATURES
# ============================================================

feature_columns = [
    "daily_return",
    "return_30d",
    "return_90d",
    "return_180d",
    "return_1y",
    "volatility_30d",
    "volatility_90d",
    "sma50_ratio",
    "sma200_ratio",
    "drawdown"
]

target_columns = [
    "target_30d",
    "target_90d",
    "target_252d"
]

final_columns = [
    "ticker",
    "date"
] + feature_columns + target_columns


data = data[final_columns]


# ============================================================
# 7. REMOVE INVALID VALUES
# ============================================================

# Convert infinite values to NaN
data = data.replace([float("inf"), float("-inf")], pd.NA)

# Remove rows where any feature is unavailable
data = data.dropna(subset=feature_columns)

# Keep rows with at least one available target.
# Horizon-specific datasets will later remove rows
# missing their particular target.
data = data.dropna(subset=target_columns, how="all")


# ============================================================
# 8. SORT DATA
# ============================================================

data = data.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)


# ============================================================
# 9. CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = os.path.join(
    "data_pipeline",
    "ml",
    "output"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 10. SAVE COMPLETE STOCK ML DATASET
# ============================================================

output_file = os.path.join(
    output_dir,
    "stock_ml_dataset.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. PRINT VALIDATION INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("STOCK ML DATASET CREATED")
print("=" * 60)

print(f"Rows: {len(data):,}")
print(f"Columns: {len(data.columns)}")
print(f"Stocks: {data['ticker'].nunique()}")

print(
    f"Date range: "
    f"{data['date'].min()} → {data['date'].max()}"
)

print("\nFeature availability:")
print(data[feature_columns].notna().sum())

print("\nTarget availability:")
print(data[target_columns].notna().sum())

print(f"\nSaved to: {output_file}")

print("\nDataset preview:")
print(data.head())