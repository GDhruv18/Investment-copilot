from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"

for file in STOCK_FOLDER.glob("*.parquet"):
    print("\n" + "=" * 50)
    print("FILE:", file.name)

    data = pd.read_parquet(file)

    print("Rows:", len(data))
    print("Columns:", data.columns.tolist())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())