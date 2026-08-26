from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed" / "stocks"

for file in PROCESSED_FOLDER.glob("*.parquet"):
    data = pd.read_parquet(file)

    print("\n" + "=" * 50)
    print("FILE:", file.name)
    print("Rows:", len(data))
    print("Missing values:")
    print(data.isnull().sum())
    print("Duplicate rows:", data.duplicated().sum())