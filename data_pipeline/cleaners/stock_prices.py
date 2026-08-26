from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed" / "stocks"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


for file in RAW_FOLDER.glob("*.parquet"):

    print(f"Cleaning {file.name}...")

    # Read raw data
    data = pd.read_parquet(file)

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Remove rows containing missing values
    data = data.dropna()

    # Sort by date
    data = data.sort_index()

    # Save cleaned data
    output_path = PROCESSED_FOLDER / file.name
    data.to_parquet(output_path)

    print(f"Saved cleaned data: {output_path}")


print("\nData cleaning completed!")