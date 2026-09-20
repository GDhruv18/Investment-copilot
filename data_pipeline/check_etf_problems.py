import os
import pandas as pd
import json

DATA_DIR = "data/raw/etfs"
CONFIG_FILE = "data_pipeline/config/etfs_42_validated.json"

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

if isinstance(config, dict):
    expected = set(config["etfs"])
else:
    expected = set(config)

print("=" * 70)
print("ETF DATA PROBLEM DIAGNOSTIC")
print("=" * 70)

files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet")
)

actual = {
    f.replace(".parquet", "").replace("_NS", ".NS")
    for f in files
}

print(f"\nExpected ETFs: {len(expected)}")
print(f"Parquet files: {len(files)}")

print("\n" + "-" * 70)
print("EXTRA FILES")
print("-" * 70)

extra = sorted(actual - expected)

if extra:
    for ticker in extra:
        print(ticker)
else:
    print("None")


print("\n" + "-" * 70)
print("MISSING FROM EXPECTED UNIVERSE")
print("-" * 70)

missing = sorted(expected - actual)

if missing:
    for ticker in missing:
        print(ticker)
else:
    print("None")


print("\n" + "-" * 70)
print("FILES WITH MISSING VALUES")
print("-" * 70)

problem_files = []

for file in files:
    path = os.path.join(DATA_DIR, file)

    try:
        df = pd.read_parquet(path)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            print(f"\n{file}")
            print(f"  Missing columns: {missing}")
            problem_files.append(file)
            continue

        null_counts = df[required].isnull().sum()
        null_counts = null_counts[null_counts > 0]

        if len(null_counts) > 0:
            print(f"\n{file}")
            print(f"  Rows: {len(df)}")
            print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
            print("  Missing values:")

            for column, count in null_counts.items():
                print(f"    {column}: {count}")

            problem_files.append(file)

    except Exception as e:
        print(f"\n{file}")
        print(f"  ERROR: {e}")
        problem_files.append(file)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total Parquet files:          {len(files)}")
print(f"Expected universe:            {len(expected)}")
print(f"Extra files:                  {len(extra)}")
print(f"Missing expected files:       {len(missing)}")
print(f"Files with data problems:     {len(problem_files)}")
print("=" * 70)