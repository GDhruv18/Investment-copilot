from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_FOLDER = PROJECT_ROOT / "data" / "raw" / "stocks"

files = sorted(STOCK_FOLDER.glob("*.parquet"))

print("=" * 70)
print("NEW STOCK DATA — COMPACT VERIFICATION")
print("=" * 70)

print(f"Parquet files: {len(files)}")

expected = {
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
}

bad_schema = []
empty_files = []
missing_values = []
date_problems = []

for file in files:
    ticker = file.stem

    try:
        df = pd.read_parquet(file)

        # Flatten Yahoo MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            columns = list(df.columns.get_level_values(0))
        else:
            columns = list(df.columns)

        columns_set = set(columns)

        if not expected.issubset(columns_set):
            bad_schema.append(
                (ticker, columns)
            )

        if len(df) == 0:
            empty_files.append(ticker)
            continue

        # Date information
        dates = pd.to_datetime(df.index)

        if dates.duplicated().any():
            date_problems.append(
                (ticker, "duplicate dates")
            )

        # Missing values
        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]

        # Access using flattened columns when necessary
        temp = df.copy()

        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)

        missing = temp[required].isna().sum().sum()

        if missing > 0:
            missing_values.append(
                (ticker, int(missing))
            )

    except Exception as e:
        bad_schema.append(
            (ticker, str(e))
        )

print()
print("SCHEMA PROBLEMS:", len(bad_schema))

for item in bad_schema:
    print(" ", item)

print()
print("EMPTY FILES:", len(empty_files))

for item in empty_files:
    print(" ", item)

print()
print("FILES WITH MISSING VALUES:", len(missing_values))

for ticker, count in missing_values:
    print(f"  {ticker}: {count}")

print()
print("DATE PROBLEMS:", len(date_problems))

for item in date_problems:
    print(" ", item)

print()
print("DATE RANGE SUMMARY")
print("-" * 70)

start_2014 = 0
later_start = 0

for file in files:
    try:
        df = pd.read_parquet(file)

        dates = pd.to_datetime(df.index)

        first = dates.min()
        last = dates.max()

        if first <= pd.Timestamp("2014-01-01"):
            start_2014 += 1
        else:
            later_start += 1

        print(
            f"{file.stem:25} "
            f"{len(df):5} rows | "
            f"{first.date()} → {last.date()}"
        )

    except Exception:
        pass

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total files:                 {len(files)}")
print(f"Starting on/before 2014:     {start_2014}")
print(f"Starting after 2014:         {later_start}")
print(f"Schema problems:             {len(bad_schema)}")
print(f"Files with missing values:   {len(missing_values)}")
print(f"Date problems:               {len(date_problems)}")
print("=" * 70)