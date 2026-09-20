import os
import pandas as pd

DATA_DIR = "data/raw/etfs"

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet")
)

print("=" * 70)
print("NEW ETF DATA — COMPACT VERIFICATION")
print("=" * 70)

schema_problems = []
empty_files = []
missing_values = []
date_problems = []

summary = []

for file in files:

    path = os.path.join(DATA_DIR, file)

    try:
        df = pd.read_parquet(path)

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Schema check
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            schema_problems.append(
                (file, missing_columns)
            )

        # Empty check
        if df.empty:
            empty_files.append(file)
            continue

        # Missing values
        if df[REQUIRED_COLUMNS].isnull().any().any():
            missing_values.append(file)

        # Date checks
        dates = pd.to_datetime(df.index)

        if dates.duplicated().any():
            date_problems.append(
                (file, "duplicate dates")
            )

        if not dates.is_monotonic_increasing:
            date_problems.append(
                (file, "dates not sorted")
            )

        summary.append(
            (
                file.replace(".parquet", ""),
                len(df),
                dates.min().date(),
                dates.max().date()
            )
        )

    except Exception as e:
        schema_problems.append(
            (file, str(e))
        )


print("\nDATE RANGE SUMMARY")
print("-" * 70)

for name, rows, first, last in summary:
    print(
        f"{name:<25} "
        f"{rows:>5} rows | "
        f"{first} → {last}"
    )

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total files:                 {len(files)}")
print(
    f"Starting on/before 2014:     "
    f"{sum(1 for x in summary if x[2].year <= 2014)}"
)
print(
    f"Starting after 2014:        "
    f"{sum(1 for x in summary if x[2].year > 2014)}"
)

print(f"Schema problems:             {len(schema_problems)}")
print(f"Empty files:                 {len(empty_files)}")
print(f"Files with missing values:   {len(missing_values)}")
print(f"Date problems:               {len(date_problems)}")

print("=" * 70)