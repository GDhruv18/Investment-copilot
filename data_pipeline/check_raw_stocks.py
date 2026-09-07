from pathlib import Path
import pandas as pd


folder = Path("data/raw/stocks")

files = list(folder.glob("*.parquet"))

print("=" * 60)
print("RAW STOCK DATA VERIFICATION")
print("=" * 60)

print(f"Parquet files: {len(files)}")

results = []

for file in files:

    data = pd.read_parquet(file)

    results.append({
        "file": file.name,
        "rows": len(data),
        "first_date": data.index.min(),
        "last_date": data.index.max()
    })


results.sort(key=lambda x: x["first_date"])


print(
    "\nFiles with history beginning in 2018:"
)

count_2018 = 0

for result in results:

    if result["first_date"].year == 2018:

        count_2018 += 1


print(count_2018)


print("\nEarliest 10 stocks:")

for result in results[:10]:

    print(
        result["file"],
        "| Rows:", result["rows"],
        "|",
        result["first_date"].date(),
        "→",
        result["last_date"].date()
    )


print("\nLatest 10 stocks:")

for result in results[-10:]:

    print(
        result["file"],
        "| Rows:", result["rows"],
        "|",
        result["first_date"].date(),
        "→",
        result["last_date"].date()
    )


print("\nStocks with history starting after 2018:")

for result in results:

    if result["first_date"].year > 2018:

        print(
            result["file"],
            "|",
            result["first_date"].date()
        )


print("\n" + "=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)