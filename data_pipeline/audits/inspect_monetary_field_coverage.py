import pandas as pd

path = "data/clean/fundamentals/normalization_candidate/fundamental_observations_normalization_candidate.parquet"

df = pd.read_parquet(
    path,
    columns=["source", "statement_type", "metric_raw"]
)

patterns = [
    "revenue", "sales", "asset", "borrow",
    "debt", "cash flow", "operating cash"
]

for pattern in patterns:
    matches = df[
        df["metric_raw"].astype(str).str.contains(
            pattern, case=False, regex=False, na=False
        )
    ]

    print(f"\n=== {pattern.upper()} ===")

    if matches.empty:
        print("No matches")
    else:
        print(
            matches.groupby(
                ["source", "statement_type", "metric_raw"],
                dropna=False
            ).size().to_string()
        )
