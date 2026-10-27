
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT / "data/clean/fundamentals/integration_candidate/"
    "fundamental_observations_candidate.parquet"
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Candidate not found: {INPUT}")

    df = pd.read_parquet(INPUT)

    print("\nFUNDAMENTAL METRIC VOCABULARY")
    print("=" * 78)
    print(f"Total observations: {len(df):,}")

    for source, group in df.groupby("source", dropna=False):
        print("\n" + "-" * 78)
        print(f"Source: {source}")
        print(f"Rows: {len(group):,}")
        print(f"Unique raw metric names: {group['metric_raw'].nunique():,}")

        metrics = (
            group.groupby("metric_raw", dropna=False)
            .agg(
                rows=("metric_raw", "size"),
                tickers=("ticker", "nunique"),
                statements=("statement_type", "nunique"),
            )
            .sort_values("rows", ascending=False)
        )

        print("\nMost frequent metrics:")
        print(metrics.head(60).to_string())

        print("\nAll metric names:")
        for metric in sorted(
            group["metric_raw"].dropna().astype(str).unique()
        ):
            print(f"  {metric}")

    print("\n" + "=" * 78)
    print("Read-only inspection complete. No output files created.")


if __name__ == "__main__":
    main()
