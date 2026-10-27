
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CANDIDATE = (
    ROOT / "data/clean/fundamentals/integration_candidate/"
    "fundamental_observations_candidate.parquet"
)
PROFILE = (
    ROOT / "data/clean/fundamentals/integration_candidate/"
    "company_profile_snapshot.parquet"
)
CONFIG_CANDIDATES = [
    ROOT / "data_pipeline/config/stocks_100.json",
    ROOT / "data_pipeline/config/stocks.json",
]


def load_universe():
    config_path = next(
        (p for p in CONFIG_CANDIDATES if p.exists()), None
    )
    if config_path is None:
        print("WARNING: Stock universe config not found.")
        return None

    data = json.loads(config_path.read_text(encoding="utf-8"))

    tickers = set()

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in {"ticker", "symbol", "provider_symbol"}:
                    if isinstance(value, str) and value.endswith(".NS"):
                        tickers.add(value)
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        elif isinstance(obj, str) and obj.endswith(".NS"):
            tickers.add(obj)

    walk(data)
    print(f"Universe config: {config_path.relative_to(ROOT)}")
    print(f"Configured tickers found: {len(tickers)}")
    return tickers


def main():
    if not CANDIDATE.exists() or not PROFILE.exists():
        raise FileNotFoundError(
            "Integration candidate or profile snapshot is missing."
        )

    df = pd.read_parquet(CANDIDATE)
    profiles = pd.read_parquet(PROFILE)

    print("\n" + "=" * 72)
    print("FUNDAMENTAL INTEGRATION CANDIDATE AUDIT — READ ONLY")
    print("=" * 72)
    print(f"Observation rows: {len(df):,}")
    print(f"Profile rows: {len(profiles):,}")

    candidate_tickers = set(df["ticker"].dropna().astype(str))
    profile_tickers = set(profiles["ticker"].dropna().astype(str))

    print(f"\nUnique observation tickers: {len(candidate_tickers)}")
    print(f"Unique profile tickers: {len(profile_tickers)}")

    print("\nTickers in observations but not profiles:")
    print(sorted(candidate_tickers - profile_tickers) or "None")

    print("\nProfile tickers absent from observations:")
    print(sorted(profile_tickers - candidate_tickers) or "None")

    universe = load_universe()
    if universe is not None:
        print("\nCandidate tickers outside configured universe:")
        print(sorted(candidate_tickers - universe) or "None")

        print("\nConfigured tickers missing from candidate:")
        print(sorted(universe - candidate_tickers) or "None")

    print("\nRows by source:")
    print(df.groupby("source", dropna=False).size().to_string())

    print("\nRows by source and statement type:")
    print(
        df.groupby(["source", "statement_type"], dropna=False)
        .size()
        .to_string()
    )

    print("\nReporting period coverage by source:")
    for source, group in df.groupby("source", dropna=False):
        dates = pd.to_datetime(group["period_end"], errors="coerce")
        print(
            f"{source}: rows={len(group):,}, "
            f"valid_dates={dates.notna().sum():,}, "
            f"earliest={dates.min()}, latest={dates.max()}"
        )

    print("\nDuplicate observation-key checks:")
    keys = [
        "ticker", "source", "statement_type",
        "metric_raw", "period_end"
    ]
    duplicates = df.duplicated(keys, keep=False)
    print(f"Rows involved in duplicate keys: {duplicates.sum():,}")
    if duplicates.any():
        print("Examples:")
        print(
            df.loc[duplicates, keys]
            .sort_values(keys)
            .head(20)
            .to_string(index=False)
        )

    print("\nMissing / invalid values:")
    for col in [
        "ticker", "source", "statement_type",
        "metric_raw", "period_end", "value"
    ]:
        print(f"  {col}: {df[col].isna().sum():,}")

    print("\nUnit verification:")
    print(df.groupby("source")["unit_verified"].agg(
        ["count", "sum"]
    ).to_string())

    print("\nCurrent scoring-eligible rows:")
    print(int(df["scoring_eligible"].sum()))

    print("\nNo files written. No source data modified.")
    print("Audit complete; investigate findings before scoring.")


if __name__ == "__main__":
    main()
