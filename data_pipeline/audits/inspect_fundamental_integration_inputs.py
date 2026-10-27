
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUTS = {
    "Yahoo Finance company profile": (
        ROOT / "data/clean/fundamentals/yfinance/"
        "yfinance_company_snapshot.parquet"
    ),
    "Yahoo Finance statement observations": (
        ROOT / "data/clean/fundamentals/yfinance/"
        "yfinance_statement_observations.parquet"
    ),
    "Screener.in consolidated observations": (
        ROOT / "data/raw/fundamentals/screener_consolidated_100/"
        "consolidated_financials_long.csv"
    ),
}

TICKER_COLUMNS = [
    "ticker", "symbol", "stock_ticker", "company_ticker"
]

DATE_COLUMNS = [
    "period_end", "report_date", "date", "as_of_date",
    "end_date", "collected_at_utc"
]


def find_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def inspect_dataset(name, path):
    print("\n" + "=" * 85)
    print(name)
    print("=" * 85)
    print(f"Path: {path}")

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        return

    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            print("STATUS: Unsupported file type")
            return
    except Exception as exc:
        print(f"STATUS: READ ERROR — {exc}")
        return

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names and data types:")

    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    ticker_col = find_column(df.columns, TICKER_COLUMNS)
    date_col = find_column(df.columns, DATE_COLUMNS)

    if ticker_col:
        print(f"\nTicker column: {ticker_col}")
        print(f"Unique non-null tickers: {df[ticker_col].nunique():,}")
        print(f"Null ticker rows: {df[ticker_col].isna().sum():,}")
        print("Sample tickers:")
        print(
            df[ticker_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(12)
            .to_list()
        )
    else:
        print("\nTicker column: NOT DETECTED")

    if date_col:
        parsed_dates = pd.to_datetime(
            df[date_col], errors="coerce"
        )
        print(f"\nDate column: {date_col}")
        print(f"Parsed dates: {parsed_dates.notna().sum():,}")
        if parsed_dates.notna().any():
            print(f"Earliest date: {parsed_dates.min()}")
            print(f"Latest date:   {parsed_dates.max()}")

    for col in [
        "section", "section_id", "metric", "period_original",
        "reporting_basis", "source"
    ]:
        if col in df.columns:
            print(f"\n{col} — top values:")
            print(
                df[col]
                .fillna("<NULL>")
                .astype(str)
                .value_counts()
                .head(12)
                .to_string()
            )

    if ticker_col:
        for ticker in ["AUBANK.NS", "BANDHANBNK.NS"]:
            subset = df[
                df[ticker_col].astype(str).eq(ticker)
            ]
            if not subset.empty:
                print(f"\n{ticker}: {len(subset):,} rows")
                visible_cols = list(df.columns[:8])
                print(subset[visible_cols].head(3).to_string(index=False))

    print("\nSample records:")
    print(df.head(3).to_string(index=False, max_cols=12))


def main():
    print("FUNDAMENTAL INPUT INSPECTION — READ ONLY")
    print("No output files will be created.")
    print(f"Project root: {ROOT}")

    for name, path in INPUTS.items():
        inspect_dataset(name, path)

    print("\n" + "=" * 85)
    print("Inspection complete. No input files were modified.")
    print("=" * 85)


if __name__ == "__main__":
    main()
