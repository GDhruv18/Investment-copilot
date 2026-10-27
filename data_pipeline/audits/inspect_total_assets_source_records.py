from pathlib import Path
import pandas as pd

tickers = {
    "ZYDUSLIFE.NS", "DRREDDY.NS", "JUBLFOOD.NS", "KPITTECH.NS",
    "TORNTPHARM.NS", "TATASTEEL.NS", "HAL.NS", "MARICO.NS",
    "BHARTIARTL.NS", "ADANIPORTS.NS"
}

paths = {
    "Screener": Path("data/raw/fundamentals/screener_consolidated_100/consolidated_financials_long.csv"),
    "Yahoo": Path("data/clean/fundamentals/yfinance/yfinance_statement_observations.parquet"),
}

def find_col(df, candidates):
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None

for source, path in paths.items():
    print("\n" + "=" * 80)
    print(f"SOURCE: {source} | {path}")
    print("=" * 80)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    df = pd.read_csv(path, low_memory=False) if path.suffix == ".csv" else pd.read_parquet(path)
    print("Columns:", df.columns.tolist())

    ticker_col = find_col(df, ["ticker", "symbol", "stock", "canonical_ticker"])
    metric_col = find_col(df, ["metric", "field", "item", "line_item", "statement_item", "concept"])
    date_col = find_col(df, ["period_end", "date", "end_date", "as_of_date", "report_date", "timestamp"])
    value_col = find_col(df, ["value", "amount", "reported_value", "numeric_value"])

    if ticker_col is None:
        print("Could not identify ticker column; inspect the columns above.")
        continue

    mask = df[ticker_col].astype(str).str.upper().isin(tickers)
    subset = df.loc[mask].copy()

    if metric_col is not None:
        subset = subset[
            subset[metric_col].astype(str).str.contains("total.?assets|assets", case=False, na=False, regex=True)
        ]

    if date_col is not None:
        dates = pd.to_datetime(subset[date_col], errors="coerce")
        subset = subset[dates.between("2025-01-01", "2026-12-31")]

    print(f"Matching rows: {len(subset)}")
    if not subset.empty:
        print(subset.to_string(index=False, max_rows=150))
    else:
        print("No matching rows found with the detected filters.")

