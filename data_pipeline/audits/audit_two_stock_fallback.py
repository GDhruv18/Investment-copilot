
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TARGETS = {"AUBANK.NS", "BANDHANBNK.NS"}

FILES = {
    "yfinance_profile": (
        ROOT / "data" / "clean" / "fundamentals"
        / "yfinance" / "yfinance_company_snapshot.parquet"
    ),
    "yfinance_statements": (
        ROOT / "data" / "clean" / "fundamentals"
        / "yfinance" / "yfinance_statement_observations.parquet"
    ),
}

OUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "fallback_audit"
)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    for dataset_name, path in FILES.items():
        if not path.exists():
            report.append({
                "dataset": dataset_name,
                "status": "FILE_MISSING",
                "path": str(path),
            })
            continue

        df = pd.read_parquet(path)
        print(f"\n{dataset_name}")
        print("Columns:", list(df.columns))

        ticker_col = next(
            (
                col for col in ["ticker", "symbol", "stock_ticker"]
                if col in df.columns
            ),
            None,
        )

        if ticker_col is None:
            report.append({
                "dataset": dataset_name,
                "status": "TICKER_COLUMN_NOT_IDENTIFIED",
                "columns": ", ".join(map(str, df.columns)),
                "rows": len(df),
            })
            continue

        subset = df[
            df[ticker_col].astype(str).isin(TARGETS)
        ].copy()

        subset.to_csv(
            OUT_DIR / f"{dataset_name}_two_stocks.csv",
            index=False,
        )

        date_col = next(
            (
                col for col in [
                    "period_end", "report_date", "date",
                    "as_of_date", "end_date"
                ] if col in subset.columns
            ),
            None,
        )

        for ticker in sorted(TARGETS):
            rows = subset[subset[ticker_col].astype(str) == ticker]
            report.append({
                "dataset": dataset_name,
                "ticker": ticker,
                "status": "DATA_FOUND" if not rows.empty else "NO_DATA",
                "rows": len(rows),
                "date_column": date_col or "",
                "earliest_period": (
                    str(rows[date_col].min())
                    if date_col and not rows.empty else ""
                ),
                "latest_period": (
                    str(rows[date_col].max())
                    if date_col and not rows.empty else ""
                ),
                "columns": ", ".join(map(str, subset.columns)),
            })

    result = pd.DataFrame(report)
    result.to_csv(OUT_DIR / "fallback_coverage.csv", index=False)

    print("\nFALLBACK COVERAGE AUDIT")
    print(result.to_string(index=False))
    print(f"\nOutput: {OUT_DIR / 'fallback_coverage.csv'}")

if __name__ == "__main__":
    main()
