
from pathlib import Path
import pandas as pd

SOURCE = Path(
    "data/clean/fundamentals/yfinance/yfinance_statement_observations.parquet"
)
OUTPUT = Path(
    "data/processed/fundamentals/quarterly_coverage_audit.csv"
)

ITEMS = [
    "Total Revenue",
    "Operating Revenue",
    "Net Income Common Stockholders",
    "Net Income",
]

def main():
    df = pd.read_parquet(SOURCE)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    df = df[
        df["statement_type"].eq("quarterly_income_statement")
        & df["financial_item"].isin(ITEMS)
        & df["period_end"].notna()
    ].copy()

    rows = []

    for (ticker, item), group in df.groupby(
        ["ticker", "financial_item"]
    ):
        # Unique dates only; this checks date coverage, not whether values
        # are standalone quarterly results rather than year-to-date values.
        dates = sorted(group["period_end"].drop_duplicates())
        latest_dates = dates[-4:]

        consecutive = False
        if len(latest_dates) == 4:
            consecutive = all(
                (latest_dates[i + 1] - latest_dates[i]).days
                in range(75, 105)
                for i in range(3)
            )

        rows.append({
            "ticker": ticker,
            "financial_item": item,
            "unique_quarter_dates": len(dates),
            "latest_period": dates[-1].date().isoformat() if dates else None,
            "four_latest_dates": " | ".join(
                d.date().isoformat() for d in latest_dates
            ),
            "has_four_dates": len(latest_dates) == 4,
            "approximately_consecutive": consecutive,
        })

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("Ticker/item rows:", len(result))
    print("\nCoverage summary:")
    print(
        result.groupby("financial_item")[
            ["has_four_dates", "approximately_consecutive"]
        ].sum().to_string()
    )

    print("\nNot approximately consecutive:")
    print(
        result[~result["approximately_consecutive"]]
        .sort_values(["ticker", "financial_item"])
        .to_string(index=False)
    )

    print("\nReport:", OUTPUT)
    print("Read-only audit; no source files were modified.")

if __name__ == "__main__":
    main()
