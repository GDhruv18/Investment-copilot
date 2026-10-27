
from pathlib import Path
import pandas as pd

STATEMENT_PATH = Path(
    "data/clean/fundamentals/yfinance/yfinance_statement_observations.parquet"
)
PROFILE_PATH = Path(
    "data/clean/fundamentals/yfinance/yfinance_company_snapshot.parquet"
)

TICKERS = [
    "HCLTECH.NS", "INFY.NS", "INDIGO.NS", "HAL.NS",
    "NATIONALUM.NS", "VEDL.NS", "DELHIVERY.NS",
    "SIEMENS.NS", "BHEL.NS", "HDFCBANK.NS",
]

ITEMS = [
    "Total Revenue",
    "Operating Revenue",
    "Net Income Common Stockholders",
    "Net Income",
    "Total Debt",
]

def main():
    statements = pd.read_parquet(STATEMENT_PATH)
    profiles = pd.read_parquet(PROFILE_PATH)

    statements["period_end"] = pd.to_datetime(
        statements["period_end"], errors="coerce"
    )
    statements["value"] = pd.to_numeric(
        statements["value"], errors="coerce"
    )

    profile_fields = [
        c for c in [
            "ticker", "currency", "financialCurrency",
            "totalRevenue", "netIncomeToCommon", "totalDebt",
        ]
        if c in profiles.columns
    ]

    print("=" * 90)
    print("PROFILE VALUES")
    print("=" * 90)
    print(
        profiles[profiles["ticker"].isin(TICKERS)][profile_fields]
        .sort_values("ticker")
        .to_string(index=False)
    )

    for ticker in TICKERS:
        print("\n" + "=" * 90)
        print(ticker)
        print("=" * 90)

        subset = statements[
            (statements["ticker"] == ticker)
            & statements["statement_type"].str.startswith(
                "quarterly_", na=False
            )
            & statements["financial_item"].isin(ITEMS)
        ].copy()

        if subset.empty:
            print("No matching quarterly observations.")
            continue

        subset = subset.sort_values(
            ["financial_item", "period_end"]
        )

        for item in ITEMS:
            item_rows = subset[
                subset["financial_item"] == item
            ].tail(8)

            if item_rows.empty:
                continue

            print(f"\n--- {item} ---")
            print(
                item_rows[[
                    "statement_type", "financial_item",
                    "period_end", "value",
                ]].to_string(index=False)
            )

    print("\nInspection complete. No files were modified.")

if __name__ == "__main__":
    main()
