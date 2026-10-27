
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PROFILE_PATH = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_company_snapshot.parquet"
)
STATEMENT_PATH = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_statement_observations.parquet"
)
OUTPUT_DIR = ROOT / "data/processed/fundamentals"

TICKERS = [
    "ADANIGREEN.NS",
    "ASHOKLEY.NS",
    "INDIGO.NS",
    "JSWENERGY.NS",
    "JUBLFOOD.NS",
    "TVSMOTOR.NS",
    "DELHIVERY.NS",
    "NYKAA.NS",
    "PAYTM.NS",
]

ITEMS = [
    "Total Debt",
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
    "Net Debt",
    "Net Income Common Stockholders",
    "Net Income",
    "Basic EPS",
    "Diluted EPS",
    "Total Revenue",
    "Operating Revenue",
]


def main():
    profile = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    statements = statements[
        statements["ticker"].isin(TICKERS)
        & statements["statement_type"].isin([
            "annual_balance_sheet",
            "annual_income_statement",
        ])
        & statements["financial_item"].isin(ITEMS)
    ].copy()

    statements["period_end"] = pd.to_datetime(
        statements["period_end"], errors="coerce"
    )
    statements["value"] = pd.to_numeric(
        statements["value"], errors="coerce"
    )

    statements = statements.dropna(subset=["period_end"])

    # Preserve the period end for every ticker/item. Do not assume
    # all statement items have the same latest reporting date.
    latest = (
        statements.sort_values("period_end")
        .groupby(
            ["ticker", "statement_type", "financial_item"],
            as_index=False,
        )
        .tail(1)
    )

    latest = latest.sort_values(
        ["ticker", "statement_type", "financial_item"]
    )

    latest_values = latest.pivot(
        index="ticker",
        columns=["statement_type", "financial_item"],
        values="value",
    )

    latest_dates = latest.pivot(
        index="ticker",
        columns=["statement_type", "financial_item"],
        values="period_end",
    )

    latest_values.columns = [
        f"{statement}__{item}"
        for statement, item in latest_values.columns
    ]
    latest_dates.columns = [
        f"date__{statement}__{item}"
        for statement, item in latest_dates.columns
    ]

    pcols = [
        c for c in [
            "ticker", "trailingPE", "forwardPE", "priceToBook",
            "debtToEquity", "totalDebt", "totalRevenue",
            "netIncomeToCommon", "marketCap",
        ]
        if c in profile.columns
    ]

    profile_subset = profile[
        profile["ticker"].isin(TICKERS)
    ][pcols].drop_duplicates("ticker").set_index("ticker")

    result = profile_subset.join(latest_values, how="left")
    result = result.join(latest_dates, how="left")
    result = result.reset_index()

    debt_col = (
        "annual_balance_sheet__Total Debt"
    )
    equity_col = (
        "annual_balance_sheet__Stockholders Equity"
    )

    if debt_col in result.columns and equity_col in result.columns:
        debt = result[debt_col]
        equity = result[equity_col]

        result["diagnostic_debt_to_equity_pct"] = np.where(
            equity > 0,
            debt / equity * 100,
            np.nan,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    values_path = OUTPUT_DIR / "extreme_ratio_period_aligned_values.csv"
    latest.to_csv(
        OUTPUT_DIR / "extreme_ratio_latest_statement_observations.csv",
        index=False,
    )
    result.to_csv(values_path, index=False)

    print("=" * 88)
    print("EXTREME-RATIO PERIOD AND VALUE VALIDATION")
    print("=" * 88)
    print(f"Companies inspected: {result['ticker'].nunique()}")
    print(f"Annual statement observations selected: {len(latest)}")

    date_cols = [
        c for c in result.columns
        if c.startswith("date__")
    ]
    print("\nLatest statement period dates:")
    if date_cols:
        print(
            result[["ticker"] + date_cols]
            .to_string(index=False)
        )

    print("\nProfile and latest statement values:")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 280,
    ):
        print(result.to_string(index=False))

    print(f"\nDetailed report: {values_path}")
    print("Source data modified: No")
    print("Database modified: No")
    print("Models modified: No")


if __name__ == "__main__":
    main()
