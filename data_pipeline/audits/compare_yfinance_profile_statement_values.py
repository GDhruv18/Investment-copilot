
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
    "ADANIGREEN.NS", "ASHOKLEY.NS", "INDIGO.NS",
    "JSWENERGY.NS", "JUBLFOOD.NS", "TVSMOTOR.NS",
    "DELHIVERY.NS", "NYKAA.NS", "PAYTM.NS",
]

MAPPINGS = {
    "profile_totalDebt": (
        "totalDebt", "annual_balance_sheet", "Total Debt"
    ),
    "profile_totalRevenue": (
        "totalRevenue", "annual_income_statement", "Total Revenue"
    ),
    "profile_netIncomeToCommon": (
        "netIncomeToCommon", "annual_income_statement",
        "Net Income Common Stockholders"
    ),
}


def main():
    profile = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    profile = profile[profile["ticker"].isin(TICKERS)].copy()

    statements = statements[
        statements["ticker"].isin(TICKERS)
        & statements["statement_type"].isin([
            "annual_balance_sheet",
            "annual_income_statement",
        ])
        & statements["financial_item"].isin([
            "Total Debt",
            "Total Revenue",
            "Net Income Common Stockholders",
        ])
    ].copy()

    statements["period_end"] = pd.to_datetime(
        statements["period_end"], errors="coerce"
    )
    statements["value"] = pd.to_numeric(
        statements["value"], errors="coerce"
    )

    statements = statements.dropna(subset=["period_end"])

    latest = (
        statements.sort_values("period_end")
        .groupby(
            ["ticker", "statement_type", "financial_item"],
            as_index=False,
        )
        .tail(1)
    )

    latest["statement_field"] = (
        latest["statement_type"] + "__" + latest["financial_item"]
    )

    values = latest.pivot(
        index="ticker",
        columns="statement_field",
        values="value",
    )

    dates = latest.pivot(
        index="ticker",
        columns="statement_field",
        values="period_end",
    )
    dates.columns = [f"date__{c}" for c in dates.columns]

    profile = profile.set_index("ticker")
    result = profile.join(values, how="left").join(dates, how="left")

    for output, (profile_col, statement_type, item) in MAPPINGS.items():
        statement_col = f"{statement_type}__{item}"

        if profile_col not in result.columns:
            continue
        if statement_col not in result.columns:
            continue

        p = pd.to_numeric(result[profile_col], errors="coerce")
        s = pd.to_numeric(result[statement_col], errors="coerce")

        # Comparison is diagnostic only; the profile and statement
        # may differ in reporting period or provider definition.
        result[f"{output}_absolute_difference"] = p - s
        result[f"{output}_relative_difference_pct"] = np.where(
            s.abs() > 1e-12,
            (p - s) / s.abs() * 100,
            np.nan,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "profile_statement_value_comparison.csv"
    result.reset_index().to_csv(path, index=False)

    print("PROFILE VS STATEMENT COMPARISON")
    print("=" * 75)

    for ticker in TICKERS:
        if ticker not in result.index:
            print(f"\n{ticker}: missing profile")
            continue

        row = result.loc[ticker]
        print(f"\n{ticker}")

        for output, (profile_col, statement_type, item) in MAPPINGS.items():
            statement_col = f"{statement_type}__{item}"
            date_col = f"date__{statement_col}"

            if profile_col not in result.columns:
                continue

            print(
                f"  {profile_col}: {row.get(profile_col, np.nan)}"
            )
            print(
                f"  statement {item}: "
                f"{row.get(statement_col, np.nan)}"
            )
            print(
                f"  statement date: {row.get(date_col, np.nan)}"
            )
            print(
                f"  relative difference (%): "
                f"{row.get(output + '_relative_difference_pct', np.nan)}"
            )

    print(f"\nSaved: {path}")
    print("Source data modified: No")
    print("Database modified: No")
    print("Models modified: No")


if __name__ == "__main__":
    main()
