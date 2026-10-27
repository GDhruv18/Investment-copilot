
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "data" / "clean" / "fundamentals" / "yfinance"
REPORT_DIR = ROOT / "data" / "processed" / "fundamentals"

PROFILE_PATH = DATA_DIR / "yfinance_company_snapshot.parquet"
STATEMENT_PATH = DATA_DIR / "yfinance_statement_observations.parquet"


def main():
    if not PROFILE_PATH.exists() or not STATEMENT_PATH.exists():
        raise FileNotFoundError(
            "Staging files are missing. Run "
            "prepare_yfinance_fundamentals.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    print("=" * 65)
    print("FUNDAMENTALS DATA-QUALITY AUDIT")
    print("=" * 65)

    print("\nDataset dimensions:")
    print(f"Profile rows: {len(profiles):,}")
    print(f"Unique profile tickers: {profiles['ticker'].nunique():,}")
    print(f"Statement observations: {len(statements):,}")
    print(
        "Unique statement tickers: "
        f"{statements['ticker'].nunique():,}"
    )

    # 1. Duplicate keys
    statement_keys = [
        "ticker",
        "statement_type",
        "financial_item",
        "period_end",
    ]

    duplicate_mask = statements.duplicated(
        statement_keys,
        keep=False,
    )

    duplicate_rows = statements.loc[duplicate_mask].copy()
    duplicate_path = REPORT_DIR / "yfinance_duplicate_statement_keys.csv"
    duplicate_rows.to_csv(duplicate_path, index=False)

    print(f"\nDuplicate statement-key rows: {len(duplicate_rows):,}")

    # 2. Profile coverage
    fields = [
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "returnOnEquity",
        "returnOnAssets",
        "profitMargins",
        "operatingMargins",
        "revenueGrowth",
        "earningsGrowth",
        "totalRevenue",
        "netIncomeToCommon",
        "totalDebt",
        "totalCash",
        "debtToEquity",
        "currentRatio",
        "freeCashflow",
        "operatingCashflow",
    ]

    print("\nProfile field coverage:")
    for field in fields:
        if field not in profiles.columns:
            print(f"{field:25s} COLUMN MISSING")
            continue

        count = int(profiles[field].notna().sum())
        print(f"{field:25s} {count:3d}/{len(profiles)}")

    # 3. Numeric range checks. These are review flags, not
    # automatic corrections or proof that a value is wrong.
    numeric_fields = [
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "returnOnEquity",
        "returnOnAssets",
        "profitMargins",
        "operatingMargins",
        "revenueGrowth",
        "earningsGrowth",
        "totalRevenue",
        "netIncomeToCommon",
        "totalDebt",
        "totalCash",
        "debtToEquity",
        "currentRatio",
        "freeCashflow",
        "operatingCashflow",
        "dividendYield",
        "beta",
    ]

    for field in numeric_fields:
        if field in profiles.columns:
            profiles[field] = pd.to_numeric(
                profiles[field],
                errors="coerce",
            )

    flag_conditions = {
        "negative_market_cap": (
            profiles["marketCap"] < 0
            if "marketCap" in profiles else None
        ),
        "negative_price_to_book": (
            profiles["priceToBook"] < 0
            if "priceToBook" in profiles else None
        ),
        "negative_debt": (
            profiles["totalDebt"] < 0
            if "totalDebt" in profiles else None
        ),
        "negative_total_revenue": (
            profiles["totalRevenue"] < 0
            if "totalRevenue" in profiles else None
        ),
        "extreme_debt_to_equity_over_200": (
            profiles["debtToEquity"] > 200
            if "debtToEquity" in profiles else None
        ),
        "extreme_trailing_pe_over_150": (
            profiles["trailingPE"] > 150
            if "trailingPE" in profiles else None
        ),
        "negative_trailing_pe": (
            profiles["trailingPE"] < 0
            if "trailingPE" in profiles else None
        ),
    }

    flagged = []

    for flag_name, condition in flag_conditions.items():
        if condition is None:
            continue

        mask = condition.fillna(False)
        print(f"{flag_name}: {int(mask.sum())}")

        for _, row in profiles.loc[mask].iterrows():
            flagged.append({
                "ticker": row.get("ticker"),
                "flag": flag_name,
                "value": row.get(
                    flag_name.replace(
                        "extreme_debt_to_equity_over_200",
                        "debtToEquity",
                    ).replace(
                        "extreme_trailing_pe_over_150",
                        "trailingPE",
                    ).replace(
                        "negative_trailing_pe",
                        "trailingPE",
                    ).replace(
                        "negative_market_cap",
                        "marketCap",
                    ).replace(
                        "negative_price_to_book",
                        "priceToBook",
                    ).replace(
                        "negative_debt",
                        "totalDebt",
                    ).replace(
                        "negative_total_revenue",
                        "totalRevenue",
                    )
                ),
            })

    flags_df = pd.DataFrame(
        flagged,
        columns=["ticker", "flag", "value"],
    )
    flags_path = REPORT_DIR / "yfinance_quality_flags.csv"
    flags_df.to_csv(flags_path, index=False)

    # 4. Statement period coverage by company and statement type
    period_coverage = (
        statements.groupby(["statement_type", "ticker"])
        .agg(
            period_count=("period_end", "nunique"),
            item_count=("financial_item", "nunique"),
            earliest_period=("period_end", "min"),
            latest_period=("period_end", "max"),
            observation_count=("value", "size"),
        )
        .reset_index()
    )

    coverage_path = REPORT_DIR / "yfinance_statement_period_coverage.csv"
    period_coverage.to_csv(coverage_path, index=False)

    print("\nStatement period coverage:")
    if not period_coverage.empty:
        print(
            period_coverage.groupby("statement_type").agg(
                companies=("ticker", "nunique"),
                total_periods=("period_count", "sum"),
                observations=("observation_count", "sum"),
            ).to_string()
        )

    # 5. Ensure numeric statement values are finite.
    if not statements.empty:
        statement_values = pd.to_numeric(
            statements["value"],
            errors="coerce",
        )
        nonfinite = int(
            (~np.isfinite(statement_values.dropna())).sum()
        )
    else:
        nonfinite = 0

    print(f"\nNon-finite numeric statement values: {nonfinite:,}")
    print(f"Quality flags: {flags_path.relative_to(ROOT)}")
    print(f"Duplicate report: {duplicate_path.relative_to(ROOT)}")
    print(f"Period coverage: {coverage_path.relative_to(ROOT)}")
    print("\nAudit completed. No source data or database was modified.")


if __name__ == "__main__":
    main()
