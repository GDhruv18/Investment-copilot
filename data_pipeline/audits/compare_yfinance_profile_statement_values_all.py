
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

# Profile column -> statement type -> financial item
METRICS = {
    "totalDebt": (
        "annual_balance_sheet",
        "Total Debt",
    ),
    "totalRevenue": (
        "annual_income_statement",
        "Total Revenue",
    ),
    "netIncomeToCommon": (
        "annual_income_statement",
        "Net Income Common Stockholders",
    ),
}

THRESHOLDS = [5, 10, 20]


def latest_observations(statements):
    """Return the latest observation for each ticker and metric."""
    needed = {
        (statement_type, item)
        for statement_type, item in METRICS.values()
    }

    data = statements[
        statements.apply(
            lambda row: (
                row["statement_type"],
                row["financial_item"],
            ) in needed,
            axis=1,
        )
    ].copy()

    data["period_end"] = pd.to_datetime(
        data["period_end"], errors="coerce"
    )
    data["value"] = pd.to_numeric(
        data["value"], errors="coerce"
    )
    data = data.dropna(subset=["period_end"])

    # Latest date per ticker, statement type, and item.
    data = (
        data.sort_values("period_end")
        .groupby(
            ["ticker", "statement_type", "financial_item"],
            as_index=False,
        )
        .tail(1)
    )

    return data


def main():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(PROFILE_PATH)
    if not STATEMENT_PATH.exists():
        raise FileNotFoundError(STATEMENT_PATH)

    profile = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    if "ticker" not in profile.columns:
        raise ValueError("Profile dataset has no ticker column.")

    required_statement_columns = {
        "ticker", "statement_type", "financial_item",
        "period_end", "value",
    }
    missing = required_statement_columns - set(statements.columns)
    if missing:
        raise ValueError(
            f"Statement dataset missing columns: {sorted(missing)}"
        )

    profile = profile.drop_duplicates("ticker").copy()
    latest = latest_observations(statements)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    for _, company in profile.iterrows():
        ticker = company["ticker"]

        for profile_field, (statement_type, item) in METRICS.items():
            if profile_field not in profile.columns:
                profile_value = np.nan
            else:
                profile_value = pd.to_numeric(
                    pd.Series([company[profile_field]]),
                    errors="coerce",
                ).iloc[0]

            match = latest[
                latest["ticker"].eq(ticker)
                & latest["statement_type"].eq(statement_type)
                & latest["financial_item"].eq(item)
            ]

            if match.empty:
                statement_value = np.nan
                period_end = pd.NaT
            else:
                selected = match.iloc[0]
                statement_value = selected["value"]
                period_end = selected["period_end"]

            both_present = (
                pd.notna(profile_value)
                and pd.notna(statement_value)
            )

            if both_present:
                absolute_difference = (
                    profile_value - statement_value
                )

                if abs(statement_value) > 1e-12:
                    relative_difference_pct = (
                        absolute_difference
                        / abs(statement_value)
                        * 100
                    )
                else:
                    relative_difference_pct = np.nan
            else:
                absolute_difference = np.nan
                relative_difference_pct = np.nan

            if not both_present:
                status = "missing_comparison_value"
            elif statement_value == 0:
                status = "zero_statement_denominator"
            elif abs(relative_difference_pct) <= 5:
                status = "within_5_percent"
            elif abs(relative_difference_pct) <= 10:
                status = "5_to_10_percent"
            elif abs(relative_difference_pct) <= 20:
                status = "10_to_20_percent"
            else:
                status = "over_20_percent"

            comparison_rows.append({
                "ticker": ticker,
                "metric": profile_field,
                "statement_type": statement_type,
                "financial_item": item,
                "profile_value": profile_value,
                "statement_value": statement_value,
                "statement_period_end": period_end,
                "absolute_difference": absolute_difference,
                "relative_difference_pct": relative_difference_pct,
                "comparison_status": status,
            })

    comparison = pd.DataFrame(comparison_rows)

    comparison_path = (
        OUTPUT_DIR / "profile_statement_comparison_all_stocks.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    summary_rows = []

    for metric, group in comparison.groupby("metric"):
        valid = group["relative_difference_pct"].dropna()
        summary = {
            "metric": metric,
            "companies_in_profile": int(group["ticker"].nunique()),
            "comparable_values": int(valid.size),
            "missing_or_noncomparable": int(valid.isna().sum())
            + int(group["relative_difference_pct"].isna().sum())
            - int(valid.isna().sum()),
            "median_absolute_difference_pct": (
                float(valid.abs().median())
                if not valid.empty else np.nan
            ),
            "mean_absolute_difference_pct": (
                float(valid.abs().mean())
                if not valid.empty else np.nan
            ),
        }

        for threshold in THRESHOLDS:
            summary[f"count_over_{threshold}_pct"] = int(
                (valid.abs() > threshold).sum()
            )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = (
        OUTPUT_DIR / "profile_statement_discrepancy_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print("=" * 78)
    print("ALL-STOCK PROFILE VS STATEMENT AUDIT")
    print("=" * 78)
    print(f"Unique profile tickers: {profile['ticker'].nunique()}")
    print(f"Comparison rows: {len(comparison)}")
    print("\nMetric summary:")
    print(summary_df.to_string(index=False))

    print("\nLargest absolute percentage differences:")
    for metric, group in comparison.groupby("metric"):
        print(f"\n{metric}")
        cols = [
            "ticker", "profile_value", "statement_value",
            "statement_period_end", "relative_difference_pct",
            "comparison_status",
        ]
        print(
            group.assign(
                abs_diff=group["relative_difference_pct"].abs()
            )
            .sort_values("abs_diff", ascending=False)
            .head(10)[cols]
            .to_string(index=False)
        )

    print("\nComparison status counts:")
    print(
        comparison.groupby(
            ["metric", "comparison_status"]
        ).size().to_string()
    )

    print(f"\nDetailed report: {comparison_path}")
    print(f"Summary report: {summary_path}")
    print("Source data modified: No")
    print("Database modified: No")
    print("Models modified: No")


if __name__ == "__main__":
    main()
