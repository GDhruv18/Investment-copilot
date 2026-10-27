
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_PATH = Path(
    "data/clean/fundamentals/yfinance/yfinance_company_snapshot.parquet"
)
STATEMENT_PATH = Path(
    "data/clean/fundamentals/yfinance/yfinance_statement_observations.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/fundamentals/profile_period_alignment_audit.csv"
)

METRICS = {
    "totalRevenue": {
        "statement_type": "income_statement",
        "items": ["Total Revenue", "Operating Revenue"],
        "mode": "flow",
    },
    "netIncomeToCommon": {
        "statement_type": "income_statement",
        "items": ["Net Income Common Stockholders", "Net Income"],
        "mode": "flow",
    },
    "totalDebt": {
        "statement_type": "balance_sheet",
        "items": ["Total Debt"],
        "mode": "point_in_time",
    },
}


def pct_difference(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan

    return abs(a - b) / abs(b) * 100


def choose_item(frame, items):
    """Choose the preferred available item, preserving its reporting dates."""
    for item in items:
        selected = frame[frame["financial_item"] == item].copy()
        if not selected.empty:
            return selected
    return frame.iloc[0:0].copy()


def main():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(PROFILE_PATH)
    if not STATEMENT_PATH.exists():
        raise FileNotFoundError(STATEMENT_PATH)

    profiles = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    statements["period_end"] = pd.to_datetime(
        statements["period_end"], errors="coerce"
    )
    statements["value"] = pd.to_numeric(
        statements["value"], errors="coerce"
    )

    rows = []

    for _, profile in profiles.iterrows():
        ticker = profile["ticker"]

        for profile_field, spec in METRICS.items():
            profile_value = profile.get(profile_field)

            matching = statements[
                (statements["ticker"] == ticker)
                & statements["statement_type"].str.endswith(
                    spec["statement_type"], na=False
                )
                & statements["financial_item"].isin(spec["items"])
                & statements["period_end"].notna()
                & statements["value"].notna()
            ].copy()

            for frequency in ["annual", "quarterly"]:
                subset = matching[
                    matching["statement_type"].str.startswith(
                        frequency, na=False
                    )
                ].copy()

                if subset.empty:
                    rows.append({
                        "ticker": ticker,
                        "profile_field": profile_field,
                        "profile_value": profile_value,
                        "comparison": f"{frequency}_unavailable",
                        "statement_item": None,
                        "period_end": None,
                        "period_count": 0,
                        "statement_value": np.nan,
                        "absolute_difference_pct": np.nan,
                        "note": "No matching statement observations",
                    })
                    continue

                subset = choose_item(subset, spec["items"])
                subset = subset.sort_values("period_end")

                if subset.empty:
                    continue

                if spec["mode"] == "flow":
                    if frequency == "annual":
                        latest_date = subset["period_end"].max()
                        latest = subset[
                            subset["period_end"] == latest_date
                        ]
                        value = latest.iloc[0]["value"]
                        period_end = latest_date
                        period_count = 1
                        comparison = "latest_annual"
                        item = latest.iloc[0]["financial_item"]
                        note = "One annual reporting period"
                    else:
                        # TTM approximation: sum the latest four distinct
                        # quarterly observations for the selected item.
                        subset = subset.drop_duplicates(
                            "period_end", keep="last"
                        ).sort_values("period_end")
                        latest = subset.tail(4)

                        if len(latest) < 4:
                            value = np.nan
                            period_end = (
                                latest["period_end"].max()
                                if not latest.empty else pd.NaT
                            )
                            period_count = len(latest)
                            comparison = "quarterly_ttm_unavailable"
                            item = (
                                latest.iloc[-1]["financial_item"]
                                if not latest.empty else None
                            )
                            note = "Fewer than four quarterly observations"
                        else:
                            value = latest["value"].sum()
                            period_end = latest["period_end"].max()
                            period_count = len(latest)
                            comparison = "sum_latest_four_quarters"
                            item = latest.iloc[-1]["financial_item"]
                            note = (
                                "TTM approximation; verify quarters are "
                                "consecutive and comparable"
                            )
                else:
                    # Balance-sheet values are point-in-time, not additive.
                    latest_date = subset["period_end"].max()
                    latest = subset[subset["period_end"] == latest_date]
                    value = latest.iloc[0]["value"]
                    period_end = latest_date
                    period_count = 1
                    comparison = f"latest_{frequency}_balance_sheet"
                    item = latest.iloc[0]["financial_item"]
                    note = "Point-in-time balance-sheet value"

                rows.append({
                    "ticker": ticker,
                    "profile_field": profile_field,
                    "profile_value": profile_value,
                    "comparison": comparison,
                    "statement_item": item,
                    "period_end": (
                        period_end.date().isoformat()
                        if pd.notna(period_end) else None
                    ),
                    "period_count": period_count,
                    "statement_value": value,
                    "absolute_difference_pct": pct_difference(
                        profile_value, value
                    ),
                    "note": note,
                })

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 75)
    print("PROFILE PERIOD-ALIGNMENT AUDIT")
    print("=" * 75)
    print("Comparison rows:", len(result))

    for metric in METRICS:
        print(f"\n{metric}:")
        section = result[result["profile_field"] == metric]
        print(
            section.groupby("comparison")["absolute_difference_pct"]
            .agg(["count", "median", "mean"])
            .round(2)
            .to_string()
        )

    print("\nLargest remaining differences:")
    available = result[
        result["absolute_difference_pct"].notna()
    ].sort_values("absolute_difference_pct", ascending=False)

    print(available[[
        "ticker", "profile_field", "comparison",
        "profile_value", "statement_value", "period_end",
        "absolute_difference_pct", "note",
    ]].head(30).to_string(index=False))

    print("\nReport:", OUTPUT_PATH)
    print("Read-only audit; no source data, database, or models changed.")


if __name__ == "__main__":
    main()
