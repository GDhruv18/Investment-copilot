
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCREENER_PATH = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100"
    / "consolidated_financials_long.csv"
)

YAHOO_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "yfinance" / "yfinance_statement_observations.parquet"
)

QUARANTINED = {"HCLTECH.NS", "INFY.NS"}

# Metric names are explicit; no automatic accounting-equivalence assumptions.
COMPARISONS = [
    {
        "name": "Annual total assets",
        "section": "Balance Sheet",
        "metric": "Total Assets",
        "statement": "annual_balance_sheet",
        "items": ["Total Assets"],
    },
    {
        "name": "Annual borrowings vs total debt",
        "section": "Balance Sheet",
        "metric": "Borrowings +",
        "statement": "annual_balance_sheet",
        "items": ["Total Debt"],
    },
    {
        "name": "Annual total liabilities",
        "section": "Balance Sheet",
        "metric": "Total Liabilities",
        "statement": "annual_balance_sheet",
        "items": ["Total Liabilities Net Minority Interest"],
    },
    {
        "name": "Annual operating cash flow",
        "section": "Cash Flow",
        "metric": "Cash from Operating Activity +",
        "statement": "annual_cash_flow",
        "items": ["Operating Cash Flow"],
    },
    {
        "name": "Annual free cash flow",
        "section": "Cash Flow",
        "metric": "Free Cash Flow",
        "statement": "annual_cash_flow",
        "items": ["Free Cash Flow"],
    },
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SCREENER_PATH.exists():
        raise FileNotFoundError(f"Missing file: {SCREENER_PATH}")
    if not YAHOO_PATH.exists():
        raise FileNotFoundError(f"Missing file: {YAHOO_PATH}")

    s = pd.read_csv(SCREENER_PATH)
    y = pd.read_parquet(YAHOO_PATH)

    required_s = {
        "ticker", "section", "metric", "period_end",
        "numeric_value", "unit", "unit_verified",
    }
    required_y = {
        "ticker", "statement_type", "financial_item",
        "period_end", "value",
    }

    if missing := required_s - set(s.columns):
        raise ValueError(f"Screener columns missing: {sorted(missing)}")
    if missing := required_y - set(y.columns):
        raise ValueError(f"Yahoo columns missing: {sorted(missing)}")

    for frame, columns in (
        (s, ["period_end"]),
        (y, ["period_end"]),
    ):
        frame["period_end"] = pd.to_datetime(
            frame["period_end"], errors="coerce"
        ).dt.normalize()

    s["ticker"] = s["ticker"].astype(str).str.strip()
    s["section"] = s["section"].astype(str).str.strip()
    s["metric"] = s["metric"].astype(str).str.strip()
    s["numeric_value"] = pd.to_numeric(
        s["numeric_value"], errors="coerce"
    )

    y["ticker"] = y["ticker"].astype(str).str.strip()
    y["statement_type"] = y["statement_type"].astype(str).str.strip()
    y["financial_item"] = y["financial_item"].astype(str).str.strip()
    y["value"] = pd.to_numeric(y["value"], errors="coerce")

    return s, y


def audit_one(s: pd.DataFrame, y: pd.DataFrame, spec: dict) -> None:
    left = s.loc[
        s["section"].eq(spec["section"])
        & s["metric"].eq(spec["metric"]),
        ["ticker", "period_end", "numeric_value", "unit", "unit_verified"],
    ].copy()

    right_all = y.loc[
        y["statement_type"].eq(spec["statement"])
        & y["financial_item"].isin(spec["items"]),
        ["ticker", "period_end", "financial_item", "value"],
    ].copy()

    print(f"\n{spec['name']}")
    print("-" * len(spec["name"]))

    for item in spec["items"]:
        right = right_all.loc[
            right_all["financial_item"].eq(item)
        ].copy()

        # If multiple rows share the same join key, report and skip them.
        # We must not silently duplicate or aggregate statement observations.
        left_dup = left.duplicated(
            ["ticker", "period_end"], keep=False
        )
        right_dup = right.duplicated(
            ["ticker", "period_end"], keep=False
        )

        left_keys = left.loc[
            ~left_dup, ["ticker", "period_end"]
        ]
        right_keys = right.loc[
            ~right_dup, ["ticker", "period_end"]
        ]

        duplicate_left = int(left_dup.sum())
        duplicate_right = int(right_dup.sum())

        merged = left.loc[~left_dup].merge(
            right.loc[~right_dup],
            on=["ticker", "period_end"],
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["numeric_value", "value"])

        merged["quarantined"] = merged["ticker"].isin(QUARANTINED)

        positive = merged.loc[
            ~merged["quarantined"]
            & merged["numeric_value"].gt(0)
            & merged["value"].gt(0)
        ].copy()

        ratios = (
            positive["value"] / positive["numeric_value"]
        ).replace([np.inf, -np.inf], np.nan).dropna()

        print(f"Yahoo item: {item}")
        print(f"  Screener observations: {len(left):,}")
        print(f"  Yahoo observations: {len(right):,}")
        print(f"  Duplicate Screener key rows excluded: {duplicate_left:,}")
        print(f"  Duplicate Yahoo key rows excluded: {duplicate_right:,}")
        print(f"  Matched observations: {len(merged):,}")
        print(f"  Matched tickers: {merged['ticker'].nunique():,}")
        print(
            "  Quarantined matches: "
            f"{int(merged['quarantined'].sum()):,}"
        )
        print(
            "  Unit-verified Screener matches: "
            f"{int(merged['unit_verified'].fillna(False).astype(bool).sum()):,}"
            f"/{len(merged):,}"
        )

        if ratios.empty:
            print("  Positive ratio pairs: 0")
            continue

        log_ratio = np.log10(ratios)
        print(f"  Positive ratio pairs: {len(ratios):,}")
        print(f"  Median Yahoo/Screener ratio: {ratios.median():,.6g}")
        print(f"  Median log10 ratio: {log_ratio.median():.4f}")
        print(
            "  Within 0.25 log10 of 10^7: "
            f"{int((log_ratio - 7).abs().le(0.25).sum()):,}"
            f"/{len(ratios):,}"
        )
        print(
            "  Within 0.25 log10 of 1: "
            f"{int(log_ratio.abs().le(0.25).sum()):,}"
            f"/{len(ratios):,}"
        )

        positive["ratio"] = positive["value"] / positive["numeric_value"]
        positive["log10_ratio"] = np.log10(positive["ratio"])
        positive["distance_to_1e7"] = (
            positive["log10_ratio"] - 7
        ).abs()

        print("  Largest deviations from 10^7:")
        for row in positive.nlargest(
            3, "distance_to_1e7"
        ).itertuples(index=False):
            print(
                f"    {row.ticker} {row.period_end.date()} | "
                f"Screener={row.numeric_value:g} | "
                f"Yahoo={row.value:g} | ratio={row.ratio:.6g}"
            )


def main() -> None:
    screener, yahoo = load_inputs()

    print("=" * 72)
    print("BALANCE SHEET & CASH FLOW SCALE AUDIT — READ ONLY")
    print("=" * 72)
    print(f"Screener rows: {len(screener):,}")
    print(f"Yahoo statement rows: {len(yahoo):,}")
    print("Raw data will not be modified.")
    print("No normalized values or files will be created.")
    print("Scoring eligibility remains unchanged.")

    for spec in COMPARISONS:
        audit_one(screener, yahoo, spec)

    print("\nAudit complete. Interpret scale and accounting definitions separately.")


if __name__ == "__main__":
    main()
