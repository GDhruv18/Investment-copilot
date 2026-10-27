
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SCREENER_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fundamentals"
    / "screener_consolidated_100"
    / "consolidated_financials_long.csv"
)

YAHOO_PATH = (
    ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "yfinance"
    / "yfinance_statement_observations.parquet"
)

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}

# Each comparison is deliberately explicit.
# Alternative Yahoo metrics are audited separately to avoid
# silently treating different accounting definitions as equivalent.
COMPARISONS = [
    {
        "name": "Quarterly revenue",
        "screener_section": "Quarterly Results",
        "screener_metric": "Sales +",
        "yahoo_statement": "quarterly_income_statement",
        "yahoo_items": ["Total Revenue", "Operating Revenue"],
        "kind": "amount",
    },
    {
        "name": "Quarterly net profit vs net income including NCI",
        "screener_section": "Quarterly Results",
        "screener_metric": "Net Profit +",
        "yahoo_statement": "quarterly_income_statement",
        "yahoo_items": ["Net Income Including Noncontrolling Interests"],
        "kind": "amount",
    },
    {
        "name": "Quarterly net profit vs common-stockholder income",
        "screener_section": "Quarterly Results",
        "screener_metric": "Net Profit +",
        "yahoo_statement": "quarterly_income_statement",
        "yahoo_items": ["Net Income Common Stockholders"],
        "kind": "amount",
    },
    {
        "name": "Quarterly EPS",
        "screener_section": "Quarterly Results",
        "screener_metric": "EPS in Rs",
        "yahoo_statement": "quarterly_income_statement",
        "yahoo_items": ["Basic EPS"],
        "kind": "per_share",
    },
    {
        "name": "Annual revenue",
        "screener_section": "Annual Profit and Loss",
        "screener_metric": "Sales +",
        "yahoo_statement": "annual_income_statement",
        "yahoo_items": ["Total Revenue", "Operating Revenue"],
        "kind": "amount",
    },
    {
        "name": "Annual net profit vs net income including NCI",
        "screener_section": "Annual Profit and Loss",
        "screener_metric": "Net Profit +",
        "yahoo_statement": "annual_income_statement",
        "yahoo_items": ["Net Income Including Noncontrolling Interests"],
        "kind": "amount",
    },
    {
        "name": "Annual EPS",
        "screener_section": "Annual Profit and Loss",
        "screener_metric": "EPS in Rs",
        "yahoo_statement": "annual_income_statement",
        "yahoo_items": ["Basic EPS"],
        "kind": "per_share",
    },
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SCREENER_PATH.exists():
        raise FileNotFoundError(f"Missing Screener input: {SCREENER_PATH}")

    if not YAHOO_PATH.exists():
        raise FileNotFoundError(f"Missing Yahoo input: {YAHOO_PATH}")

    screener = pd.read_csv(SCREENER_PATH)
    yahoo = pd.read_parquet(YAHOO_PATH)

    screener_required = {
        "ticker",
        "section",
        "metric",
        "period_end",
        "numeric_value",
        "unit",
        "unit_verified",
    }
    yahoo_required = {
        "ticker",
        "statement_type",
        "financial_item",
        "period_end",
        "value",
    }

    missing_screener = screener_required - set(screener.columns)
    missing_yahoo = yahoo_required - set(yahoo.columns)

    if missing_screener:
        raise ValueError(
            f"Screener columns missing: {sorted(missing_screener)}"
        )

    if missing_yahoo:
        raise ValueError(
            f"Yahoo columns missing: {sorted(missing_yahoo)}"
        )

    screener["ticker"] = screener["ticker"].astype(str).str.strip()
    screener["section"] = screener["section"].astype(str).str.strip()
    screener["metric"] = screener["metric"].astype(str).str.strip()
    screener["period_end"] = pd.to_datetime(
        screener["period_end"], errors="coerce"
    ).dt.normalize()
    screener["numeric_value"] = pd.to_numeric(
        screener["numeric_value"], errors="coerce"
    )

    yahoo["ticker"] = yahoo["ticker"].astype(str).str.strip()
    yahoo["statement_type"] = (
        yahoo["statement_type"].astype(str).str.strip()
    )
    yahoo["financial_item"] = (
        yahoo["financial_item"].astype(str).str.strip()
    )
    yahoo["period_end"] = pd.to_datetime(
        yahoo["period_end"], errors="coerce"
    ).dt.normalize()
    yahoo["value"] = pd.to_numeric(yahoo["value"], errors="coerce")

    return screener, yahoo


def summarize_comparison(
    screener: pd.DataFrame,
    yahoo: pd.DataFrame,
    spec: dict,
) -> None:
    s = screener.loc[
        screener["section"].eq(spec["screener_section"])
        & screener["metric"].eq(spec["screener_metric"]),
        ["ticker", "period_end", "numeric_value", "unit", "unit_verified"],
    ].copy()

    y = yahoo.loc[
        yahoo["statement_type"].eq(spec["yahoo_statement"])
        & yahoo["financial_item"].isin(spec["yahoo_items"]),
        ["ticker", "period_end", "financial_item", "value"],
    ].copy()

    # Multiple Yahoo aliases may represent the same underlying metric.
    # Keep each alias distinct and report its result independently.
    for item in spec["yahoo_items"]:
        yi = y.loc[y["financial_item"].eq(item)].copy()

        merged = s.merge(
            yi,
            on=["ticker", "period_end"],
            how="inner",
            validate="many_to_many",
        )

        merged = merged.dropna(
            subset=["numeric_value", "value"]
        )

        if merged.empty:
            print(f"\n{spec['name']} | Yahoo item: {item}")
            print("  Matched observations: 0")
            continue

        merged["is_quarantined"] = merged["ticker"].isin(
            QUARANTINED_TICKERS
        )

        # Only use positive, non-zero pairs to estimate scale.
        # Signed/zero values remain visible in coverage diagnostics.
        scale_pairs = merged.loc[
            (~merged["is_quarantined"])
            & merged["numeric_value"].gt(0)
            & merged["value"].gt(0)
        ].copy()

        scale_pairs["ratio"] = (
            scale_pairs["value"] / scale_pairs["numeric_value"]
        )

        ratios = scale_pairs["ratio"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        print(f"\n{spec['name']} | Yahoo item: {item}")
        print(f"  Matched observations: {len(merged):,}")
        print(
            "  Unique matched tickers: "
            f"{merged['ticker'].nunique():,}"
        )
        print(
            "  Quarantined matched observations: "
            f"{int(merged['is_quarantined'].sum()):,}"
        )
        print(
            "  Screener unit verified: "
            f"{int(merged['unit_verified'].fillna(False).astype(bool).sum()):,}"
            f"/{len(merged):,}"
        )

        if ratios.empty:
            print("  Positive scale-comparison pairs: 0")
        else:
            log_ratio = np.log10(ratios)
            print(f"  Positive scale-comparison pairs: {len(ratios):,}")
            print(
                "  Median Yahoo/Screener ratio: "
                f"{ratios.median():,.6g}"
            )
            print(
                "  Median log10 ratio: "
                f"{log_ratio.median():.4f}"
            )
            print(
                "  Ratios within 0.25 log10 of 1: "
                f"{int(log_ratio.abs().le(0.25).sum()):,}"
                f"/{len(ratios):,}"
            )
            print(
                "  Ratios within 0.25 log10 of 10^7: "
                f"{int((log_ratio - 7).abs().le(0.25).sum()):,}"
                f"/{len(ratios):,}"
            )

        if spec["kind"] == "per_share":
            valid = merged.loc[
                merged["numeric_value"].ne(0)
            ].copy()

            if not valid.empty:
                valid["relative_difference"] = (
                    (valid["value"] - valid["numeric_value"]).abs()
                    / valid["numeric_value"].abs()
                )
                print(
                    "  EPS within 1%: "
                    f"{int(valid['relative_difference'].le(0.01).sum()):,}"
                    f"/{len(valid):,}"
                )

        # Show a few mismatches for interpretation, not automatic rejection.
        if spec["kind"] == "amount" and not scale_pairs.empty:
            scale_pairs["log10_ratio"] = np.log10(
                scale_pairs["ratio"]
            )
            scale_pairs["distance_to_1e7"] = (
                scale_pairs["log10_ratio"] - 7
            ).abs()

            examples = scale_pairs.sort_values(
                "distance_to_1e7", ascending=False
            ).head(3)

            if not examples.empty:
                print("  Examples furthest from 10^7:")
                for row in examples.itertuples(index=False):
                    print(
                        f"    {row.ticker} {row.period_end.date()} | "
                        f"Screener={row.numeric_value:g} | "
                        f"Yahoo={row.value:g} | "
                        f"ratio={row.ratio:.6g}"
                    )


def main() -> None:
    screener, yahoo = load_inputs()

    print("=" * 72)
    print("FUNDAMENTAL SCALE RECONCILIATION — READ ONLY")
    print("=" * 72)
    print(f"Screener rows: {len(screener):,}")
    print(f"Yahoo statement rows: {len(yahoo):,}")
    print(f"Screener tickers: {screener['ticker'].nunique():,}")
    print(f"Yahoo tickers: {yahoo['ticker'].nunique():,}")
    print(
        "Quarantined tickers: "
        + ", ".join(sorted(QUARANTINED_TICKERS))
    )
    print("No input files will be modified.")
    print("No normalized values will be created.")
    print("No scoring eligibility will be changed.")

    for spec in COMPARISONS:
        summarize_comparison(screener, yahoo, spec)

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE — INTERPRET RESULTS BEFORE ANY NORMALIZATION")
    print("=" * 72)


if __name__ == "__main__":
    main()
