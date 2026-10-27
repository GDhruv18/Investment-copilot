
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "integration_candidate"
    / "fundamental_observations_candidate.parquet"
)

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}

# Candidate comparisons only. These pairs are not approved for scoring.
# Candidate order establishes a deterministic preference when labels overlap.
METRIC_PAIRS = {
    "revenue": {
        "screener": ["Revenue +", "Sales +"],
        "yahoo": ["Total Revenue"],
    },
    "net_income": {
        "screener": ["Net Profit +"],
        "yahoo": ["Net Income Common Stockholders"],
    },
    "operating_profit": {
        "screener": ["Operating Profit"],
        "yahoo": ["Operating Income"],
    },
    "eps": {
        "screener": ["EPS in Rs"],
        "yahoo": ["Basic EPS"],
    },
    "total_assets": {
        "screener": ["Total Assets"],
        "yahoo": ["Total Assets"],
    },
    "total_debt": {
        "screener": ["Borrowings +", "Borrowing"],
        "yahoo": ["Total Debt"],
    },
    "total_liabilities": {
        "screener": ["Total Liabilities"],
        "yahoo": ["Total Liabilities Net Minority Interest"],
    },
    "operating_cash_flow": {
        "screener": ["Cash from Operating Activity +"],
        "yahoo": ["Operating Cash Flow"],
    },
    "free_cash_flow": {
        "screener": ["Free Cash Flow"],
        "yahoo": ["Free Cash Flow"],
    },
}

SCREENER = "Screener.in"
YAHOO = "Yahoo Finance via yfinance"


def stop(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}")


def period_class(statement_type: object) -> str:
    """Normalize known source statement labels to annual/quarterly."""
    label = str(statement_type).strip().lower()

    if "quarter" in label:
        return "quarterly"

    # Screener's imported balance_sheet and cash_flow tables are annual.
    # This assumption must be checked against the source staging audit.
    if label in {"balance_sheet", "cash_flow"}:
        return "annual"

    if "annual" in label:
        return "annual"

    return "unknown"


def choose_candidate(
    frame: pd.DataFrame,
    candidate_names: list[str],
) -> pd.DataFrame:
    """
    Choose the first available candidate per ticker/period/class.
    Never sum or average alternative provider labels.
    """
    if frame.empty:
        return frame.copy()

    keys = ["ticker", "period_end", "period_class"]
    parts = []

    for priority, name in enumerate(candidate_names):
        subset = frame.loc[frame["metric_raw"].eq(name)].copy()
        subset["_priority"] = priority
        parts.append(subset)

    if not parts:
        return frame.iloc[0:0].copy()

    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates.sort_values(
        ["_priority", "metric_raw"],
        kind="stable",
    )

    # If the preferred metric has duplicate rows at the same key, exclude
    # that key rather than silently choosing an arbitrary value.
    selected = []
    for _, group in candidates.groupby(keys, dropna=False, sort=False):
        best_priority = group["_priority"].min()
        best = group.loc[group["_priority"].eq(best_priority)]

        if len(best) == 1:
            selected.append(best)

    if not selected:
        return candidates.iloc[0:0].drop(columns="_priority")

    return pd.concat(selected, ignore_index=True).drop(
        columns="_priority"
    )


def source_rows(
    data: pd.DataFrame,
    source: str,
    candidates: list[str],
) -> pd.DataFrame:
    frame = data.loc[
        data["source"].eq(source)
        & data["metric_raw"].isin(candidates)
    ].copy()

    frame["period_class"] = frame["statement_type"].map(period_class)

    # Unknown statement-period semantics must not be matched.
    frame = frame.loc[
        frame["period_class"].isin(["annual", "quarterly"])
    ].copy()

    return choose_candidate(frame, candidates)


def main() -> None:
    print("=" * 78)
    print("FUNDAMENTAL SOURCE RECONCILIATION — CORRECTED AUDIT")
    print("READ-ONLY | NO OUTPUT FILES | SCORING DISABLED")
    print("=" * 78)

    if not DATA_PATH.exists():
        stop(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    required = {
        "ticker", "source", "metric_raw",
        "statement_type", "period_end", "value",
    }
    missing = required - set(data.columns)
    if missing:
        stop(f"Missing required columns: {sorted(missing)}")

    for col in ["ticker", "source", "metric_raw", "statement_type"]:
        data[col] = data[col].astype("string").str.strip()

    data["period_end"] = pd.to_datetime(
        data["period_end"], errors="coerce"
    )
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    print(f"\nRows: {len(data):,}")
    print(f"Tickers: {data['ticker'].nunique():,}")
    print("Comparison unit metadata: UNVERIFIED")
    print("Currency quarantine:", ", ".join(sorted(QUARANTINED_TICKERS)))

    totals = {
        "aligned": 0,
        "sign_disagreements": 0,
        "zero_pairs": 0,
        "review_scale": 0,
        "quarantined": 0,
        "ambiguous_excluded": 0,
    }

    for metric, spec in METRIC_PAIRS.items():
        screener = source_rows(data, SCREENER, spec["screener"])
        yahoo = source_rows(data, YAHOO, spec["yahoo"])

        keys = ["ticker", "period_end", "period_class"]

        print("\n" + "-" * 78)
        print(f"METRIC: {metric}")
        print(
            f"Selected Screener rows/tickers: {len(screener):,}/"
            f"{screener['ticker'].nunique() if not screener.empty else 0}"
        )
        print(
            f"Selected Yahoo rows/tickers: {len(yahoo):,}/"
            f"{yahoo['ticker'].nunique() if not yahoo.empty else 0}"
        )

        if screener.empty or yahoo.empty:
            print("RESULT: no source rows available for comparison")
            continue

        # Defensive duplicate-key check after candidate selection.
        s_dup = screener.duplicated(keys, keep=False)
        y_dup = yahoo.duplicated(keys, keep=False)

        ambiguous_keys = set(
            map(tuple, screener.loc[s_dup, keys].to_numpy())
        ) | set(
            map(tuple, yahoo.loc[y_dup, keys].to_numpy())
        )

        totals["ambiguous_excluded"] += len(ambiguous_keys)

        if ambiguous_keys:
            print(
                f"Ambiguous ticker-period keys excluded: "
                f"{len(ambiguous_keys):,}"
            )

        screener = screener.loc[~s_dup].copy()
        yahoo = yahoo.loc[~y_dup].copy()

        joined = screener[
            keys + ["metric_raw", "value"]
        ].merge(
            yahoo[keys + ["metric_raw", "value"]],
            on=keys,
            how="inner",
            suffixes=("_screener", "_yahoo"),
            validate="one_to_one",
        )

        joined = joined.dropna(
            subset=["value_screener", "value_yahoo"]
        ).copy()

        if joined.empty:
            print("RESULT: no aligned numeric pairs")
            continue

        joined["ticker"] = joined["ticker"].astype(str)
        joined["quarantined"] = joined["ticker"].isin(
            QUARANTINED_TICKERS
        )

        s = joined["value_screener"].astype(float)
        y = joined["value_yahoo"].astype(float)

        joined["both_nonzero"] = (s != 0) & (y != 0)
        joined["sign_disagreement"] = (
            (s < 0) & (y > 0)
        ) | ((s > 0) & (y < 0))

        joined["abs_ratio"] = np.where(
            joined["both_nonzero"],
            np.maximum(np.abs(s), np.abs(y))
            / np.minimum(np.abs(s), np.abs(y)),
            np.nan,
        )

        joined["yahoo_to_screener_ratio"] = np.where(
            joined["both_nonzero"],
            np.abs(y) / np.abs(s),
            np.nan,
        )

        # Exclude quarantined tickers from scale diagnostics.
        diagnostic = joined.loc[
            ~joined["quarantined"]
            & joined["both_nonzero"]
            & ~joined["sign_disagreement"]
        ].copy()

        print(f"Aligned numeric pairs: {len(joined):,}")
        print(
            f"Sign disagreements: "
            f"{int(joined['sign_disagreement'].sum()):,}"
        )
        print(
            f"Pairs with at least one zero: "
            f"{int((~joined['both_nonzero']).sum()):,}"
        )
        print(
            f"Quarantined pairs: "
            f"{int(joined['quarantined'].sum()):,}"
        )

        totals["aligned"] += len(joined)
        totals["sign_disagreements"] += int(
            joined["sign_disagreement"].sum()
        )
        totals["zero_pairs"] += int((~joined["both_nonzero"]).sum())
        totals["quarantined"] += int(joined["quarantined"].sum())

        if diagnostic.empty:
            print("Scale diagnostic: insufficient eligible pairs")
            continue

        ratios = diagnostic["yahoo_to_screener_ratio"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        log_ratios = np.log10(ratios[ratios > 0])

        if log_ratios.empty:
            print("Scale diagnostic: no valid positive ratios")
            continue

        median_log = float(log_ratios.median())
        median_ratio = float(10 ** median_log)
        near_power = round(median_log)

        print(
            f"Median |Yahoo| / |Screener|: {median_ratio:.4g}"
        )
        print(
            f"Median log10 ratio: {median_log:.3f}"
        )
        print(
            f"Nearest power-of-ten scale factor: "
            f"10^{near_power}"
        )

        # Triage only; this does not prove the units.
        scale_flag = np.abs(log_ratios - near_power) <= 0.15
        flagged_count = int(scale_flag.sum())
        totals["review_scale"] += flagged_count

        print(
            f"Pairs near that scale factor (within 0.15 log10): "
            f"{flagged_count:,}/{len(log_ratios):,}"
        )

        print("Largest absolute-magnitude ratios (diagnostic sample):")
        sample = diagnostic.assign(
            ratio_log=diagnostic["abs_ratio"].map(
                lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan
            )
        ).sort_values("ratio_log", ascending=False).head(5)

        for row in sample.itertuples(index=False):
            print(
                f"  {row.ticker} | {row.period_end.date()} | "
                f"{row.metric_raw_screener} vs {row.metric_raw_yahoo} | "
                f"Screener={row.value_screener:,.5g} | "
                f"Yahoo={row.value_yahoo:,.5g} | "
                f"absolute ratio={row.abs_ratio:.4g}"
            )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Aligned pairs: {totals['aligned']:,}")
    print(f"Sign disagreements: {totals['sign_disagreements']:,}")
    print(f"Pairs containing zero: {totals['zero_pairs']:,}")
    print(f"Quarantined pairs: {totals['quarantined']:,}")
    print(
        f"Pair-level scale-factor flags across metrics: "
        f"{totals['review_scale']:,}"
    )
    print(
        f"Ambiguous keys excluded: {totals['ambiguous_excluded']:,}"
    )

    print("\nCAUTIONS")
    print("- A scale factor is evidence to investigate, not a unit conversion.")
    print("- This audit does not assume either provider is correct.")
    print("- Revenue and profit definitions may differ across providers.")
    print("- A match on date does not prove identical accounting periods.")
    print("- No currency conversion or unit scaling has been applied.")
    print("- No metrics are approved for fundamental scoring.")
    print("\nRESULT: corrected diagnostic audit completed; no files written.")
    print("=" * 78)


if __name__ == "__main__":
    main()
