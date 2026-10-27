
from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "clean" / "fundamentals"
    / "screener_consolidated_pilot" / "separated"
    / "all_records_with_table_roles.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_metric_review"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_summary_metric(value):
    """Remove pandas duplicate-column suffixes from summary metric labels."""
    value = str(value).strip()
    return re.sub(r"\.\d+$", "", value)


def classify_metric(metric):
    """Initial semantic screen only; not final financial validation."""
    m = str(metric).strip().lower()

    if "%" in m or "margin" in m or "payout" in m:
        return "PERCENT_OR_MARGIN_REVIEW"

    if "eps" in m or "per share" in m:
        return "INR_PER_SHARE_REVIEW"

    if any(term in m for term in [
        "growth", "cagr", "return on equity", "roe",
        "return on capital", "roce", "share of operating profit"
    ]):
        return "RATIO_OR_GROWTH_REVIEW"

    if any(term in m for term in [
        "sales", "revenue", "expense", "profit", "income",
        "interest", "tax", "depreciation", "borrow", "debt",
        "asset", "liabilit", "cash", "deposit", "investment",
        "working capital", "equity", "cwip", "financing"
    ]):
        return "AMOUNT_UNIT_CONFIRMATION_REQUIRED"

    return "UNCLASSIFIED_REVIEW"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [
        "company", "section", "metric",
        "period_original", "raw_value", "table_index"
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Preserve the source columns. These canonical columns correct the
    # metric/period orientation specifically for annual P&L summary tables.
    df["canonical_metric"] = df["metric"].astype(str).str.strip()
    df["canonical_period"] = (
        df["period_original"].astype(str).str.strip()
    )
    df["canonicalization_rule"] = "SOURCE_METRIC_AND_PERIOD"

    is_annual_pl_summary = (
        df["section"].astype(str).str.strip().eq("Annual Profit and Loss")
        & pd.to_numeric(df["table_index"], errors="coerce").isin([1, 2, 3, 4])
    )

    # In these four summary tables, extraction placed the period in metric
    # and the metric name in period_original. Swap them in canonical fields.
    df.loc[is_annual_pl_summary, "canonical_metric"] = (
        df.loc[is_annual_pl_summary, "period_original"]
        .map(clean_summary_metric)
    )
    df.loc[is_annual_pl_summary, "canonical_period"] = (
        df.loc[is_annual_pl_summary, "metric"].astype(str).str.strip()
    )
    df.loc[is_annual_pl_summary, "canonicalization_rule"] = (
        "SWAPPED_ANNUAL_PL_SUMMARY_FIELDS"
    )

    # Keep a complete corrected master, including every original column.
    master_path = OUTPUT_DIR / "corrected_metric_period_master.csv"
    df.to_csv(master_path, index=False)

    # Make one row per corrected section/metric combination.
    records = []

    for (section, metric), group in df.groupby(
        ["section", "canonical_metric"], dropna=False
    ):
        sample_rows = group.head(5)

        units = (
            sorted(group["unit"].dropna().astype(str).unique().tolist())
            if "unit" in group.columns else []
        )

        roles = (
            sorted(group["table_role"].dropna().astype(str).unique().tolist())
            if "table_role" in group.columns else []
        )

        records.append({
            "section": section,
            "canonical_metric": metric,
            "row_count": int(len(group)),
            "company_count": int(group["company"].nunique()),
            "distinct_period_count": int(
                group["canonical_period"].nunique()
            ),
            "observed_unit_labels": " | ".join(units),
            "table_roles": " | ".join(roles),
            "sample_company_period_values": " || ".join(
                f"{row['company']}: {row['canonical_period']} = "
                f"{row['raw_value']}"
                for _, row in sample_rows.iterrows()
            ),
            "unit_verified": False,
            "metric_semantics_verified": False,
            "review_status": "MANUAL_SOURCE_VERIFICATION_REQUIRED",
        })

    review = pd.DataFrame(records).sort_values(
        ["section", "canonical_metric"], kind="stable"
    )

    review.to_csv(
        OUTPUT_DIR / "metric_dictionary_review.csv", index=False
    )

    # Focused review file for the four corrected summary tables.
    df.loc[is_annual_pl_summary].to_csv(
        OUTPUT_DIR / "annual_pl_summary_corrected.csv", index=False
    )

    # Explicitly audit how many records were corrected.
    correction_audit = (
        df.groupby(
            ["section", "table_index", "canonicalization_rule"],
            dropna=False
        )
        .agg(
            rows=("company", "size"),
            companies=("company", "nunique"),
            canonical_metrics=("canonical_metric", "nunique"),
            canonical_periods=("canonical_period", "nunique"),
        )
        .reset_index()
    )

    correction_audit.to_csv(
        OUTPUT_DIR / "canonicalization_audit.csv", index=False
    )

    summary = {
        "input_rows": int(len(df)),
        "companies": int(df["company"].nunique()),
        "corrected_annual_pl_summary_rows": int(
            is_annual_pl_summary.sum()
        ),
        "canonical_section_metric_pairs": int(len(review)),
        "all_original_columns_preserved": True,
        "all_rows_preserved": True,
        "unit_verified": False,
        "metric_semantics_verified": False,
        "approved_for_recommendation_scoring": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "CANONICAL_METRIC_PERIODS_CREATED_REVIEW_REQUIRED",
    }

    with open(
        OUTPUT_DIR / "metric_review_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nCORRECTED METRIC REVIEW CREATED")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
