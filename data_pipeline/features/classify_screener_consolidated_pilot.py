
from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_pilot"
    / "consolidated_financials_long.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "clean" / "fundamentals"
    / "screener_consolidated_pilot"
)

# These are labels observed in the pilot, not statement periods.
SUMMARY_LABELS = {
    "1 year",
    "last year",
    "ttm",
    "3 years",
    "5 years",
    "10 years",
}

CASH_FLOW_DERIVED = re.compile(
    r"cfo/ebitda|share of operating profit converting into cash",
    re.IGNORECASE,
)

PERCENT_METRICS = re.compile(
    r"%|margin|dividend payout|growth|cfo/ebitda",
    re.IGNORECASE,
)

EPS_METRICS = re.compile(
    r"\beps\b|earnings per share",
    re.IGNORECASE,
)

# Common statement metrics that are generally amounts.
# This is a classification aid, not proof of units.
AMOUNT_METRICS = re.compile(
    r"sales|revenue|expenses|operating profit|financing profit|"
    r"net profit|profit before tax|other income|interest|"
    r"depreciation|borrowings|equity capital|reserves|"
    r"liabilities|assets|investments|cash flow|free cash flow",
    re.IGNORECASE,
)


def normalize_metric(value):
    if pd.isna(value):
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    # Remove display marker at the end, e.g. "Sales +".
    value = re.sub(r"\s*\+\s*$", "", value)
    return value


def classify_record(row):
    metric = row["metric_clean"]
    period = str(row["period_original"]).strip()
    section = row["section"]

    period_key = re.sub(r"[:\s]+$", "", period).lower()

    # Summary labels can appear in the metric column because of
    # the HTML table layout. Preserve, but do not use as statements.
    if metric.lower().rstrip(":") in SUMMARY_LABELS:
        return "SUMMARY_INDICATOR", "NON_STATEMENT_PERIOD_LABEL"

    if period_key in SUMMARY_LABELS:
        return "SUMMARY_INDICATOR", "NON_STATEMENT_PERIOD_LABEL"

    if CASH_FLOW_DERIVED.search(metric):
        return "DERIVED_CASH_FLOW_METRIC", "RATIO_REQUIRES_REVIEW"

    if EPS_METRICS.search(metric):
        return "STATEMENT_CANDIDATE", "INR_PER_SHARE"

    if PERCENT_METRICS.search(metric):
        return "STATEMENT_CANDIDATE", "PERCENT_AS_DISPLAYED"

    if AMOUNT_METRICS.search(metric):
        return "STATEMENT_CANDIDATE", "AMOUNT_UNIT_REQUIRES_VERIFICATION"

    # Keep unfamiliar metrics instead of dropping them.
    return "UNCLASSIFIED_REVIEW", "METRIC_REQUIRES_REVIEW"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "company", "reporting_basis", "section", "metric",
        "period_original", "period_end", "raw_value",
        "numeric_value", "source_url",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["metric_original"] = df["metric"]
    df["metric_clean"] = df["metric"].apply(normalize_metric)

    classified = df.apply(classify_record, axis=1)
    df["record_class"] = [x[0] for x in classified]
    df["unit_classification"] = [x[1] for x in classified]

    df["period_end_parsed"] = pd.to_datetime(
        df["period_end"], errors="coerce"
    )

    df["classification_flags"] = ""

    df.loc[
        df["record_class"].eq("SUMMARY_INDICATOR"),
        "classification_flags",
    ] += "SUMMARY_NOT_STATEMENT;"

    df.loc[
        df["record_class"].eq("DERIVED_CASH_FLOW_METRIC"),
        "classification_flags",
    ] += "DERIVED_CASH_FLOW_REVIEW;"

    df.loc[
        df["record_class"].eq("UNCLASSIFIED_REVIEW"),
        "classification_flags",
    ] += "UNCLASSIFIED_METRIC;"

    df.loc[
        df["period_end_parsed"].isna()
        & df["record_class"].eq("STATEMENT_CANDIDATE"),
        "classification_flags",
    ] += "PERIOD_REQUIRES_REVIEW;"

    # Preserve all rows in the master classification output.
    df.to_csv(
        OUTPUT_DIR / "consolidated_metric_classification.csv",
        index=False,
    )

    statement_candidates = df[
        df["record_class"].eq("STATEMENT_CANDIDATE")
    ].copy()

    summary_indicators = df[
        df["record_class"].eq("SUMMARY_INDICATOR")
    ].copy()

    derived_cash_flow = df[
        df["record_class"].eq("DERIVED_CASH_FLOW_METRIC")
    ].copy()

    unclassified = df[
        df["record_class"].eq("UNCLASSIFIED_REVIEW")
    ].copy()

    statement_candidates.to_csv(
        OUTPUT_DIR / "statement_candidates.csv", index=False
    )
    summary_indicators.to_csv(
        OUTPUT_DIR / "summary_indicators.csv", index=False
    )
    derived_cash_flow.to_csv(
        OUTPUT_DIR / "derived_cash_flow_review.csv", index=False
    )
    unclassified.to_csv(
        OUTPUT_DIR / "unclassified_metrics_review.csv", index=False
    )

    class_counts = (
        df["record_class"].value_counts(dropna=False).to_dict()
    )

    metric_audit = (
        df.groupby(
            ["section", "metric_clean", "record_class",
             "unit_classification"],
            dropna=False,
        )
        .agg(
            rows=("metric_clean", "size"),
            companies=("company", "nunique"),
        )
        .reset_index()
    )

    metric_audit.to_csv(
        OUTPUT_DIR / "metric_classification_audit.csv",
        index=False,
    )

    summary = {
        "input_rows": len(df),
        "companies": int(df["company"].nunique()),
        "reporting_basis_values": sorted(
            df["reporting_basis"].dropna().astype(str).unique().tolist()
        ),
        "classification_counts": class_counts,
        "statement_candidate_rows": len(statement_candidates),
        "summary_indicator_rows": len(summary_indicators),
        "derived_cash_flow_rows": len(derived_cash_flow),
        "unclassified_rows": len(unclassified),
        "amount_units_independently_verified": False,
        "cash_flow_period_semantics_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "CLASSIFIED_REQUIRES_FINANCIAL_VALIDATION",
    }

    (OUTPUT_DIR / "classification_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCLASSIFICATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
