
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

SUMMARY_PERIODS = {
    "1 year", "last year", "ttm", "3 years", "5 years", "10 years"
}

PERCENT_PATTERN = re.compile(
    r"%|margin|dividend payout|growth|cfo/ebitda",
    re.IGNORECASE,
)

EPS_PATTERN = re.compile(r"\beps\b|earnings per share", re.IGNORECASE)

DERIVED_CASH_FLOW_PATTERN = re.compile(
    r"cfo/ebitda|share of operating profit converting into cash",
    re.IGNORECASE,
)

# Recognized monetary statement line items.
AMOUNT_METRICS = {
    "sales",
    "revenue",
    "expenses",
    "operating profit",
    "financing profit",
    "net profit",
    "profit before tax",
    "other income",
    "interest",
    "depreciation",
    "borrowings",
    "borrowing",
    "deposits",
    "cwip",
    "equity capital",
    "reserves",
    "fixed assets",
    "investments",
    "other assets",
    "other liabilities",
    "total assets",
    "total liabilities",
    "cash from operating activity",
    "cash from investing activity",
    "cash from financing activity",
    "free cash flow",
    "net cash flow",
}


def normalize_metric(value):
    if pd.isna(value):
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(r"\s*\+\s*$", "", value)


def normalize_period(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[:\s]+$", "", str(value).strip()).lower()


def classify(row):
    metric = row["metric_clean"]
    metric_lower = metric.lower()
    period_key = normalize_period(row["period_original"])

    # Summary periods are not individual financial statement dates.
    if metric_lower.rstrip(":") in SUMMARY_PERIODS:
        return (
            "SUMMARY_INDICATOR",
            "SUMMARY_PERIOD",
            "EXCLUDE_FROM_STATEMENT_SERIES",
        )

    if period_key in SUMMARY_PERIODS:
        return (
            "SUMMARY_INDICATOR",
            "SUMMARY_PERIOD",
            "EXCLUDE_FROM_STATEMENT_SERIES",
        )

    if DERIVED_CASH_FLOW_PATTERN.search(metric):
        return (
            "DERIVED_METRIC",
            "RATIO_OR_CONVERSION_METRIC",
            "REVIEW_FORMULA_AND_SOURCE",
        )

    if EPS_PATTERN.search(metric):
        return (
            "STATEMENT_CANDIDATE",
            "INR_PER_SHARE",
            "UNIT_LABEL_RECOGNIZED_NOT_INDEPENDENTLY_VERIFIED",
        )

    if PERCENT_PATTERN.search(metric):
        return (
            "STATEMENT_CANDIDATE",
            "PERCENT_AS_DISPLAYED",
            "UNIT_LABEL_RECOGNIZED_NOT_INDEPENDENTLY_VERIFIED",
        )

    if metric_lower in AMOUNT_METRICS:
        return (
            "STATEMENT_CANDIDATE",
            "FINANCIAL_AMOUNT",
            "VERIFY_SOURCE_UNIT_AND_SCALE",
        )

    return (
        "UNCLASSIFIED_REVIEW",
        "UNKNOWN_METRIC",
        "MANUAL_REVIEW_REQUIRED",
    )


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

    classified = df.apply(classify, axis=1)
    df["record_class"] = [item[0] for item in classified]
    df["unit_class"] = [item[1] for item in classified]
    df["validation_action"] = [item[2] for item in classified]

    df["period_end_parsed"] = pd.to_datetime(
        df["period_end"], errors="coerce"
    )

    # Summary-period observations can have a numeric value and are
    # preserved, but they are not eligible for annual time-series use.
    df["eligible_for_statement_series"] = (
        df["record_class"].eq("STATEMENT_CANDIDATE")
        & df["period_end_parsed"].notna()
    )

    df["quality_flags"] = ""

    df.loc[
        df["record_class"].eq("SUMMARY_INDICATOR"),
        "quality_flags",
    ] += "SUMMARY_PERIOD_NOT_STATEMENT_DATE;"

    df.loc[
        df["record_class"].eq("DERIVED_METRIC"),
        "quality_flags",
    ] += "DERIVED_METRIC_REQUIRES_REVIEW;"

    df.loc[
        df["record_class"].eq("UNCLASSIFIED_REVIEW"),
        "quality_flags",
    ] += "UNKNOWN_METRIC;"

    df.loc[
        df["unit_class"].eq("VERIFY_SOURCE_UNIT_AND_SCALE"),
        "quality_flags",
    ] += "UNIT_SCALE_NOT_VERIFIED;"

    df.to_csv(
        OUTPUT_DIR / "refined_metric_classification.csv",
        index=False,
    )

    metric_audit = (
        df.groupby(
            ["section", "metric_clean", "record_class", "unit_class"],
            dropna=False,
        )
        .agg(
            rows=("metric_clean", "size"),
            companies=("company", "nunique"),
            statement_series_rows=(
                "eligible_for_statement_series", "sum"
            ),
        )
        .reset_index()
    )

    metric_audit.to_csv(
        OUTPUT_DIR / "refined_metric_dictionary_audit.csv",
        index=False,
    )

    # Period coverage audit for candidate statement series.
    statement_df = df[df["eligible_for_statement_series"]].copy()

    coverage = (
        statement_df.groupby(
            ["company", "section", "metric_clean"],
            dropna=False,
        )
        .agg(
            rows=("period_end_parsed", "size"),
            distinct_periods=("period_end_parsed", "nunique"),
            first_period=("period_end_parsed", "min"),
            last_period=("period_end_parsed", "max"),
        )
        .reset_index()
    )

    coverage.to_csv(
        OUTPUT_DIR / "refined_statement_period_coverage.csv",
        index=False,
    )

    counts = df["record_class"].value_counts().to_dict()

    summary = {
        "input_rows": len(df),
        "companies": int(df["company"].nunique()),
        "reporting_basis_values": sorted(
            df["reporting_basis"].dropna().astype(str).unique().tolist()
        ),
        "classification_counts": counts,
        "statement_series_eligible_rows": int(
            df["eligible_for_statement_series"].sum()
        ),
        "unknown_metric_rows": int(
            df["record_class"].eq("UNCLASSIFIED_REVIEW").sum()
        ),
        "unit_scale_independently_verified": False,
        "cash_flow_semantics_verified": False,
        "all_source_values_preserved": True,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "REFINED_CLASSIFICATION_REQUIRES_VALIDATION",
    }

    (OUTPUT_DIR / "refined_metric_dictionary_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nREFINED CLASSIFICATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
