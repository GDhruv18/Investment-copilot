
from pathlib import Path
import json
import re

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_pilot"
    / "screener_financials_pilot_long.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "screener_pilot"
)

CLEAN_FILE = OUTPUT_DIR / "screener_financials_clean.csv"
SUMMARY_FILE = OUTPUT_DIR / "screener_cleaning_audit.csv"
JSON_FILE = OUTPUT_DIR / "screener_cleaning_summary.json"


REQUIRED_COLUMNS = [
    "company",
    "screener_slug",
    "section",
    "section_id",
    "metric",
    "period",
    "raw_value",
    "source_url",
    "source",
    "unit",
    "table_index",
]

# Summary sections must not be mixed with statement observations.
SUMMARY_METRIC_PATTERN = re.compile(
    r"growth|cagr|return on equity|return on capital|"
    r"stock price|high price|low price|market price",
    re.IGNORECASE,
)

# Keep percentage metrics as percentages. Do not convert them
# into ratios or decimal fractions in this cleaning stage.
PERCENT_PATTERN = re.compile(r"%")

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def clean_metric_name(value):
    """Remove whitespace and trailing display markers."""
    if pd.isna(value):
        return None

    metric = str(value).strip()
    metric = re.sub(r"\s+", " ", metric)
    metric = re.sub(r"\s*\+\s*$", "", metric)
    return metric or None


def parse_period(value):
    """
    Convert month-year labels such as 'Jun 2026' to
    YYYY-MM-30/31 month-end dates.

    Non-date labels such as '10 Years' remain unparsed.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = re.fullmatch(
        r"([A-Za-z]{3,9})\s+(\d{4})",
        text,
    )

    if not match:
        return None

    month_text = match.group(1).lower()[:3]
    year = int(match.group(2))
    month = MONTHS.get(month_text)

    if month is None:
        return None

    return (
        pd.Timestamp(year=year, month=month, day=1)
        + pd.offsets.MonthEnd(0)
    ).date().isoformat()


def parse_numeric(value):
    """
    Parse numeric values without changing their scale.

    Examples:
      '122,627' -> 122627
      '14%'     -> 14
      '7.11'    -> 7.11

    Percentages remain in percentage-point units.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text in {"", "-", "--", "N/A", "NA", "None"}:
        return None

    text = text.replace(",", "").replace("₹", "")
    text = text.replace("Rs.", "").replace("Rs", "").strip()

    # Parentheses commonly represent negative accounting values.
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()

    text = text.replace("%", "").strip()

    try:
        return float(text)
    except ValueError:
        return None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run the pilot extraction script first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, dtype=str)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    original_rows = len(df)

    # Preserve original fields and add normalized fields.
    df["metric_original"] = df["metric"]
    df["period_original"] = df["period"]
    df["raw_value_original"] = df["raw_value"]

    df["metric_clean"] = df["metric"].apply(clean_metric_name)
    df["period_end"] = df["period"].apply(parse_period)
    df["numeric_value"] = df["raw_value"].apply(parse_numeric)

    df["is_percentage_metric"] = (
        df["metric_clean"].fillna("").str.contains(
            PERCENT_PATTERN
        )
        | df["raw_value"].fillna("").str.contains("%", regex=False)
    )

    df["is_summary_metric"] = (
        df["metric_clean"].fillna("").str.contains(
            SUMMARY_METRIC_PATTERN
        )
    )

    df["is_statement_period"] = df["period_end"].notna()

    # The source does not reliably establish reporting basis
    # in the extracted CSV. Do not infer it.
    df["reporting_basis"] = "UNVERIFIED"

    # Unit is preserved as supplied by the extraction script.
    # Do not assume every section is in INR crores.
    df["unit_status"] = "VERIFY_BY_SECTION"

    # Mark observations that require review rather than deleting them.
    df["quality_flags"] = ""

    df.loc[
        df["metric_clean"].isna(),
        "quality_flags",
    ] += "MISSING_METRIC;"

    df.loc[
        df["numeric_value"].isna(),
        "quality_flags",
    ] += "NON_NUMERIC_OR_MISSING_VALUE;"

    df.loc[
        ~df["is_statement_period"],
        "quality_flags",
    ] += "NON_STANDARD_PERIOD;"

    df.loc[
        df["is_summary_metric"],
        "quality_flags",
    ] += "SUMMARY_METRIC_REVIEW;"

    # Duplicate key audit: never silently drop observations.
    duplicate_key = [
        "company",
        "section_id",
        "metric_clean",
        "period_original",
    ]

    df["duplicate_key_flag"] = df.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    # Separate data classes for downstream review.
    statement_candidates = df[
        df["is_statement_period"]
        & ~df["is_summary_metric"]
    ].copy()

    summary_candidates = df[
        df["is_summary_metric"]
        | ~df["is_statement_period"]
    ].copy()

    # Do not claim these candidates are fully validated financial
    # statements: units and reporting basis still require verification.
    statement_candidates["validation_status"] = (
        "CANDIDATE_REQUIRES_UNIT_AND_BASIS_REVIEW"
    )

    summary_candidates["validation_status"] = (
        "SUMMARY_OR_NONSTANDARD_PERIOD_REVIEW"
    )

    df.to_csv(CLEAN_FILE, index=False)

    audit_rows = []

    for (company, section), group in df.groupby(
        ["company", "section"],
        dropna=False,
    ):
        audit_rows.append({
            "company": company,
            "section": section,
            "input_rows": len(group),
            "unique_metrics": group["metric_clean"].nunique(),
            "date_period_rows": int(
                group["is_statement_period"].sum()
            ),
            "nonstandard_period_rows": int(
                (~group["is_statement_period"]).sum()
            ),
            "summary_metric_rows": int(
                group["is_summary_metric"].sum()
            ),
            "missing_numeric_values": int(
                group["numeric_value"].isna().sum()
            ),
            "duplicate_key_rows": int(
                group["duplicate_key_flag"].sum()
            ),
            "status": "REVIEW_REQUIRED",
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(SUMMARY_FILE, index=False)

    statement_candidates.to_csv(
        OUTPUT_DIR / "statement_candidates.csv",
        index=False,
    )

    summary_candidates.to_csv(
        OUTPUT_DIR / "summary_metric_candidates.csv",
        index=False,
    )

    result = {
        "input_file": str(INPUT_FILE),
        "output_directory": str(OUTPUT_DIR),
        "input_rows": original_rows,
        "cleaned_rows": len(df),
        "companies": int(df["company"].nunique()),
        "sections": int(df["section"].nunique()),
        "statement_candidate_rows": len(statement_candidates),
        "summary_candidate_rows": len(summary_candidates),
        "rows_with_duplicate_keys": int(
            df["duplicate_key_flag"].sum()
        ),
        "missing_numeric_values": int(
            df["numeric_value"].isna().sum()
        ),
        "reporting_basis": "UNVERIFIED",
        "unit_status": "VERIFY_BY_SECTION",
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "CLEANING_COMPLETE_VALIDATION_PENDING",
    }

    JSON_FILE.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print("\nAudit files created:")
    print(f"  {CLEAN_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {JSON_FILE}")
    print(f"  {OUTPUT_DIR / 'statement_candidates.csv'}")
    print(f"  {OUTPUT_DIR / 'summary_metric_candidates.csv'}")


if __name__ == "__main__":
    main()
