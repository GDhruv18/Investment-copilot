
from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_consolidated_pilot"
    / "consolidated_financials_long.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_consolidated_pilot"
    / "audit"
)

SUMMARY_TERMS = re.compile(
    r"growth|cagr|return on equity|return on capital|"
    r"stock price|high price|low price|"
    r"dividend payout|share of operating profit",
    re.IGNORECASE,
)

PERCENT_TERMS = re.compile(
    r"%|margin|tax rate|dividend payout|growth",
    re.IGNORECASE,
)

EPS_TERMS = re.compile(r"\beps\b|earnings per share", re.IGNORECASE)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "company",
        "reporting_basis",
        "section",
        "section_id",
        "metric",
        "period_original",
        "period_end",
        "raw_value",
        "numeric_value",
        "unit",
        "source_url",
        "table_index",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Preserve original input and create audit-only flags.
    df["metric_clean"] = (
        df["metric"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace(r"\s*\+\s*$", "", regex=True)
    )

    df["is_summary_metric"] = df["metric_clean"].str.contains(
        SUMMARY_TERMS, na=False
    )

    df["is_percentage_metric"] = df.apply(
        lambda row: bool(
            PERCENT_TERMS.search(str(row["metric_clean"]))
            or "%" in str(row["raw_value"])
        ),
        axis=1,
    )

    df["is_eps_metric"] = df["metric_clean"].str.contains(
        EPS_TERMS, na=False
    )

    df["is_date_period"] = pd.to_datetime(
        df["period_end"], errors="coerce"
    ).notna()

    df["is_consolidated_basis"] = (
        df["reporting_basis"].astype(str).str.lower()
        == "consolidated"
    )

    # Same metric and period appearing in different tables can be
    # legitimate, so flag it for review rather than deleting it.
    duplicate_key = [
        "company",
        "reporting_basis",
        "section_id",
        "metric_clean",
        "period_original",
    ]

    df["duplicate_metric_period"] = df.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    df["quality_flags"] = ""

    df.loc[
        ~df["is_date_period"],
        "quality_flags",
    ] += "NON_DATE_PERIOD;"

    df.loc[
        df["is_summary_metric"],
        "quality_flags",
    ] += "SUMMARY_METRIC_REVIEW;"

    df.loc[
        ~df["is_consolidated_basis"],
        "quality_flags",
    ] += "REPORTING_BASIS_MISMATCH;"

    df.loc[
        df["numeric_value"].isna(),
        "quality_flags",
    ] += "MISSING_OR_NONNUMERIC_VALUE;"

    df.loc[
        df["duplicate_metric_period"],
        "quality_flags",
    ] += "DUPLICATE_KEY_REVIEW;"

    # Section-level audit.
    section_audit = (
        df.groupby(["company", "section"], dropna=False)
        .agg(
            rows=("metric_clean", "size"),
            unique_metrics=("metric_clean", "nunique"),
            dated_rows=("is_date_period", "sum"),
            summary_metric_rows=("is_summary_metric", "sum"),
            percentage_metric_rows=("is_percentage_metric", "sum"),
            eps_metric_rows=("is_eps_metric", "sum"),
            missing_values=("numeric_value", lambda x: x.isna().sum()),
            reporting_basis_ok=("is_consolidated_basis", "all"),
            earliest_period=("period_end", "min"),
            latest_period=("period_end", "max"),
        )
        .reset_index()
    )

    # Coverage by metric, useful for identifying inconsistent period
    # coverage across the five companies.
    metric_coverage = (
        df[df["is_date_period"]]
        .groupby(["company", "section", "metric_clean"], dropna=False)
        .agg(
            period_count=("period_end", "nunique"),
            first_period=("period_end", "min"),
            last_period=("period_end", "max"),
            rows=("period_end", "size"),
        )
        .reset_index()
    )

    # Full record-level audit output.
    df.to_csv(OUTPUT_DIR / "record_quality_flags.csv", index=False)
    section_audit.to_csv(
        OUTPUT_DIR / "section_quality_audit.csv", index=False
    )
    metric_coverage.to_csv(
        OUTPUT_DIR / "metric_period_coverage.csv", index=False
    )

    total_rows = len(df)

    summary = {
        "input_rows": total_rows,
        "companies": int(df["company"].nunique()),
        "sections": int(df["section"].nunique()),
        "all_reporting_basis_consolidated": bool(
            df["is_consolidated_basis"].all()
        ),
        "non_date_period_rows": int((~df["is_date_period"]).sum()),
        "summary_metric_rows": int(df["is_summary_metric"].sum()),
        "percentage_metric_rows": int(df["is_percentage_metric"].sum()),
        "eps_metric_rows": int(df["is_eps_metric"].sum()),
        "missing_numeric_values": int(df["numeric_value"].isna().sum()),
        "duplicate_metric_period_rows": int(
            df["duplicate_metric_period"].sum()
        ),
        "source_urls_present": int(df["source_url"].notna().sum()),
        "unit_status": "REQUIRES METRIC-LEVEL VERIFICATION",
        "cash_flow_period_semantics": "NOT YET VERIFIED",
        "financial_values_independently_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "AUDIT_COMPLETE_REVIEW_REQUIRED",
    }

    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nAUDIT SUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nSECTION AUDIT")
    print(section_audit.to_string(index=False))

    print("\nFiles written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
