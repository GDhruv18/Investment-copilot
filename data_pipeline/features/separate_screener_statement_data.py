
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_pilot"
    / "consolidated_financials_long.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "clean" / "fundamentals"
    / "screener_consolidated_pilot" / "separated"
)


def normalize_period(value):
    if pd.isna(value):
        return ""
    return str(value).strip().rstrip(":").strip().lower()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required = {
        "company", "section", "section_id", "table_index",
        "metric", "period_original", "period_end",
        "raw_value", "numeric_value", "source_url",
        "reporting_basis",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["period_key"] = df["period_original"].apply(normalize_period)
    df["period_date"] = pd.to_datetime(df["period_end"], errors="coerce")

    # Table 0 is the statement table in every section in this pilot.
    # For annual P&L, tables 1-4 are summary indicators.
    df["table_role"] = "REVIEW_REQUIRED"

    df.loc[
        df["section"].eq("Annual Profit and Loss")
        & df["table_index"].eq(0),
        "table_role",
    ] = "ANNUAL_PL_STATEMENT_TABLE"

    df.loc[
        df["section"].eq("Annual Profit and Loss")
        & df["table_index"].gt(0),
        "table_role",
    ] = "ANNUAL_PL_SUMMARY_TABLE"

    for section, role in {
        "Quarterly Results": "QUARTERLY_RESULTS_TABLE",
        "Balance Sheet": "BALANCE_SHEET_TABLE",
        "Cash Flow": "ANNUAL_CASH_FLOW_TABLE",
    }.items():
        df.loc[
            df["section"].eq(section)
            & df["table_index"].eq(0),
            "table_role",
        ] = role

    # Separate dated observations from non-date TTM values.
    df["record_bucket"] = "REVIEW_REQUIRED"

    statement_roles = {
        "ANNUAL_PL_STATEMENT_TABLE",
        "QUARTERLY_RESULTS_TABLE",
        "BALANCE_SHEET_TABLE",
        "ANNUAL_CASH_FLOW_TABLE",
    }

    is_statement_table = df["table_role"].isin(statement_roles)

    df.loc[
        is_statement_table & df["period_date"].notna(),
        "record_bucket",
    ] = "DATED_STATEMENT_CANDIDATE"

    df.loc[
        is_statement_table
        & df["period_key"].eq("ttm")
        & df["period_date"].isna(),
        "record_bucket",
    ] = "TTM_SUMMARY_CANDIDATE"

    df.loc[
        df["table_role"].eq("ANNUAL_PL_SUMMARY_TABLE"),
        "record_bucket",
    ] = "RATIO_OR_GROWTH_SUMMARY"

    df.loc[
        df["record_bucket"].eq("REVIEW_REQUIRED"),
        "record_bucket",
    ] = "OTHER_NONSTANDARD_RECORD"

    # Keep every row in the master file; do not silently discard data.
    df.to_csv(OUTPUT_DIR / "all_records_with_table_roles.csv", index=False)

    dated = df[df["record_bucket"].eq("DATED_STATEMENT_CANDIDATE")].copy()
    ttm = df[df["record_bucket"].eq("TTM_SUMMARY_CANDIDATE")].copy()
    summaries = df[df["record_bucket"].eq("RATIO_OR_GROWTH_SUMMARY")].copy()
    other = df[df["record_bucket"].eq("OTHER_NONSTANDARD_RECORD")].copy()

    dated.to_csv(OUTPUT_DIR / "dated_statement_candidates.csv", index=False)
    ttm.to_csv(OUTPUT_DIR / "ttm_summary_candidates.csv", index=False)
    summaries.to_csv(OUTPUT_DIR / "growth_ratio_summary_candidates.csv", index=False)
    other.to_csv(OUTPUT_DIR / "other_nonstandard_records.csv", index=False)

    table_audit = (
        df.groupby(
            ["company", "section", "table_index", "table_role"],
            dropna=False,
        )
        .agg(
            rows=("metric", "size"),
            metrics=("metric", "nunique"),
            dated_rows=("period_date", lambda x: x.notna().sum()),
            distinct_dates=("period_date", "nunique"),
        )
        .reset_index()
    )
    table_audit.to_csv(OUTPUT_DIR / "table_role_audit.csv", index=False)

    summary = {
        "input_rows": len(df),
        "dated_statement_candidate_rows": len(dated),
        "ttm_summary_candidate_rows": len(ttm),
        "growth_ratio_summary_rows": len(summaries),
        "other_nonstandard_rows": len(other),
        "all_input_rows_preserved_in_master": len(df) == len(pd.read_csv(INPUT_FILE)),
        "reporting_basis_values": sorted(
            df["reporting_basis"].dropna().astype(str).unique().tolist()
        ),
        "unit_scale_verified": False,
        "cash_flow_semantics_verified": False,
        "statement_candidates_approved_for_scoring": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "TABLES_SEPARATED_VALIDATION_PENDING",
    }

    (OUTPUT_DIR / "separation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSEPARATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
