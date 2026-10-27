
from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "screener_consolidated_pilot"
    / "separated"
    / "all_records_with_table_roles.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fundamentals"
    / "screener_pilot_readiness"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates, required=True):
    """Find a column without depending on capitalization."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def parse_number(value):
    """Parse a displayed financial value without assuming its unit."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {
        "-", "--", "nan", "none", "n/a", "na"
    }:
        return None

    text = text.replace(",", "").replace("₹", "").strip()
    text = text.replace("%", "").strip()
    text = re.sub(r"^\((.*)\)$", r"-\1", text)

    try:
        result = float(text)
        if pd.notna(result):
            return result
    except (ValueError, TypeError):
        pass

    return None


def classify_metric(metric):
    """Screen metric semantics; this does not independently verify them."""
    m = str(metric).strip().lower()

    if "%" in m or "margin" in m or "payout" in m:
        return "PERCENT_OR_MARGIN_REVIEW"

    if "eps" in m or "per share" in m:
        return "INR_PER_SHARE_REVIEW"

    if any(word in m for word in [
        "growth", "cagr", "return on equity", "roe",
        "return on capital", "roce", "share of operating profit"
    ]):
        return "RATIO_OR_GROWTH_REVIEW"

    if any(word in m for word in [
        "sales", "revenue", "expense", "profit", "income",
        "interest", "tax", "depreciation", "borrow",
        "debt", "asset", "liabilit", "cash", "deposit",
        "investment", "working capital", "equity", "cwip"
    ]):
        return "AMOUNT_UNIT_CONFIRMATION_REQUIRED"

    return "UNCLASSIFIED_REVIEW"


def parse_period(value):
    """Return a normalized date when possible; otherwise preserve as a label."""
    if pd.isna(value):
        return None, "MISSING_PERIOD"

    text = str(value).strip()
    if not text:
        return None, "MISSING_PERIOD"

    if text.upper() == "TTM":
        return None, "TTM_NON_DATE"

    if re.search(r"\b(years?|last year|ttm)\b", text, re.I):
        return None, "SUMMARY_NON_DATE"

    date = pd.to_datetime(text, errors="coerce")

    if pd.isna(date):
        return None, "UNPARSED_PERIOD"

    return date.normalize(), "DATE"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Confirm that the separation script has completed."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    company_col = find_column(df, ["company"])
    section_col = find_column(df, ["section"])
    metric_col = find_column(df, ["metric"])
    period_col = find_column(
        df, ["period_original", "period", "period_label"]
    )
    value_col = find_column(
        df, ["raw_value", "value", "reported_value"]
    )
    basis_col = find_column(
        df, ["reporting_basis"], required=False
    )
    table_col = find_column(
        df, ["table_index"], required=False
    )
    role_col = find_column(
        df, ["table_role"], required=False
    )
    url_col = find_column(
        df, ["source_url"], required=False
    )

    audit = df.copy()

    audit["parsed_numeric_value"] = audit[value_col].apply(parse_number)
    parsed = audit[period_col].apply(parse_period)
    audit["normalized_period"] = parsed.apply(lambda x: x[0])
    audit["period_type"] = parsed.apply(lambda x: x[1])
    audit["metric_semantic_screen"] = audit[metric_col].apply(
        classify_metric
    )

    audit["value_parse_status"] = audit["parsed_numeric_value"].apply(
        lambda x: "PARSED" if x is not None else "MISSING_OR_NONNUMERIC"
    )

    audit["source_url_present"] = (
        audit[url_col].notna()
        & audit[url_col].astype(str).str.strip().ne("")
        if url_col else False
    )

    audit["review_flags"] = ""

    def add_flag(mask, flag):
        audit.loc[mask, "review_flags"] = (
            audit.loc[mask, "review_flags"]
            .apply(lambda old: f"{old};{flag}" if old else flag)
        )

    add_flag(
        audit["period_type"].isin(["MISSING_PERIOD", "UNPARSED_PERIOD"]),
        "PERIOD_REVIEW"
    )
    add_flag(
        audit["value_parse_status"] != "PARSED",
        "VALUE_PARSE_REVIEW"
    )
    add_flag(
        audit["metric_semantic_screen"] == "UNCLASSIFIED_REVIEW",
        "METRIC_DICTIONARY_REVIEW"
    )

    if basis_col:
        non_consolidated = (
            audit[basis_col].astype(str).str.lower() != "consolidated"
        )
        add_flag(non_consolidated, "REPORTING_BASIS_REVIEW")

    if url_col:
        add_flag(~audit["source_url_present"], "SOURCE_URL_MISSING")

    # Keep the full audit output. Never discard the original raw values.
    audit_path = OUTPUT_DIR / "pilot_row_level_readiness_audit.csv"
    audit.to_csv(audit_path, index=False)

    # Summarize classification and parsing by section.
    section_summary = (
        audit.groupby([section_col, "period_type"], dropna=False)
        .agg(
            rows=(metric_col, "size"),
            metrics=(metric_col, "nunique"),
            companies=(company_col, "nunique"),
            parsed_values=("parsed_numeric_value", "count"),
        )
        .reset_index()
    )
    section_summary.to_csv(
        OUTPUT_DIR / "section_period_summary.csv", index=False
    )

    metric_summary = (
        audit.groupby(
            [company_col, section_col, metric_col,
             "metric_semantic_screen"],
            dropna=False
        )
        .agg(
            rows=(metric_col, "size"),
            numeric_values=("parsed_numeric_value", "count"),
            periods=("normalized_period", "nunique"),
        )
        .reset_index()
    )
    metric_summary.to_csv(
        OUTPUT_DIR / "metric_coverage_summary.csv", index=False
    )

    # Check quarterly period continuity at the company level.
    quarterly = audit[
        audit[section_col].astype(str).str.lower().eq("quarterly results")
        & audit["normalized_period"].notna()
    ].copy()

    continuity_rows = []

    for company, group in quarterly.groupby(company_col):
        dates = sorted(group["normalized_period"].dropna().unique())

        if len(dates) < 2:
            continuity_rows.append({
                "company": company,
                "distinct_quarter_dates": len(dates),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                "gap_count_over_100_days": None,
                "status": "INSUFFICIENT_DATES_TO_CHECK",
            })
            continue

        gaps = []
        for previous, current in zip(dates[:-1], dates[1:]):
            days = (pd.Timestamp(current) - pd.Timestamp(previous)).days
            if days > 100:
                gaps.append({
                    "previous_date": str(pd.Timestamp(previous).date()),
                    "next_date": str(pd.Timestamp(current).date()),
                    "gap_days": days,
                })

        continuity_rows.append({
            "company": company,
            "distinct_quarter_dates": len(dates),
            "first_date": pd.Timestamp(dates[0]).date().isoformat(),
            "last_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "gap_count_over_100_days": len(gaps),
            "gap_details": json.dumps(gaps),
            "status": (
                "GAPS_REVIEW_REQUIRED" if gaps
                else "NO_GAP_OVER_100_DAYS_DETECTED"
            ),
        })

    pd.DataFrame(continuity_rows).to_csv(
        OUTPUT_DIR / "quarterly_continuity_screen.csv", index=False
    )

    # Cash-flow is expected to be annual in this pilot.
    cashflow = audit[
        audit[section_col].astype(str).str.lower().eq("cash flow")
    ]
    cashflow_periods = sorted(
        cashflow["normalized_period"].dropna().unique()
    )

    summary = {
        "input_file": str(INPUT_FILE),
        "input_rows": int(len(audit)),
        "companies": int(audit[company_col].nunique()),
        "sections": sorted(
            audit[section_col].dropna().astype(str).unique().tolist()
        ),
        "numeric_values_parsed": int(
            audit["parsed_numeric_value"].notna().sum()
        ),
        "rows_with_period_review": int(
            audit["period_type"].isin(
                ["MISSING_PERIOD", "UNPARSED_PERIOD"]
            ).sum()
        ),
        "rows_with_metric_dictionary_review": int(
            (audit["metric_semantic_screen"] == "UNCLASSIFIED_REVIEW").sum()
        ),
        "cash_flow_distinct_dated_periods": int(len(cashflow_periods)),
        "source_values_preserved": True,
        "unit_scale_verified": False,
        "financial_semantics_verified": False,
        "approved_for_recommendation_scoring": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "PILOT_READINESS_SCREEN_COMPLETE_REVIEW_REQUIRED",
    }

    with open(OUTPUT_DIR / "pilot_readiness_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nPILOT READINESS AUDIT COMPLETE")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
