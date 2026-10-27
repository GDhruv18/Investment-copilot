
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "raw" / "fundamentals"
    / "screener_consolidated_100"
    / "consolidated_financials_long.csv"
)
PAGE_AUDIT_FILE = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "collection_audit.csv"
)
OUT = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_100" / "completeness_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SECTIONS = [
    "Quarterly Results",
    "Annual Profit and Loss",
    "Balance Sheet",
    "Cash Flow",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE, low_memory=False)

    required = {
        "ticker", "section", "metric", "period_original",
        "period_end", "numeric_value", "source_url",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")

    tickers = sorted(data["ticker"].dropna().unique())
    expected = pd.MultiIndex.from_product(
        [tickers, EXPECTED_SECTIONS],
        names=["ticker", "section"],
    ).to_frame(index=False)

    grouped = (
        data.groupby(["ticker", "section"], dropna=False)
        .agg(
            records=("metric", "size"),
            metrics=("metric", "nunique"),
            period_labels=("period_original", "nunique"),
            dated_periods=("period_end", "nunique"),
            numeric_values=("numeric_value", "count"),
            source_urls=("source_url", "nunique"),
        )
        .reset_index()
    )

    audit = expected.merge(
        grouped, on=["ticker", "section"], how="left"
    )

    count_cols = [
        "records", "metrics", "period_labels",
        "dated_periods", "numeric_values", "source_urls",
    ]
    audit[count_cols] = audit[count_cols].fillna(0).astype(int)

    audit["status"] = "DATA_PRESENT_REVIEW_HISTORY"
    audit.loc[audit["records"].eq(0), "status"] = "NO_RECORDS"
    audit.loc[
        audit["records"].gt(0) & audit["numeric_values"].eq(0),
        "status"
    ] = "NO_NUMERIC_VALUES"
    audit.loc[
        audit["records"].gt(0) & audit["dated_periods"].eq(0),
        "status"
    ] = "NO_PARSED_DATED_PERIODS"

    if PAGE_AUDIT_FILE.exists():
        page_audit = pd.read_csv(PAGE_AUDIT_FILE, low_memory=False)
        keep = [
            c for c in [
                "ticker", "status", "http_status",
                "sections_found", "tables_found",
                "records_extracted", "error"
            ] if c in page_audit.columns
        ]
        if "ticker" in keep:
            page_audit = page_audit[keep].rename(
                columns={"status": "page_collection_status"}
            )
            audit = audit.merge(page_audit, on="ticker", how="left")

    audit.to_csv(OUT / "stock_section_completeness.csv", index=False)

    issues = audit[audit["status"].ne(
        "DATA_PRESENT_REVIEW_HISTORY"
    )].copy()
    issues.to_csv(OUT / "completeness_issues.csv", index=False)

    summary = {
        "tickers_in_raw_data": len(tickers),
        "expected_stock_section_pairs": int(len(expected)),
        "pairs_with_records": int(audit["records"].gt(0).sum()),
        "pairs_with_no_records": int(audit["records"].eq(0).sum()),
        "pairs_without_parsed_dates": int(
            (
                audit["records"].gt(0)
                & audit["dated_periods"].eq(0)
            ).sum()
        ),
        "total_raw_records": int(len(data)),
        "unparseable_numeric_values": int(
            data["numeric_value"].isna().sum()
        ),
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "COMPLETENESS_AUDIT_COMPLETE_REVIEW_REQUIRED",
    }

    with open(OUT / "completeness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCOMPLETENESS AUDIT COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput folder: {OUT}")


if __name__ == "__main__":
    main()
