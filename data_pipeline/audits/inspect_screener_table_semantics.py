
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
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_consolidated_pilot" / "table_semantics"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    # Audit each source table independently. Do not assume that every
    # table inside a page section is a financial statement.
    table_audit = (
        df.groupby(
            ["company", "section", "section_id", "table_index"],
            dropna=False,
        )
        .agg(
            rows=("metric", "size"),
            unique_metrics=("metric", "nunique"),
            first_metrics=(
                "metric",
                lambda x: " | ".join(
                    x.dropna().astype(str).drop_duplicates().head(12)
                ),
            ),
            first_periods=(
                "period_original",
                lambda x: " | ".join(
                    x.dropna().astype(str).drop_duplicates().head(6)
                ),
            ),
            last_periods=(
                "period_original",
                lambda x: " | ".join(
                    x.dropna().astype(str).drop_duplicates().tail(4)
                ),
            ),
            source_urls=("source_url", "nunique"),
        )
        .reset_index()
    )

    table_audit.to_csv(
        OUTPUT_DIR / "source_table_structure_audit.csv",
        index=False,
    )

    # Metrics that must not be treated as ordinary monetary values.
    summary_pattern = re.compile(
        r"^(1 Year|Last Year|TTM|3 Years|5 Years|10 Years):?$",
        re.IGNORECASE,
    )

    derived_pattern = re.compile(
        r"CFO/EBITDA|Share of operating profit converting into cash",
        re.IGNORECASE,
    )

    df["is_summary_label"] = df["metric"].fillna("").str.match(
        summary_pattern
    )
    df["is_derived_metric"] = df["metric"].fillna("").str.contains(
        derived_pattern, regex=True
    )

    # Detailed examples for manual inspection.
    inspect = df[
        df["section"].isin([
            "Annual Profit and Loss",
            "Cash Flow",
            "Quarterly Results",
        ])
    ].copy()

    inspect.to_csv(
        OUTPUT_DIR / "financial_semantics_record_samples.csv",
        index=False,
    )

    cash_flow_metrics = (
        df[df["section"].eq("Cash Flow")]
        .groupby(["company", "metric", "table_index"], dropna=False)
        .agg(
            rows=("raw_value", "size"),
            first_period=("period_original", "first"),
            last_period=("period_original", "last"),
            example_values=(
                "raw_value",
                lambda x: " | ".join(
                    x.dropna().astype(str).head(5)
                ),
            ),
        )
        .reset_index()
    )

    cash_flow_metrics.to_csv(
        OUTPUT_DIR / "cash_flow_metric_inspection.csv",
        index=False,
    )

    summary = {
        "input_rows": len(df),
        "companies": int(df["company"].nunique()),
        "distinct_source_tables": int(
            df[["company", "section_id", "table_index"]]
            .drop_duplicates()
            .shape[0]
        ),
        "table_audit_file": str(
            OUTPUT_DIR / "source_table_structure_audit.csv"
        ),
        "cash_flow_inspection_file": str(
            OUTPUT_DIR / "cash_flow_metric_inspection.csv"
        ),
        "record_samples_file": str(
            OUTPUT_DIR / "financial_semantics_record_samples.csv"
        ),
        "cash_flow_period_semantics_verified": False,
        "table_roles_verified": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
    }

    (OUTPUT_DIR / "table_semantics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTABLE SEMANTICS SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTABLE STRUCTURE")
    print(table_audit.to_string(index=False))
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
