
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "clean" / "fundamentals"
    / "screener_consolidated_pilot" / "separated"
    / "all_records_with_table_roles.csv"
)
OUTPUT = (
    ROOT / "data" / "processed" / "fundamentals"
    / "screener_pilot_validation"
)
OUTPUT.mkdir(parents=True, exist_ok=True)


def norm(value):
    """Ignore spaces and punctuation when comparing source labels."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


ALIASES = {
    "sales": {"sales", "revenue"},
    "expenses": {"expenses"},
    "profit_before_tax": {"profitbeforetax"},
    "net_profit": {"netprofit"},
    "eps": {"epsinrs"},
    "fixed_assets": {"fixedassets"},
    "borrowings": {"borrowings", "borrowing"},
    "operating_cash_flow": {"cashfromoperatingactivity"},
    "investing_cash_flow": {"cashfrominvestingactivity"},
    "financing_cash_flow": {"cashfromfinancingactivity"},
    "net_cash_flow": {"netcashflow"},
    "total_assets": {"totalassets"},
    "total_liabilities": {"totalliabilities"},
    "nonbank_operating_profit": {"operatingprofit"},
    "bank_financing_profit": {"financingprofit"},
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)
    required = {
        "company", "section", "metric", "table_index",
        "period_original", "raw_value"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["table_index_num"] = pd.to_numeric(
        df["table_index"], errors="coerce"
    )
    df["metric_norm"] = df["metric"].map(norm)

    statement = df[
        (
            df["section"].eq("Annual Profit and Loss")
            & df["table_index_num"].eq(0)
        )
        | (
            df["section"].isin([
                "Balance Sheet", "Cash Flow", "Quarterly Results"
            ])
            & df["table_index_num"].eq(0)
        )
    ].copy()

    checks = []

    for company in sorted(df["company"].dropna().unique()):
        company_df = statement[statement["company"].eq(company)]
        is_bank = "BANK" in str(company).upper()

        expectations = {
            "Annual Profit and Loss": [
                ("Sales / revenue", ALIASES["sales"]),
                ("Expenses", ALIASES["expenses"]),
                ("Profit before tax", ALIASES["profit_before_tax"]),
                ("Net profit", ALIASES["net_profit"]),
                ("EPS", ALIASES["eps"]),
                (
                    "Bank financing profit" if is_bank
                    else "Operating profit",
                    ALIASES[
                        "bank_financing_profit" if is_bank
                        else "nonbank_operating_profit"
                    ],
                ),
            ],
            "Balance Sheet": [
                ("Total assets", ALIASES["total_assets"]),
                ("Total liabilities", ALIASES["total_liabilities"]),
                ("Borrowings / borrowing", ALIASES["borrowings"]),
                ("Fixed assets", ALIASES["fixed_assets"]),
            ],
            "Cash Flow": [
                ("Operating cash flow", ALIASES["operating_cash_flow"]),
                ("Investing cash flow", ALIASES["investing_cash_flow"]),
                ("Financing cash flow", ALIASES["financing_cash_flow"]),
                ("Net cash flow", ALIASES["net_cash_flow"]),
            ],
            "Quarterly Results": [
                ("Sales / revenue", ALIASES["sales"]),
                ("Net profit", ALIASES["net_profit"]),
                ("EPS", ALIASES["eps"]),
                (
                    "Bank financing profit" if is_bank
                    else "Operating profit",
                    ALIASES[
                        "bank_financing_profit" if is_bank
                        else "nonbank_operating_profit"
                    ],
                ),
            ],
        }

        for section, items in expectations.items():
            observed = company_df.loc[
                company_df["section"].eq(section), "metric_norm"
            ].tolist()

            for expected_name, accepted_labels in items:
                found = any(
                    label in accepted_labels for label in observed
                )
                checks.append({
                    "company": company,
                    "section": section,
                    "expected_metric": expected_name,
                    "found_by_alias": found,
                    "observed_labels": " | ".join(sorted(
                        company_df.loc[
                            company_df["section"].eq(section), "metric"
                        ].dropna().astype(str).unique()
                    )),
                    "status": (
                        "LABEL_FOUND_REQUIRES_UNIT_AND_VALUE_REVIEW"
                        if found else "MISSING_LABEL_REVIEW"
                    ),
                })

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "core_metric_checks.csv", index=False)

    summary = {
        "companies": int(df["company"].nunique()),
        "source_rows": int(len(df)),
        "alias_based_checks": int(len(result)),
        "labels_found": int(result["found_by_alias"].sum()),
        "labels_missing": int((~result["found_by_alias"]).sum()),
        "unit_scale_verified": False,
        "financial_semantics_verified": False,
        "approved_for_scoring": False,
        "database_modified": False,
        "models_modified": False,
        "historical_ml_data_modified": False,
        "status": "ALIAS_BASED_LABEL_CHECK_COMPLETE",
    }

    with open(OUTPUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
