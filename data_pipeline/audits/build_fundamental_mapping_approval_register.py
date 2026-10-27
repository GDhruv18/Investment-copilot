
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
)

SUMMARY_PATH = BASE / "monetary_scale_final_summary.csv"
CURRENCY_PATH = BASE / "ticker_currency_reference_audit.parquet"

REGISTER_PATH = BASE / "fundamental_mapping_approval_register.csv"
REPORT_PATH = BASE / "fundamental_mapping_approval_register_summary.json"


# These are review decisions based on the completed audit.
# None of these statuses means approved for scoring.
MAPPING_POLICIES = [
    {
        "metric": "revenue",
        "screener_field": "Sales +",
        "yahoo_field": "Total Revenue",
        "decision": "CANDIDATE_FOR_UNIT_AND_DEFINITION_REVIEW",
        "reason": (
            "361 non-mismatch pairs were near 1e7. "
            "Verify units, reporting definitions, and exceptions."
        ),
    },
    {
        "metric": "revenue",
        "screener_field": "Sales +",
        "yahoo_field": "Operating Revenue",
        "decision": "CANDIDATE_FOR_UNIT_AND_DEFINITION_REVIEW",
        "reason": (
            "361 non-mismatch pairs were near 1e7. "
            "Verify equivalence of source definitions before approval."
        ),
    },
    {
        "metric": "revenue",
        "screener_field": "Revenue +",
        "yahoo_field": "Total Revenue",
        "decision": "BLOCKED_DEFINITION_OR_SCALE_REVIEW",
        "reason": (
            "Median ratio was approximately 6.85e6; the observed scale "
            "pattern was less consistent than for Sales +."
        ),
    },
    {
        "metric": "revenue",
        "screener_field": "Revenue +",
        "yahoo_field": "Operating Revenue",
        "decision": "BLOCKED_DEFINITION_OR_SCALE_REVIEW",
        "reason": (
            "Median ratio was approximately 6.85e6; the observed scale "
            "pattern was less consistent than for Sales +."
        ),
    },
    {
        "metric": "total_assets",
        "screener_field": "Total Assets",
        "yahoo_field": "Total Assets",
        "decision": "CANDIDATE_FOR_UNIT_AND_DEFINITION_REVIEW",
        "reason": (
            "374 non-mismatch pairs were near 1e7. "
            "Confirm units and balance-sheet comparability."
        ),
    },
    {
        "metric": "total_debt",
        "screener_field": "Borrowings +",
        "yahoo_field": "Total Debt",
        "decision": "CANDIDATE_WITH_EXCEPTIONS",
        "reason": (
            "323 of 329 non-mismatch pairs were near 1e7; "
            "one sign disagreement and remaining outliers require review."
        ),
    },
    {
        "metric": "total_debt",
        "screener_field": "Borrowing",
        "yahoo_field": "Total Debt",
        "decision": "LIMITED_COVERAGE_REVIEW",
        "reason": (
            "44 pairs across 11 tickers were near 1e7. "
            "Coverage is too limited to generalize without further review."
        ),
    },
    {
        "metric": "operating_cash_flow",
        "screener_field": "Cash from Operating Activity +",
        "yahoo_field": "Operating Cash Flow",
        "decision": "CANDIDATE_WITH_EXCEPTIONS",
        "reason": (
            "370 of 372 non-mismatch pairs were near 1e7; "
            "one sign disagreement and remaining exceptions require review."
        ),
    },
    {
        "metric": "operating_cash_flow",
        "screener_field": "Cash from Operating Activity +",
        "yahoo_field": "Cash Flow From Continuing Operating Activities",
        "decision": "LIMITED_COVERAGE_REVIEW",
        "reason": (
            "Only 17 non-mismatch pairs were usable, with 15 near 1e7. "
            "Coverage is limited and requires definition review."
        ),
    },
]


def main():
    for path in [SUMMARY_PATH, CURRENCY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required audit output not found: {path}")

    summary = pd.read_csv(SUMMARY_PATH)
    currency = pd.read_parquet(CURRENCY_PATH)

    required_summary = {
        "metric",
        "screener_field",
        "yahoo_field",
        "matched_pairs",
        "distinct_tickers",
        "currency_mismatch_pairs",
        "usable_for_scale_diagnostic",
        "median_ratio_nonmismatch",
        "pairs_near_1e7_nonmismatch",
        "sign_disagreements",
    }
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(
            f"Scale summary missing columns: {sorted(missing)}"
        )

    required_currency = {
        "ticker", "currency_status", "financial_currency"
    }
    missing = required_currency - set(currency.columns)
    if missing:
        raise ValueError(
            f"Currency reference missing columns: {sorted(missing)}"
        )

    # Exact field matches only; preserve the observed audit counts.
    rows = []
    for policy in MAPPING_POLICIES:
        match = summary[
            summary["metric"].eq(policy["metric"])
            & summary["screener_field"].eq(policy["screener_field"])
            & summary["yahoo_field"].eq(policy["yahoo_field"])
        ]

        record = dict(policy)
        record["audit_row_found"] = not match.empty

        if match.empty:
            record.update({
                "matched_pairs": 0,
                "distinct_tickers": 0,
                "currency_mismatch_pairs": 0,
                "usable_for_scale_diagnostic": 0,
                "median_ratio_nonmismatch": None,
                "pairs_near_1e7_nonmismatch": 0,
                "sign_disagreements": 0,
                "audit_coverage_status": "NO_EXACT_SUMMARY_MATCH",
            })
        else:
            # Expected to be one row per metric and exact field pair.
            # If the source summary has more than one, preserve ambiguity.
            if len(match) != 1:
                record.update({
                    "matched_pairs": int(match["matched_pairs"].sum()),
                    "distinct_tickers": int(match["distinct_tickers"].max()),
                    "currency_mismatch_pairs": int(
                        match["currency_mismatch_pairs"].sum()
                    ),
                    "usable_for_scale_diagnostic": int(
                        match["usable_for_scale_diagnostic"].sum()
                    ),
                    "median_ratio_nonmismatch": None,
                    "pairs_near_1e7_nonmismatch": int(
                        match["pairs_near_1e7_nonmismatch"].sum()
                    ),
                    "sign_disagreements": int(
                        match["sign_disagreements"].sum()
                    ),
                    "audit_coverage_status": "AMBIGUOUS_SUMMARY_ROWS",
                })
                record["decision"] = "BLOCKED_AMBIGUOUS_AUDIT_COVERAGE"
            else:
                item = match.iloc[0]
                for column in [
                    "matched_pairs",
                    "distinct_tickers",
                    "currency_mismatch_pairs",
                    "usable_for_scale_diagnostic",
                    "median_ratio_nonmismatch",
                    "pairs_near_1e7_nonmismatch",
                    "sign_disagreements",
                ]:
                    value = item[column]
                    record[column] = (
                        None if pd.isna(value) else value.item()
                        if hasattr(value, "item") else value
                    )
                record["audit_coverage_status"] = "SUMMARY_ROW_FOUND"

        # Keep the gate closed even for mappings with strong scale evidence.
        record["unit_verified"] = False
        record["financial_value_independently_verified"] = False
        record["approved_for_scoring"] = False
        rows.append(record)

    register = pd.DataFrame(rows)

    currency_review = currency[
        currency["currency_status"].isin([
            "FINANCIAL_CURRENCY_MISMATCH_REVIEW",
            "MISSING_METADATA",
            "QUOTE_CURRENCY_CONFLICT_REVIEW",
        ])
    ].copy()

    review_tickers = sorted(
        currency_review["ticker"].dropna().astype(str).unique().tolist()
    )

    REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(REGISTER_PATH, index=False)

    report = {
        "register": "fundamental_mapping_approval_register",
        "mapping_rows": int(len(register)),
        "mapping_rows_with_audit_evidence": int(
            register["audit_row_found"].sum()
        ),
        "mapping_rows_approved_for_scoring": 0,
        "unit_verified_mappings": 0,
        "currency_review_tickers": review_tickers,
        "currency_review_ticker_count": len(review_tickers),
        "mapping_decisions": {
            str(k): int(v)
            for k, v in register["decision"].value_counts().items()
        },
        "outputs": [
            str(REGISTER_PATH.relative_to(ROOT)),
            str(REPORT_PATH.relative_to(ROOT)),
        ],
        "policy": (
            "This is a review register, not a scoring approval. "
            "Scale similarity alone does not verify units or metric definitions. "
            "No source data or normalization values were changed."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("FUNDAMENTAL MAPPING APPROVAL REGISTER")
    print("=" * 72)
    print("Mappings:", len(register))
    print("Mappings with audit evidence:", int(register["audit_row_found"].sum()))
    print("Mappings approved for scoring: 0")
    print("Unit-verified mappings: 0")
    print("\nDecisions:")
    print(register[
        [
            "metric", "screener_field", "yahoo_field",
            "decision", "matched_pairs",
            "median_ratio_nonmismatch",
            "pairs_near_1e7_nonmismatch",
        ]
    ].to_string(index=False))
    print("\nCurrency review tickers:", ", ".join(review_tickers) or "None")
    print("\nRegister:", REGISTER_PATH.relative_to(ROOT))
    print("Summary:", REPORT_PATH.relative_to(ROOT))
    print("\nNo source data modified. Scoring remains disabled.")


if __name__ == "__main__":
    main()
