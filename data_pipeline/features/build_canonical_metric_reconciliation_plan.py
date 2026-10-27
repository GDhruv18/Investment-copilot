import json
from pathlib import Path
import pandas as pd

root = Path(".")
base = root / "data/clean/fundamentals/normalization_candidate"
out_dir = root / "data/clean/fundamentals/integration_candidate/readiness"
out_dir.mkdir(parents=True, exist_ok=True)

register_path = base / "fundamental_mapping_review_decisions.csv"
coverage_path = base / "monetary_scale_final_field_coverage.csv"
pairs_path = base / "monetary_scale_final_pairs.csv"
summary_path = base / "monetary_scale_final_summary.csv"

for p in [register_path, coverage_path, pairs_path, summary_path]:
    if not p.exists():
        raise FileNotFoundError(f"Required existing audit output not found: {p}")

register = pd.read_csv(register_path)
coverage = pd.read_csv(coverage_path)
pairs = pd.read_csv(pairs_path)
summary = pd.read_csv(summary_path)

metrics = {
    "Revenue": {
        "canonical_definition": "Reported revenue for a defined reporting period; preserve quarterly and annual observations separately.",
        "candidate_source_fields": "Screener Sales +; Yahoo Total Revenue; Yahoo Operating Revenue",
        "required_checks": "Confirm same period, consolidated/standalone basis, financial currency, and independently verified unit conversion. Investigate material residual differences.",
        "current_decision": "BLOCKED_PENDING_BASIS_AND_EXCEPTION_REVIEW",
    },
    "Total Assets": {
        "canonical_definition": "Reported total assets at a specific balance-sheet date.",
        "candidate_source_fields": "Screener Total Assets; Yahoo Total Assets",
        "required_checks": "Confirm same date, consolidated/standalone basis, financial currency, and independently verified unit conversion. Review differences greater than 1%.",
        "current_decision": "CONDITIONAL_REVIEW_NOT_APPROVED",
    },
    "Total Debt": {
        "canonical_definition": "Reported total debt/borrowings at a specific balance-sheet date; do not equate it with total liabilities.",
        "candidate_source_fields": "Screener Borrowings +; Screener Borrowing; Yahoo Total Debt",
        "required_checks": "Confirm debt definition and balance-sheet date, source coverage, currency, units, sign consistency, and exceptions. Treat alternative Screener fields as separate candidates.",
        "current_decision": "CONDITIONAL_REVIEW_NOT_APPROVED",
    },
    "Operating Cash Flow": {
        "canonical_definition": "Net cash flow from operating activities for a defined reporting period.",
        "candidate_source_fields": "Screener Cash from Operating Activity +; Yahoo Operating Cash Flow; Yahoo Cash Flow From Continuing Operating Activities",
        "required_checks": "Confirm period, continuing/discontinued operations basis, currency, units, sign consistency, and exceptions. Treat Yahoo alternatives separately.",
        "current_decision": "CONDITIONAL_REVIEW_NOT_APPROVED",
    },
}

rows = []
for metric, info in metrics.items():
    rows.append({
        "canonical_metric": metric,
        **info,
        "unit_verified": False,
        "financial_value_independently_verified": False,
        "approved_for_scoring": False,
    })

report = pd.DataFrame(rows)
report_path = out_dir / "canonical_metric_reconciliation_plan.csv"
report.to_csv(report_path, index=False)

result = {
    "status": "REVIEW_PLAN_CREATED_NO_APPROVALS",
    "canonical_metrics": len(report),
    "approved_for_scoring": int(report["approved_for_scoring"].sum()),
    "unit_verified": int(report["unit_verified"].sum()),
    "source_register_rows": len(register),
    "scale_coverage_rows": len(coverage),
    "scale_pair_rows": len(pairs),
    "scale_summary_rows": len(summary),
    "report": str(report_path),
    "note": "This is a review plan only. It does not approve mappings, change source data, or enable scoring."
}
summary_out = out_dir / "canonical_metric_reconciliation_plan_summary.json"
summary_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))
print("\nCreated canonical metric reconciliation plan. No scoring approvals were made.")
