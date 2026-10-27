from pathlib import Path
import json
import pandas as pd

root = Path(".")
register_path = root / "data/clean/fundamentals/normalization_candidate/fundamental_mapping_review_decisions.csv"
observations_path = root / "data/clean/fundamentals/integration_candidate/latest_fundamental_observations_candidate.parquet"
snapshot_path = root / "data/clean/fundamentals/integration_candidate/company_fundamental_snapshot_candidate.parquet"
out_dir = root / "data/clean/fundamentals/integration_candidate/readiness"
out_dir.mkdir(parents=True, exist_ok=True)

for p in (register_path, observations_path, snapshot_path):
    if not p.exists():
        raise FileNotFoundError(f"Required input is missing: {p}")

register = pd.read_csv(register_path)
observations = pd.read_parquet(observations_path)
snapshot = pd.read_parquet(snapshot_path)

required_register = {
    "metric", "review_decision", "unit_verified",
    "financial_value_independently_verified", "approved_for_scoring"
}
required_obs = {
    "ticker", "source", "metric_raw", "unit_verified",
    "financial_value_independently_verified", "scoring_eligible",
    "currency_quarantine", "metric_mapping_approved",
    "integration_review_status"
}
required_snapshot = {
    "ticker", "scoring_eligible", "unit_verified",
    "financial_value_independently_verified"
}

for name, frame, required in [
    ("register", register, required_register),
    ("observations", observations, required_obs),
    ("snapshot", snapshot, required_snapshot),
]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

def true_count(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).sum()

register_approved = (
    register["approved_for_scoring"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)
register_verified = (
    register["unit_verified"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)
register_independently_verified = (
    register["financial_value_independently_verified"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)

obs_scoring = (
    observations["scoring_eligible"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)
snap_scoring = (
    snapshot["scoring_eligible"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)

# Fail closed: eligibility flags cannot override missing register approval.
gate_open = (
    register_approved
    & register_verified
    & register_independently_verified
)

# At present, no mapping should pass this gate.
if not gate_open.any():
    assert not obs_scoring.any(), (
        "Safety failure: observations are marked scoring-eligible while no "
        "mapping is approved and independently verified."
    )
    assert not snap_scoring.any(), (
        "Safety failure: snapshot rows are marked scoring-eligible while no "
        "mapping is approved and independently verified."
    )

summary = {
    "register_rows": int(len(register)),
    "register_approved_mappings": int(register_approved.sum()),
    "register_unit_verified_mappings": int(register_verified.sum()),
    "register_independently_verified_mappings": int(register_independently_verified.sum()),
    "register_mappings_passing_all_approval_checks": int(gate_open.sum()),
    "observation_rows": int(len(observations)),
    "observation_scoring_eligible_rows": int(obs_scoring.sum()),
    "observation_mapping_approved_rows": int(true_count(observations["metric_mapping_approved"])),
    "observation_currency_quarantine_rows": int(true_count(observations["currency_quarantine"])),
    "snapshot_tickers": int(snapshot["ticker"].nunique()),
    "snapshot_scoring_eligible_rows": int(snap_scoring.sum()),
    "gate_status": "PASS_CLOSED_AS_EXPECTED" if not gate_open.any()
        and not obs_scoring.any() and not snap_scoring.any()
        else "REVIEW_REQUIRED",
    "note": "This audit checks current flags; it does not approve mappings or change datasets."
}

report_path = out_dir / "fundamental_scoring_gate_audit.json"
report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nReport saved to: {report_path}")
if summary["gate_status"] != "PASS_CLOSED_AS_EXPECTED":
    raise SystemExit("Scoring gate needs review; inspect the report before proceeding.")
