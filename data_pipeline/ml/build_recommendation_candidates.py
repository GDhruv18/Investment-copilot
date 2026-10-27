from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path.cwd()
SIGNAL_DIR = ROOT / "data" / "ml" / "current_signals"
OUT_DIR = ROOT / "data" / "ml" / "recommendation_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(SIGNAL_DIR.glob("model_signals_*.parquet"))
if not files:
    raise FileNotFoundError("No model_signals_*.parquet files found.")

# Use the latest completed inference output.
signals = pd.read_parquet(files[-1])

required = {
    "asset_class", "ticker", "feature_date", "horizon",
    "prediction", "prob_weak", "prob_neutral", "prob_attractive"
}
missing = required - set(signals.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

for col in ["prob_weak", "prob_neutral", "prob_attractive"]:
    signals[col] = pd.to_numeric(signals[col], errors="coerce")

signals["feature_date"] = pd.to_datetime(signals["feature_date"], errors="coerce")
signals = signals.dropna(subset=[
    "asset_class", "ticker", "horizon", "feature_date",
    "prob_weak", "prob_neutral", "prob_attractive"
]).copy()

# Transparent ranking score: higher Attractive probability and lower Weak
# probability improve the score. This is NOT an expected-return estimate.
signals["signal_score"] = (
    signals["prob_attractive"] - signals["prob_weak"]
)

today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
signals["feature_age_days"] = (
    today - signals["feature_date"].dt.normalize()
).dt.days
signals["freshness_status"] = np.where(
    signals["feature_age_days"] <= 14,
    "RECENT_FEATURES",
    "STALE_FEATURES_REVIEW_REQUIRED"
)

profiles = {
    "conservative": {
        "allowed_assets": {"etfs", "mutual_funds"},
        "max_weak_probability": 0.30,
        "min_attractive_probability": 0.30,
        "description": "Lower-risk screening; excludes individual stocks."
    },
    "moderate": {
        "allowed_assets": {"etfs", "mutual_funds", "stocks"},
        "max_weak_probability": 0.35,
        "min_attractive_probability": 0.30,
        "description": "Balanced screening across asset classes."
    },
    "aggressive": {
        "allowed_assets": {"etfs", "mutual_funds", "stocks"},
        "max_weak_probability": 0.40,
        "min_attractive_probability": 0.25,
        "description": "Broader screening; accepts higher model-indicated risk."
    }
}

outputs = []
for profile_name, policy in profiles.items():
    df = signals[signals["asset_class"].isin(policy["allowed_assets"])].copy()

    df["risk_profile"] = profile_name
    df["profile_policy"] = policy["description"]
    df["passes_probability_filter"] = (
        (df["prob_weak"] <= policy["max_weak_probability"]) &
        (df["prob_attractive"] >= policy["min_attractive_probability"])
    )

    # Rank only within the same profile, horizon and asset class.
    df["rank_within_group"] = (
        df.groupby(["risk_profile", "horizon", "asset_class"])["signal_score"]
          .rank(method="first", ascending=False)
          .astype(int)
    )

    df["candidate_status"] = np.select(
        [
            ~df["passes_probability_filter"],
            df["freshness_status"] != "RECENT_FEATURES",
            df["prediction"].astype(str).str.lower().eq("weak")
        ],
        [
            "NOT_SHORTLISTED_PROBABILITY_FILTER",
            "REVIEW_STALE_FEATURES",
            "REVIEW_WEAK_SIGNAL"
        ],
        default="SHORTLIST_CANDIDATE"
    )

    df["reason"] = (
        "Prediction=" + df["prediction"].astype(str) +
        "; Attractive probability=" + df["prob_attractive"].round(3).astype(str) +
        "; Weak probability=" + df["prob_weak"].round(3).astype(str) +
        "; feature age=" + df["feature_age_days"].astype(str) + " days"
    )

    outputs.append(df)

result = pd.concat(outputs, ignore_index=True)
result["status"] = "SCREENING_CANDIDATE_NOT_FINANCIAL_ADVICE"

# Keep all screened records so exclusions remain auditable.
result = result.sort_values(
    ["risk_profile", "horizon", "candidate_status",
     "signal_score", "asset_class"],
    ascending=[True, True, True, False, True]
)

parquet_path = OUT_DIR / "recommendation_candidates.parquet"
csv_path = OUT_DIR / "recommendation_candidates.csv"
summary_path = OUT_DIR / "recommendation_candidates_summary.json"

result.to_parquet(parquet_path, index=False)
result.to_csv(csv_path, index=False)

summary = {
    "source_file": str(files[-1].relative_to(ROOT)),
    "rows_written": int(len(result)),
    "profiles": {},
    "important_limitations": [
        "Scores are model probability differences, not expected returns.",
        "Risk-profile filters are initial defaults and are not validated investment rules.",
        "Signals use stored feature dates and may be stale.",
        "No verified fundamental score, valuation assessment, liquidity check, "
        "portfolio allocation or personalized suitability assessment is included."
    ]
}

for profile_name in profiles:
    part = result[result["risk_profile"] == profile_name]
    summary["profiles"][profile_name] = {
        "rows": int(len(part)),
        "shortlist_candidates": int(
            (part["candidate_status"] == "SHORTLIST_CANDIDATE").sum()
        ),
        "by_horizon": (
            part.groupby(["horizon", "candidate_status"])
                .size().unstack(fill_value=0).astype(int).to_dict("index")
        )
    }

summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

print("Recommendation candidate screening completed.")
print("Rows:", len(result))
print("\nStatus counts:")
print(result.groupby(["risk_profile", "candidate_status"]).size().to_string())
print("\nHorizon summary:")
print(
    result.groupby(["risk_profile", "horizon", "candidate_status"])
          .size().to_string()
)
print("\nSaved:")
print(parquet_path)
print(csv_path)
print(summary_path)
