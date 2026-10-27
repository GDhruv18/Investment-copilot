
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "integration_candidate"
    / "fundamental_observations_candidate.parquet"
)

MAPPING_PATH = (
    ROOT
    / "data_pipeline"
    / "config"
    / "stock_fundamental_metric_map.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "normalization_candidate"
)

OUTPUT_PATH = OUTPUT_DIR / "fundamental_observations_normalization_candidate.parquet"

# Empirical candidate factor supported by the prior cross-source audits.
# This is NOT a declaration that source units are formally verified.
SCREENER_CANDIDATE_FACTOR = 10_000_000

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}

SOURCE_SCREENER = "Screener.in"
SOURCE_YAHOO = "Yahoo Finance via yfinance"


def load_inputs() -> tuple[pd.DataFrame, dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_PATH}")

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Missing metric map: {MAPPING_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    required = {
        "ticker",
        "source",
        "metric_raw",
        "period_end",
        "value",
        "raw_value",
        "unit",
        "unit_verified",
        "financial_value_independently_verified",
        "scoring_eligible",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Integration dataset is missing columns: {sorted(missing)}"
        )

    if "metrics" not in mapping:
        raise ValueError("Metric map does not contain a 'metrics' object.")

    return df, mapping


def build_candidate_map(mapping: dict) -> dict:
    """
    Build exact, source-specific metric-name lookup.
    Do not infer equivalence from partial names.
    """
    result = {}

    for canonical_metric, spec in mapping["metrics"].items():
        unit_type = spec.get("unit_type", "unknown")
        mapping_status = spec.get("mapping_status", "review_required")
        candidates = spec.get("candidates", {})

        for source, names in candidates.items():
            for name in names:
                key = (source, str(name).strip())
                result.setdefault(key, []).append({
                    "canonical_metric": canonical_metric,
                    "unit_type": unit_type,
                    "mapping_status": mapping_status,
                })

    return result


def classify_mapping(
    source: str,
    metric_raw: str,
    candidate_map: dict,
) -> tuple[str | None, str, str]:
    matches = candidate_map.get((source, metric_raw.strip()), [])

    if not matches:
        return None, "unknown", "unmapped"

    canonical_metrics = sorted({
        item["canonical_metric"] for item in matches
    })
    unit_types = sorted({
        item["unit_type"] for item in matches
    })
    statuses = sorted({
        item["mapping_status"] for item in matches
    })

    canonical_text = "|".join(canonical_metrics)
    unit_text = "|".join(unit_types)

    if len(canonical_metrics) > 1 or len(unit_types) > 1:
        status = "ambiguous_mapping_review_required"
    else:
        status = statuses[0] if len(statuses) == 1 else "review_required"

    return canonical_text, unit_text, status


def transform(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    candidate_map = build_candidate_map(mapping)
    result = df.copy()

    # Preserve the original value explicitly, independent of raw_value.
    result["original_numeric_value"] = pd.to_numeric(
        result["value"], errors="coerce"
    )

    canonical_values = []
    unit_types = []
    mapping_statuses = []

    for row in result.itertuples(index=False):
        canonical, unit_type, status = classify_mapping(
            str(row.source),
            str(row.metric_raw),
            candidate_map,
        )
        canonical_values.append(canonical)
        unit_types.append(unit_type)
        mapping_statuses.append(status)

    result["canonical_metric_candidate"] = canonical_values
    result["mapped_unit_type"] = unit_types
    result["metric_mapping_status"] = mapping_statuses

    result["normalization_candidate_value"] = result[
        "original_numeric_value"
    ]

    result["normalization_candidate_unit"] = result["unit"].astype("string")
    result["normalization_rule"] = "none"
    result["normalization_status"] = "unchanged_unmapped_or_unapproved"

    screener_rows = result["source"].eq(SOURCE_SCREENER)
    yahoo_rows = result["source"].eq(SOURCE_YAHOO)

    currency_amount = result["mapped_unit_type"].eq("currency_amount")
    mapped = result["canonical_metric_candidate"].notna()

    # Apply the candidate factor only to explicitly mapped Screener
    # currency-amount observations. Never transform EPS, ratios, or
    # percentages. Retain the original value and source unit.
    convert_mask = screener_rows & currency_amount & mapped

    result.loc[
        convert_mask, "normalization_candidate_value"
    ] = (
        result.loc[convert_mask, "original_numeric_value"]
        * SCREENER_CANDIDATE_FACTOR
    )

    result.loc[
        convert_mask, "normalization_candidate_unit"
    ] = "INR_candidate_unverified"

    result.loc[
        convert_mask, "normalization_rule"
    ] = "Screener_assumed_crore_to_INR_x_1e7_candidate"

    result.loc[
        convert_mask, "normalization_status"
    ] = "candidate_scale_applied_metric_mapping_unapproved"

    # Screener observations not selected for conversion remain unchanged.
    screener_unchanged = screener_rows & ~convert_mask
    result.loc[
        screener_unchanged, "normalization_status"
    ] = "unchanged_no_approved_currency_amount_mapping"

    # Yahoo statement values remain unchanged. We do not assume that
    # every Yahoo value is INR or that currency metadata is verified.
    result.loc[
        yahoo_rows, "normalization_candidate_unit"
    ] = "provider_unit_unverified"

    result.loc[
        yahoo_rows, "normalization_rule"
    ] = "none_yahoo_value_preserved"

    result.loc[
        yahoo_rows, "normalization_status"
    ] = "unchanged_yahoo_unit_unverified"

    # Preserve the universe but quarantine known currency/scale issues.
    result["fundamental_review_flag"] = ""

    result.loc[
        result["ticker"].isin(QUARANTINED_TICKERS),
        "fundamental_review_flag",
    ] = "currency_or_scale_review_required"

    # A transformed value is a candidate, not scoring approval.
    result["unit_verified"] = False
    result["financial_value_independently_verified"] = False
    result["scoring_eligible"] = False

    result["eligibility_reason"] = (
        "Normalization candidate only; unit, metric definition, "
        "reporting basis, and point-in-time availability not fully approved"
    )

    # Defensive checks: no rows should become scoring eligible.
    if result["scoring_eligible"].any():
        raise AssertionError("Scoring eligibility must remain disabled.")

    if result["ticker"].nunique() != df["ticker"].nunique():
        raise AssertionError("The stock universe changed unexpectedly.")

    if len(result) != len(df):
        raise AssertionError("The number of observations changed.")

    if not result["raw_value"].equals(df["raw_value"]):
        raise AssertionError("Original raw values were modified.")

    return result


def main() -> None:
    df, mapping = load_inputs()
    result = transform(df, mapping)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print("=" * 68)
    print("FUNDAMENTAL NORMALIZATION CANDIDATE — COMPLETE")
    print("=" * 68)
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(result):,}")
    print(f"Unique tickers: {result['ticker'].nunique():,}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Normalization status:")
    print(result["normalization_status"].value_counts().to_string())
    print()
    print("Mapping status:")
    print(result["metric_mapping_status"].value_counts().to_string())
    print()
    print(
        "Screener candidate-scale rows: "
        f"{int(result['normalization_rule'].eq('Screener_assumed_crore_to_INR_x_1e7_candidate').sum()):,}"
    )
    print(
        "Rows flagged for currency/scale review: "
        f"{int(result['fundamental_review_flag'].ne('').sum()):,}"
    )
    print(
        "Rows eligible for scoring: "
        f"{int(result['scoring_eligible'].sum()):,}"
    )
    print()
    print("Raw input dataset was not modified.")
    print("The candidate output is not approved for recommendation scoring.")


if __name__ == "__main__":
    main()
