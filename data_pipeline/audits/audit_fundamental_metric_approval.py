
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
    / "fundamental_observations_normalization_candidate.parquet"
)

OUTPUT_DIR = INPUT_PATH.parent

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}

REVIEW_METRICS = {
    "revenue",
    "net_income",
    "operating_profit",
    "total_assets",
    "total_liabilities",
    "total_debt",
    "operating_cash_flow",
    "free_cash_flow",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {
        "ticker", "source", "statement_type", "metric_raw",
        "period_end", "original_numeric_value",
        "canonical_metric_candidate", "normalization_candidate_value",
        "normalization_rule", "normalization_status",
        "metric_mapping_status", "unit_verified", "scoring_eligible",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["original_numeric_value"] = pd.to_numeric(
        df["original_numeric_value"], errors="coerce"
    )
    df["normalization_candidate_value"] = pd.to_numeric(
        df["normalization_candidate_value"], errors="coerce"
    )
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["source"] = df["source"].astype("string").fillna("UNKNOWN")
    df["canonical_metric_candidate"] = (
        df["canonical_metric_candidate"].astype("string").fillna("UNMAPPED")
    )
    df["is_scaled"] = df["normalization_rule"].eq(
        "Screener_assumed_crore_to_INR_x_1e7_candidate"
    )
    df["is_quarantined"] = df["ticker"].isin(QUARANTINED_TICKERS)

    # 1. Coverage by candidate metric and source.
    coverage = (
        df.groupby(
            ["canonical_metric_candidate", "source"],
            dropna=False,
        )
        .agg(
            observations=("ticker", "size"),
            distinct_tickers=("ticker", "nunique"),
            distinct_periods=("period_end", "nunique"),
            valid_original_values=("original_numeric_value", "count"),
            valid_candidate_values=("normalization_candidate_value", "count"),
            candidate_scaled_rows=("is_scaled", "sum"),
            quarantined_rows=("is_quarantined", "sum"),
            raw_metric_names=(
                "metric_raw",
                lambda x: " | ".join(sorted(set(x.dropna().astype(str)))),
            ),
            mapping_statuses=(
                "metric_mapping_status",
                lambda x: " | ".join(sorted(set(x.dropna().astype(str)))),
            ),
        )
        .reset_index()
        .sort_values(
            ["canonical_metric_candidate", "source"], kind="stable"
        )
    )

    # 2. Duplicate-key audit. Source is part of the key because different
    # sources are expected to contain observations for the same metric/date.
    key_cols = [
        "ticker", "source", "canonical_metric_candidate",
        "statement_type", "period_end",
    ]
    df["duplicate_key"] = df.duplicated(key_cols, keep=False)

    duplicates = (
        df.groupby(
            ["source", "canonical_metric_candidate"], dropna=False
        )
        .agg(
            observations=("ticker", "size"),
            rows_in_duplicate_keys=("duplicate_key", "sum"),
        )
        .reset_index()
    )

    # 3. Cross-source comparison. Compare only exact ticker, candidate metric,
    # statement type and period. Do not compare ratios with monetary amounts,
    # or annual observations with quarterly observations.
    compare = df[
        df["canonical_metric_candidate"].ne("UNMAPPED")
        & df["period_end"].notna()
        & df["normalization_candidate_value"].notna()
    ].copy()

    pair_keys = [
        "ticker", "canonical_metric_candidate",
        "statement_type", "period_end",
    ]

    pair_rows = []

    for key, group in compare.groupby(pair_keys, dropna=False, sort=False):
        if group["source"].nunique() < 2:
            continue

        source_counts = group.groupby("source")[
            "normalization_candidate_value"
        ].nunique(dropna=True)

        if (source_counts > 1).any():
            pair_rows.append({
                **dict(zip(pair_keys, key)),
                "source_a": None,
                "source_b": None,
                "value_a": np.nan,
                "value_b": np.nan,
                "absolute_ratio_a_to_b": np.nan,
                "pair_status": "AMBIGUOUS_MULTIPLE_VALUES_WITHIN_SOURCE",
            })
            continue

        values = (
            group.drop_duplicates("source")
            .set_index("source")["normalization_candidate_value"]
        )
        sources = list(values.index)

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source_a, source_b = sources[i], sources[j]
                a, b = float(values[source_a]), float(values[source_b])

                if a == 0 or b == 0:
                    ratio = np.nan
                    status = "ZERO_VALUE_REVIEW"
                elif np.sign(a) != np.sign(b):
                    ratio = abs(a / b)
                    status = "SIGN_DISAGREEMENT"
                else:
                    ratio = abs(a / b)
                    status = (
                        "CLOSE_WITHIN_1_PERCENT"
                        if abs(np.log(ratio)) <= np.log(1.01)
                        else "SOURCE_VALUE_DIFFERENCE_REVIEW"
                    )

                pair_rows.append({
                    **dict(zip(pair_keys, key)),
                    "source_a": source_a,
                    "source_b": source_b,
                    "value_a": a,
                    "value_b": b,
                    "absolute_ratio_a_to_b": ratio,
                    "pair_status": status,
                })

    pair_audit = pd.DataFrame(pair_rows)

    if pair_audit.empty:
        pair_summary = pd.DataFrame(
            columns=["canonical_metric_candidate", "pair_status", "pair_count"]
        )
    else:
        pair_summary = (
            pair_audit.groupby(
                ["canonical_metric_candidate", "pair_status"],
                dropna=False,
            )
            .size()
            .rename("pair_count")
            .reset_index()
        )

    # 4. Conservative disposition. This script never approves a metric.
    metric_rows = []

    for metric, group in df.groupby(
        "canonical_metric_candidate", dropna=False
    ):
        verified = group["unit_verified"].fillna(False).astype(bool)
        eligible = group["scoring_eligible"].fillna(False).astype(bool)

        if metric == "UNMAPPED":
            disposition = "UNMAPPED_EXCLUDE_FROM_SCORING"
        else:
            disposition = "REVIEW_REQUIRED_NOT_APPROVED"

        if group["is_quarantined"].any():
            disposition += "_QUARANTINE_PRESENT"

        metric_rows.append({
            "canonical_metric_candidate": metric,
            "observations": len(group),
            "distinct_tickers": group["ticker"].nunique(),
            "candidate_scaled_rows": int(group["is_scaled"].sum()),
            "unit_verified_rows": int(verified.sum()),
            "scoring_eligible_rows": int(eligible.sum()),
            "quarantined_rows": int(group["is_quarantined"].sum()),
            "distinct_raw_metric_names": group["metric_raw"].nunique(),
            "workflow_disposition": disposition,
        })

    metric_review = pd.DataFrame(metric_rows).sort_values(
        "canonical_metric_candidate", kind="stable"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "metric_source_coverage_audit.csv": coverage,
        "metric_duplicate_key_audit.csv": duplicates,
        "metric_source_pair_audit.csv": pair_audit,
        "metric_source_pair_summary.csv": pair_summary,
        "metric_approval_review.csv": metric_review,
    }

    for filename, table in outputs.items():
        table.to_csv(OUTPUT_DIR / filename, index=False)

    summary = {
        "audit": "fundamental_metric_approval_review",
        "input_rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()),
        "candidate_scaled_rows": int(df["is_scaled"].sum()),
        "verified_unit_rows": int(
            df["unit_verified"].fillna(False).astype(bool).sum()
        ),
        "scoring_eligible_rows": int(
            df["scoring_eligible"].fillna(False).astype(bool).sum()
        ),
        "cross_source_pair_rows": int(len(pair_audit)),
        "metric_count_including_unmapped": int(
            df["canonical_metric_candidate"].nunique()
        ),
        "quarantined_tickers": sorted(QUARANTINED_TICKERS),
        "approval_policy": (
            "Read-only screening. No metric approved automatically. "
            "Units and financial definitions require independent verification."
        ),
        "outputs": list(outputs.keys()),
    }

    summary_path = OUTPUT_DIR / "metric_approval_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 68)
    print("FUNDAMENTAL METRIC APPROVAL REVIEW — READ ONLY")
    print("=" * 68)
    print(f"Rows: {len(df):,}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print(f"Candidate-scaled rows: {int(df['is_scaled'].sum()):,}")
    print(f"Verified-unit rows: {summary['verified_unit_rows']:,}")
    print(f"Scoring-eligible rows: {summary['scoring_eligible_rows']:,}")
    print(f"Cross-source pair rows: {len(pair_audit):,}")

    print("\nMETRIC REVIEW")
    print(
        metric_review[
            [
                "canonical_metric_candidate",
                "observations",
                "distinct_tickers",
                "candidate_scaled_rows",
                "workflow_disposition",
            ]
        ].to_string(index=False)
    )

    print("\nCROSS-SOURCE PAIR SUMMARY")
    if pair_summary.empty:
        print("No exact-key cross-source pairs found.")
    else:
        print(pair_summary.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)
    print("\nNo input dataset was modified; no metric was approved.")


if __name__ == "__main__":
    main()
