
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "clean" / "fundamentals"
NORMALIZATION_DIR = BASE / "normalization_candidate"
INTEGRATION_DIR = BASE / "integration_candidate"

OBSERVATIONS_PATH = (
    INTEGRATION_DIR / "latest_fundamental_observations_candidate.parquet"
)
CURRENCY_PATH = NORMALIZATION_DIR / "ticker_currency_reference_audit.parquet"

OUTPUT_DIR = INTEGRATION_DIR / "total_assets_validation"
PAIRS_PATH = OUTPUT_DIR / "total_assets_source_comparison.csv"
DUPLICATES_PATH = OUTPUT_DIR / "total_assets_duplicate_review.csv"
SUMMARY_PATH = OUTPUT_DIR / "total_assets_validation_summary.json"

SCREENER_FIELD = "Total Assets"
YAHOO_FIELD = "Total Assets"
CANDIDATE_SCALE = 1e7
LOG_TOLERANCE = 0.25


def require_columns(df, required, label):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def canonical_ticker(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return {"M_M.NS": "M&M.NS"}.get(value, value)


def prepare_source(df, source_name):
    """Collapse only exact numeric duplicates; exclude conflicting keys."""
    keys = ["ticker", "period_end"]

    if df.empty:
        return df[keys + ["original_numeric_value"]].copy(), pd.DataFrame(
            columns=["ticker", "period_end", "source", "row_count",
                     "distinct_numeric_values", "values", "duplicate_status"]
        )

    duplicate_rows = df[df.duplicated(keys, keep=False)].copy()
    review_records = []
    accepted_records = []

    for key, group in df.groupby(keys, dropna=False, sort=False):
        values = group["original_numeric_value"].dropna()
        distinct = values.nunique()

        if len(group) == 1:
            if len(values) == 1:
                accepted_records.append({
                    "ticker": key[0],
                    "period_end": key[1],
                    "original_numeric_value": values.iloc[0],
                })
            continue

        if distinct == 1 and len(values) == len(group):
            accepted_records.append({
                "ticker": key[0],
                "period_end": key[1],
                "original_numeric_value": values.iloc[0],
            })
            status = "IDENTICAL_VALUES_COLLAPSED_FOR_COMPARISON"
        elif distinct == 0:
            status = "ALL_VALUES_MISSING_EXCLUDED"
        else:
            status = "CONFLICTING_VALUES_EXCLUDED"

        review_records.append({
            "ticker": key[0],
            "period_end": (
                key[1].isoformat() if pd.notna(key[1]) else None
            ),
            "source": source_name,
            "row_count": int(len(group)),
            "distinct_numeric_values": int(distinct),
            "values": " | ".join(
                map(str, sorted(values.unique().tolist()))
            ),
            "duplicate_status": status,
        })

    accepted = pd.DataFrame(
        accepted_records,
        columns=["ticker", "period_end", "original_numeric_value"],
    )
    review = pd.DataFrame(review_records)

    return accepted, review


def main():
    for path in [OBSERVATIONS_PATH, CURRENCY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    obs = pd.read_parquet(OBSERVATIONS_PATH)
    currency = pd.read_parquet(CURRENCY_PATH)

    require_columns(
        obs,
        [
            "ticker", "source", "metric_raw", "period_end",
            "original_numeric_value", "currency_quarantine",
        ],
        "Latest observations",
    )
    require_columns(
        currency,
        [
            "ticker", "currency_status", "financial_currency",
            "currency_review_required",
        ],
        "Currency reference",
    )

    obs["ticker"] = obs["ticker"].map(canonical_ticker)
    currency["ticker"] = currency["ticker"].map(canonical_ticker)
    obs["period_end"] = pd.to_datetime(
        obs["period_end"], errors="coerce"
    )
    obs["original_numeric_value"] = pd.to_numeric(
        obs["original_numeric_value"], errors="coerce"
    )

    if currency["ticker"].duplicated().any():
        raise ValueError("Duplicate canonical tickers in currency reference.")

    # Exact raw field names and source selection.
    screener_raw = obs[
        (obs["source"].astype(str).str.strip() == "Screener.in")
        & (obs["metric_raw"].astype(str).str.strip() == SCREENER_FIELD)
    ].copy()

    yahoo_raw = obs[
        (obs["source"].astype(str).str.contains("yahoo", case=False, na=False))
        & (obs["metric_raw"].astype(str).str.strip() == YAHOO_FIELD)
    ].copy()

    for frame in [screener_raw, yahoo_raw]:
        frame.dropna(
            subset=["ticker", "period_end"], inplace=True
        )

    screener, screener_review = prepare_source(
        screener_raw, "Screener.in"
    )
    yahoo, yahoo_review = prepare_source(yahoo_raw, "Yahoo")

    duplicate_review = pd.concat(
        [screener_review, yahoo_review],
        ignore_index=True,
    )

    # Create the output directory before writing any files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not duplicate_review.empty:
        duplicate_review.to_csv(DUPLICATES_PATH, index=False)

    screener = screener.rename(columns={
        "original_numeric_value": "screener_value"
    })
    yahoo = yahoo.rename(columns={
        "original_numeric_value": "yahoo_value"
    })

    pairs = screener.merge(
        yahoo,
        on=["ticker", "period_end"],
        how="inner",
        validate="one_to_one",
    )

    pairs = pairs.merge(
        currency[
            ["ticker", "currency_status", "financial_currency",
             "currency_review_required"]
        ],
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    # The source-level quarantine flag is retained from the existing audit.
    quarantine_tickers = set(
        obs.loc[
            obs["currency_quarantine"].fillna(True).astype(bool),
            "ticker",
        ].dropna().unique()
    )

    pairs["currency_quarantine"] = (
        pairs["ticker"].isin(quarantine_tickers)
        | pairs["currency_review_required"].fillna(True).astype(bool)
    )

    pairs["both_positive"] = (
        (pairs["screener_value"] > 0)
        & (pairs["yahoo_value"] > 0)
    )

    pairs["scale_ratio_yahoo_to_screener"] = np.where(
        pairs["both_positive"],
        pairs["yahoo_value"] / pairs["screener_value"],
        np.nan,
    )

    pairs["candidate_screener_scaled_value"] = (
        pairs["screener_value"] * CANDIDATE_SCALE
    )

    pairs["candidate_abs_pct_difference"] = np.where(
        pairs["both_positive"],
        (
            pairs["candidate_screener_scaled_value"] - pairs["yahoo_value"]
        ).abs() / pairs["yahoo_value"].abs() * 100,
        np.nan,
    )

    pairs["candidate_log10_scale_distance"] = np.where(
        pairs["both_positive"],
        np.abs(
            np.log10(pairs["scale_ratio_yahoo_to_screener"])
            - np.log10(CANDIDATE_SCALE)
        ),
        np.nan,
    )

    pairs["near_candidate_scale"] = (
        pairs["candidate_log10_scale_distance"] <= LOG_TOLERANCE
    )
    pairs["candidate_difference_within_1pct"] = (
        pairs["candidate_abs_pct_difference"] <= 1.0
    )
    pairs["sign_disagreement"] = (
        np.sign(pairs["screener_value"])
        != np.sign(pairs["yahoo_value"])
    )

    pairs["comparison_status"] = np.select(
        [
            pairs["currency_quarantine"],
            ~pairs["both_positive"],
            pairs["candidate_difference_within_1pct"],
            pairs["near_candidate_scale"],
        ],
        [
            "CURRENCY_REVIEW_EXCLUDED",
            "NONPOSITIVE_OR_INVALID_VALUE",
            "CANDIDATE_SCALE_MATCH_WITHIN_1PCT",
            "CANDIDATE_SCALE_APPROXIMATE_REVIEW",
        ],
        default="CANDIDATE_SCALE_MISMATCH_REVIEW",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(PAIRS_PATH, index=False)

    usable = pairs[
        pairs["both_positive"] & ~pairs["currency_quarantine"]
    ].copy()

    ratio = usable["scale_ratio_yahoo_to_screener"].dropna()
    pct_diff = usable["candidate_abs_pct_difference"].dropna()
    log_distance = usable["candidate_log10_scale_distance"].dropna()

    duplicate_status_counts = (
        duplicate_review["duplicate_status"].value_counts().to_dict()
        if not duplicate_review.empty else {}
    )

    summary = {
        "stage": "total_assets_mapping_validation_candidate",
        "screener_source_field": SCREENER_FIELD,
        "yahoo_source_field": YAHOO_FIELD,
        "candidate_scale_factor": CANDIDATE_SCALE,
        "candidate_scale_factor_is_approved": False,
        "mapping_approved_for_scoring": False,
        "unit_verified": False,
        "financial_values_independently_verified": False,
        "scoring_eligible_rows": 0,
        "screener_raw_rows": int(len(screener_raw)),
        "yahoo_raw_rows": int(len(yahoo_raw)),
        "screener_comparison_keys_retained": int(len(screener)),
        "yahoo_comparison_keys_retained": int(len(yahoo)),
        "exact_ticker_date_matched_pairs": int(len(pairs)),
        "usable_positive_pairs_outside_currency_quarantine": int(len(usable)),
        "currency_quarantined_pairs": int(
            pairs["currency_quarantine"].sum()
        ),
        "positive_pairs_near_candidate_scale": int(
            usable["near_candidate_scale"].sum()
        ),
        "positive_pairs_within_1pct_after_candidate_scaling": int(
            usable["candidate_difference_within_1pct"].sum()
        ),
        "positive_pairs_with_sign_disagreement": int(
            usable["sign_disagreement"].sum()
        ),
        "duplicate_review_rows": int(len(duplicate_review)),
        "duplicate_review_status_counts": {
            str(k): int(v) for k, v in duplicate_status_counts.items()
        },
        "median_yahoo_to_screener_ratio": (
            float(ratio.median()) if not ratio.empty else None
        ),
        "median_candidate_abs_pct_difference": (
            float(pct_diff.median()) if not pct_diff.empty else None
        ),
        "median_log10_distance_from_candidate_scale": (
            float(log_distance.median())
            if not log_distance.empty else None
        ),
        "comparison_status_counts": {
            str(k): int(v)
            for k, v in pairs["comparison_status"]
            .value_counts(dropna=False).items()
        },
        "outputs": [
            str(PAIRS_PATH.relative_to(ROOT)),
            str(DUPLICATES_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
        ],
        "limitations": [
            "This uses the latest retained observation per source field.",
            "Only exact ticker and reporting-date matches are compared.",
            "Identical duplicate numeric values are collapsed for comparison only.",
            "Conflicting duplicate values are excluded and reported separately.",
            "A scale relationship does not independently verify units.",
            "No mapping, normalization, or scoring approval is granted.",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("TOTAL ASSETS MAPPING VALIDATION")
    print("=" * 72)
    print("Screener raw rows:", len(screener_raw))
    print("Yahoo raw rows:", len(yahoo_raw))
    print("Screener comparison keys:", len(screener))
    print("Yahoo comparison keys:", len(yahoo))
    print("Exact ticker/date matched pairs:", len(pairs))
    print("Usable positive pairs outside currency quarantine:", len(usable))
    print(
        "Pairs near candidate scale:",
        summary["positive_pairs_near_candidate_scale"],
    )
    print(
        "Pairs within 1% after candidate scaling:",
        summary["positive_pairs_within_1pct_after_candidate_scaling"],
    )
    print("Duplicate review rows:", len(duplicate_review))
    print("Duplicate review statuses:", duplicate_status_counts)
    print("Median Yahoo/Screener ratio:", summary[
        "median_yahoo_to_screener_ratio"
    ])
    print("Mapping approved: False")
    print("Scoring eligible rows: 0")
    print("\nComparison:", PAIRS_PATH.relative_to(ROOT))
    print("Duplicate review:", DUPLICATES_PATH.relative_to(ROOT))
    print("Summary:", SUMMARY_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
