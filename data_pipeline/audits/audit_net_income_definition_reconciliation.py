
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
    / "fundamental_observations_normalization_candidate.parquet"
)

OUTPUT_DIR = INPUT_PATH.parent

SCREENER_FIELD = "Net Profit +"

YAHOO_FIELDS = {
    "Net Income": "yahoo_net_income",
    "Net Income Common Stockholders": "yahoo_net_income_common",
    "Net Income From Continuing Operation Net Minority Interest":
        "yahoo_net_income_continuing_nci",
}

SCREENER_SOURCE = "Screener.in"
YAHOO_SOURCE = "Yahoo Finance via yfinance"

# Comparison tolerance is descriptive only; it is not an approval threshold.
CLOSE_TOLERANCE = 0.01


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    required = {
        "ticker",
        "source",
        "statement_type",
        "metric_raw",
        "period_end",
        "period_type",
        "reporting_basis",
        "normalization_candidate_value",
        "normalization_candidate_unit",
        "unit_verified",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["normalization_candidate_value"] = pd.to_numeric(
        df["normalization_candidate_value"], errors="coerce"
    )

    return df


def build_pairs(df):
    # Keep exact source fields. Do not combine distinct Yahoo definitions.
    screener = df[
        df["source"].eq(SCREENER_SOURCE)
        & df["metric_raw"].astype("string").str.strip().eq(SCREENER_FIELD)
        & df["statement_type"].isin(
            ["annual_income_statement", "quarterly_income_statement"]
        )
    ].copy()

    yahoo = df[
        df["source"].eq(YAHOO_SOURCE)
        & df["metric_raw"].isin(YAHOO_FIELDS)
        & df["statement_type"].isin(
            ["annual_income_statement", "quarterly_income_statement"]
        )
    ].copy()

    keys = ["ticker", "statement_type", "period_end"]

    # Check that each source has one value for each exact comparison key.
    screener_counts = (
        screener.groupby(keys, dropna=False)["normalization_candidate_value"]
        .nunique(dropna=True)
    )
    yahoo_counts = (
        yahoo.groupby(keys + ["metric_raw"], dropna=False)[
            "normalization_candidate_value"
        ].nunique(dropna=True)
    )

    screener_ambiguous = set(screener_counts[screener_counts > 1].index)
    yahoo_ambiguous = set(yahoo_counts[yahoo_counts > 1].index)

    screener["screener_ambiguous"] = [
        key in screener_ambiguous
        for key in screener[keys].itertuples(index=False, name=None)
    ]

    yahoo["yahoo_ambiguous"] = [
        key in yahoo_ambiguous
        for key in yahoo[
            keys + ["metric_raw"]
        ].itertuples(index=False, name=None)
    ]

    screener = screener[
        ~screener["screener_ambiguous"]
        & screener["normalization_candidate_value"].notna()
        & screener["period_end"].notna()
    ].copy()

    yahoo = yahoo[
        ~yahoo["yahoo_ambiguous"]
        & yahoo["normalization_candidate_value"].notna()
        & yahoo["period_end"].notna()
    ].copy()

    screener = screener[
        keys + [
            "normalization_candidate_value",
            "normalization_candidate_unit",
            "unit_verified",
            "reporting_basis",
            "period_type",
        ]
    ].rename(columns={
        "normalization_candidate_value": "screener_value_candidate",
        "normalization_candidate_unit": "screener_candidate_unit",
        "unit_verified": "screener_unit_verified",
        "reporting_basis": "screener_reporting_basis",
        "period_type": "screener_period_type",
    })

    yahoo = yahoo[
        keys + [
            "metric_raw",
            "normalization_candidate_value",
            "normalization_candidate_unit",
            "unit_verified",
            "reporting_basis",
            "period_type",
        ]
    ].rename(columns={
        "metric_raw": "yahoo_field",
        "normalization_candidate_value": "yahoo_value",
        "normalization_candidate_unit": "yahoo_candidate_unit",
        "unit_verified": "yahoo_unit_verified",
        "reporting_basis": "yahoo_reporting_basis",
        "period_type": "yahoo_period_type",
    })

    pairs = screener.merge(yahoo, on=keys, how="inner", validate="one_to_many")

    # Exact annual-to-annual and quarterly-to-quarterly matches only.
    pairs["period_class_match"] = (
        pairs["screener_period_type"].astype("string")
        == pairs["yahoo_period_type"].astype("string")
    )
    pairs["reporting_basis_match"] = (
        pairs["screener_reporting_basis"].astype("string")
        == pairs["yahoo_reporting_basis"].astype("string")
    )

    a = pairs["screener_value_candidate"].astype(float)
    b = pairs["yahoo_value"].astype(float)

    both_nonzero = a.ne(0) & b.ne(0)
    same_sign = np.sign(a) == np.sign(b)

    pairs["sign_disagreement"] = both_nonzero & ~same_sign
    pairs["zero_value_review"] = a.eq(0) | b.eq(0)

    pairs["absolute_percentage_difference"] = np.nan
    valid = both_nonzero & same_sign
    pairs.loc[valid, "absolute_percentage_difference"] = (
        (a[valid] - b[valid]).abs()
        / np.maximum(a[valid].abs(), b[valid].abs())
        * 100
    )

    pairs["comparison_status"] = "VALUE_DIFFERENCE_REVIEW"
    pairs.loc[pairs["zero_value_review"], "comparison_status"] = (
        "ZERO_VALUE_REVIEW"
    )
    pairs.loc[pairs["sign_disagreement"], "comparison_status"] = (
        "SIGN_DISAGREEMENT"
    )

    close = (
        pairs["absolute_percentage_difference"].notna()
        & pairs["absolute_percentage_difference"].le(CLOSE_TOLERANCE * 100)
    )
    pairs.loc[close, "comparison_status"] = "CLOSE_WITHIN_1_PERCENT"

    # Flag potentially incompatible reporting metadata rather than silently
    # excluding or relabelling observations.
    metadata_mismatch = (
        ~pairs["period_class_match"]
        | ~pairs["reporting_basis_match"]
    )
    pairs.loc[metadata_mismatch, "comparison_status"] = (
        "REPORTING_METADATA_REVIEW"
    )

    return pairs, screener, yahoo, screener_ambiguous, yahoo_ambiguous


def make_summary(pairs):
    if pairs.empty:
        return pd.DataFrame(columns=[
            "statement_type",
            "yahoo_field",
            "matched_observations",
            "distinct_tickers",
            "close_within_1_percent",
            "median_absolute_percentage_difference",
            "sign_disagreements",
            "zero_value_reviews",
            "metadata_review_rows",
            "verified_unit_pairs",
        ])

    summary = (
        pairs.groupby(["statement_type", "yahoo_field"], dropna=False)
        .agg(
            matched_observations=("ticker", "size"),
            distinct_tickers=("ticker", "nunique"),
            close_within_1_percent=(
                "comparison_status",
                lambda s: int((s == "CLOSE_WITHIN_1_PERCENT").sum()),
            ),
            median_absolute_percentage_difference=(
                "absolute_percentage_difference", "median"
            ),
            sign_disagreements=("sign_disagreement", "sum"),
            zero_value_reviews=("zero_value_review", "sum"),
            metadata_review_rows=(
                "comparison_status",
                lambda s: int((s == "REPORTING_METADATA_REVIEW").sum()),
            ),
        )
        .reset_index()
    )

    pairs = pairs.copy()
    pairs["both_units_verified"] = (
        pairs["screener_unit_verified"].fillna(False).astype(bool)
        & pairs["yahoo_unit_verified"].fillna(False).astype(bool)
    )

    verified = (
        pairs.groupby(["statement_type", "yahoo_field"])["both_units_verified"]
        .sum()
        .rename("verified_unit_pairs")
        .reset_index()
    )

    return summary.merge(
        verified, on=["statement_type", "yahoo_field"], how="left"
    )


def main():
    df = load_data()

    pairs, screener, yahoo, screener_ambiguous, yahoo_ambiguous = (
        build_pairs(df)
    )

    summary = make_summary(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs_path = OUTPUT_DIR / "net_income_definition_pairs.csv"
    summary_path = OUTPUT_DIR / "net_income_definition_summary.csv"
    screener_path = OUTPUT_DIR / "net_income_screener_coverage.csv"
    yahoo_path = OUTPUT_DIR / "net_income_yahoo_field_coverage.csv"

    pairs.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)

    screener_coverage = (
        screener.groupby("statement_type")
        .agg(
            observations=("ticker", "size"),
            tickers=("ticker", "nunique"),
            periods=("period_end", "nunique"),
        )
        .reset_index()
    )
    screener_coverage.to_csv(screener_path, index=False)

    yahoo_coverage = (
        yahoo.groupby(["statement_type", "yahoo_field"])
        .agg(
            observations=("ticker", "size"),
            tickers=("ticker", "nunique"),
            periods=("period_end", "nunique"),
        )
        .reset_index()
    )
    yahoo_coverage.to_csv(yahoo_path, index=False)

    print("=" * 72)
    print("NET-INCOME DEFINITION RECONCILIATION — READ ONLY")
    print("=" * 72)
    print("Screener observations retained:", len(screener))
    print("Yahoo observations retained:", len(yahoo))
    print("Matched source-field pairs:", len(pairs))
    print("Ambiguous Screener keys excluded:", len(screener_ambiguous))
    print("Ambiguous Yahoo field keys excluded:", len(yahoo_ambiguous))
    print("\nSummary by statement type and Yahoo field:")
    if summary.empty:
        print("No comparable observations found.")
    else:
        print(summary.to_string(index=False))

    print("\nMetadata checks:")
    if not pairs.empty:
        print(
            "Reporting-basis matches:",
            int(pairs["reporting_basis_match"].sum()),
            "/",
            len(pairs),
        )
        print(
            "Period-class matches:",
            int(pairs["period_class_match"].sum()),
            "/",
            len(pairs),
        )
        print(
            "Pairs with both units verified:",
            int(
                (
                    pairs["screener_unit_verified"].fillna(False).astype(bool)
                    & pairs["yahoo_unit_verified"].fillna(False).astype(bool)
                ).sum()
            ),
        )

    print("\nOutput files:")
    for p in [pairs_path, summary_path, screener_path, yahoo_path]:
        print(p.relative_to(ROOT))

    print("\nNo source data modified. No metric approved for scoring.")


if __name__ == "__main__":
    main()
