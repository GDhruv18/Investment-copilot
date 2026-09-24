"""
Generate Final Stock 2Y Targets
===============================

LOCKED Stock 2Y methodology:

    50% Return Quality
    25% Risk Quality
    25% Drawdown Quality

    Minimum cohort = 80 stocks

    Composite score is ranked cross-sectionally by date.

    Bottom 30%  -> Weak
    Middle 40%  -> Neutral
    Top 30%     -> Attractive

This script generates:
    data/targets/stock_2y_targets.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/targets/stock_2y_diagnostics.parquet"
)

OUTPUT_FILE = Path(
    "data/targets/stock_2y_targets.parquet"
)

MIN_COHORT = 80

RETURN_WEIGHT = 0.50
RISK_WEIGHT = 0.25
DRAWDOWN_WEIGHT = 0.25


# ============================================================
# HELPERS
# ============================================================

def print_separator(char="=", width=78):
    print(char * width)


def percentile_rank(series):
    return series.rank(
        method="average",
        pct=True,
    )


def inverse_percentile_rank(series):
    """
    Lower volatility = better risk quality.
    """

    n = series.notna().sum()

    if n == 0:
        return series

    return (
        1.0
        - series.rank(
            method="average",
            pct=True,
        )
        + (1.0 / n)
    )


def label_from_percentile(value):

    if pd.isna(value):
        return np.nan

    if value <= 0.30:
        return "Weak"

    if value <= 0.70:
        return "Neutral"

    return "Attractive"


# ============================================================
# START
# ============================================================

print_separator()

print(
    "GENERATE FINAL STOCK 2Y TARGETS"
)

print_separator()


# ============================================================
# INPUT CHECK
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


print(
    f"Input file: {INPUT_FILE}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    INPUT_FILE
)


print()
print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Stocks: {df['ticker'].nunique():,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "ticker",
    "date",
    "adj_close",
    "target_date_2y",
    "future_date_2y",
    "future_adj_close_2y",
    "actual_years_2y",
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# DATE NORMALIZATION
# ============================================================

df["date"] = pd.to_datetime(
    df["date"]
)

df["target_date_2y"] = pd.to_datetime(
    df["target_date_2y"]
)

df["future_date_2y"] = pd.to_datetime(
    df["future_date_2y"]
)


print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# SOURCE INTEGRITY
# ============================================================

print()
print_separator("-")

print(
    "SOURCE INTEGRITY VALIDATION"
)

print_separator("-")


duplicate_count = (
    df.duplicated(
        subset=[
            "ticker",
            "date",
        ]
    ).sum()
)


print(
    f"Duplicate ticker/date rows: "
    f"{duplicate_count:,}"
)


if duplicate_count != 0:
    raise ValueError(
        "Duplicate ticker/date rows detected."
    )


print(
    "Source integrity: PASS"
)


# ============================================================
# VALID 2Y OBSERVATIONS
# ============================================================

valid_mask = (
    df["annualized_return_2y"].notna()
    &
    df["annualized_volatility_2y"].notna()
    &
    df["max_drawdown_2y"].notna()
)


valid = df.loc[
    valid_mask
].copy()


missing_count = (
    len(df)
    - len(valid)
)


print()
print_separator("-")

print(
    "2Y OUTCOME COVERAGE"
)

print_separator("-")


print(
    f"Valid 2Y outcomes: "
    f"{len(valid):,}"
)

print(
    f"Missing/incomplete 2Y outcomes: "
    f"{missing_count:,}"
)


# ============================================================
# COHORT FILTER
# ============================================================

cohort_sizes = (
    valid
    .groupby("date")["ticker"]
    .nunique()
)


eligible_dates = cohort_sizes[
    cohort_sizes >= MIN_COHORT
].index


print()
print_separator("-")

print(
    "CROSS-SECTIONAL COHORT FILTER"
)

print_separator("-")


print(
    f"Minimum cohort required: "
    f"{MIN_COHORT}"
)

print(
    f"Total valid dates: "
    f"{len(cohort_sizes):,}"
)

print(
    f"Eligible dates: "
    f"{len(eligible_dates):,}"
)


scoring = valid[
    valid["date"].isin(
        eligible_dates
    )
].copy()


print(
    f"Rows eligible for scoring: "
    f"{len(scoring):,}"
)


actual_min_cohort = (
    scoring
    .groupby("date")["ticker"]
    .nunique()
    .min()
)


actual_max_cohort = (
    scoring
    .groupby("date")["ticker"]
    .nunique()
    .max()
)


print(
    f"Actual cohort range: "
    f"{actual_min_cohort} → "
    f"{actual_max_cohort}"
)


if actual_min_cohort < MIN_COHORT:
    raise ValueError(
        "Cohort filter validation failed."
    )


# ============================================================
# COMPONENT PERCENTILES
# ============================================================

print()
print_separator("-")

print(
    "COMPONENT PERCENTILES"
)

print_separator("-")


# Higher return = better
scoring["return_percentile_2y"] = (
    scoring
    .groupby("date")[
        "annualized_return_2y"
    ]
    .transform(
        percentile_rank
    )
)


# Lower volatility = better
scoring["risk_percentile_2y"] = (
    scoring
    .groupby("date")[
        "annualized_volatility_2y"
    ]
    .transform(
        inverse_percentile_rank
    )
)


# Less-negative drawdown = better
scoring["drawdown_percentile_2y"] = (
    scoring
    .groupby("date")[
        "max_drawdown_2y"
    ]
    .transform(
        percentile_rank
    )
)


component_columns = [
    "return_percentile_2y",
    "risk_percentile_2y",
    "drawdown_percentile_2y",
]


print()

for column in component_columns:

    invalid = (
        scoring[column].isna()
        |
        (scoring[column] < 0)
        |
        (scoring[column] > 1)
    ).sum()

    print(
        f"{column:28s} "
        f"min={scoring[column].min():.6f} "
        f"max={scoring[column].max():.6f} "
        f"invalid={invalid}"
    )


if (
    scoring[component_columns]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        "Component percentile calculation failed."
    )


print(
    "Component percentile validation: PASS"
)


# ============================================================
# COMPOSITE SCORE
# ============================================================

print()
print_separator("-")

print(
    "LOCKED STOCK 2Y COMPOSITE SCORE"
)

print_separator("-")


print(
    f"Return weight:   {RETURN_WEIGHT:.0%}"
)

print(
    f"Risk weight:     {RISK_WEIGHT:.0%}"
)

print(
    f"Drawdown weight: {DRAWDOWN_WEIGHT:.0%}"
)


if not np.isclose(
    RETURN_WEIGHT
    + RISK_WEIGHT
    + DRAWDOWN_WEIGHT,
    1.0,
):
    raise ValueError(
        "Composite weights do not sum to 100%."
    )


scoring["composite_score_2y"] = (
    RETURN_WEIGHT
    * scoring["return_percentile_2y"]

    +

    RISK_WEIGHT
    * scoring["risk_percentile_2y"]

    +

    DRAWDOWN_WEIGHT
    * scoring["drawdown_percentile_2y"]
)


print(
    f"Composite score range: "
    f"{scoring['composite_score_2y'].min():.6f}"
    f" → "
    f"{scoring['composite_score_2y'].max():.6f}"
)


# ============================================================
# COMPOSITE SCORE RANKING
# ============================================================

print()
print_separator("-")

print(
    "CROSS-SECTIONAL COMPOSITE SCORE RANKING"
)

print_separator("-")


print(
    "The composite score is ranked within each "
    "observation date."
)

print(
    "Raw composite-score thresholds are NOT used."
)


scoring["score_percentile_2y"] = (
    scoring
    .groupby("date")[
        "composite_score_2y"
    ]
    .transform(
        percentile_rank
    )
)


# ============================================================
# LABELS
# ============================================================

scoring["label_2y"] = (
    scoring[
        "score_percentile_2y"
    ]
    .map(
        label_from_percentile
    )
)


# ============================================================
# SCORE VALIDATION
# ============================================================

print()
print_separator("-")

print(
    "SCORE AND LABEL VALIDATION"
)

print_separator("-")


invalid_score = (
    scoring["composite_score_2y"].isna()
    |
    (scoring["composite_score_2y"] < 0)
    |
    (scoring["composite_score_2y"] > 1)
).sum()


invalid_percentile = (
    scoring["score_percentile_2y"].isna()
    |
    (scoring["score_percentile_2y"] < 0)
    |
    (scoring["score_percentile_2y"] > 1)
).sum()


invalid_labels = (
    scoring["label_2y"].notna()
    &
    ~scoring["label_2y"].isin(
        [
            "Weak",
            "Neutral",
            "Attractive",
        ]
    )
).sum()


print(
    f"Invalid composite scores: "
    f"{invalid_score}"
)

print(
    f"Invalid score percentiles: "
    f"{invalid_percentile}"
)

print(
    f"Invalid labels: "
    f"{invalid_labels}"
)


if (
    invalid_score
    or invalid_percentile
    or invalid_labels
):
    raise ValueError(
        "Score/label validation failed."
    )


print(
    "Score and label validation: PASS"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print_separator()

print(
    "FINAL STOCK 2Y LABEL DISTRIBUTION"
)

print_separator()


label_counts = (
    scoring["label_2y"]
    .value_counts()
    .reindex(
        [
            "Weak",
            "Neutral",
            "Attractive",
        ],
        fill_value=0,
    )
)


total_labeled = (
    label_counts.sum()
)


for label in [
    "Weak",
    "Neutral",
    "Attractive",
]:

    count = label_counts[label]

    percentage = (
        count
        / total_labeled
        * 100
    )

    print(
        f"{label:12s} "
        f"{count:>10,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# PER-DATE LABEL BALANCE
# ============================================================

print()
print_separator()

print(
    "PER-DATE LABEL BALANCE"
)

print_separator("-")


per_date_labels = (
    scoring
    .groupby("date")[
        "label_2y"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
    .reindex(
        columns=[
            "Weak",
            "Neutral",
            "Attractive",
        ],
        fill_value=0,
    )
)


per_date_cohort = (
    per_date_labels.sum(axis=1)
)


weak_pct = (
    per_date_labels["Weak"]
    / per_date_cohort
    * 100
)


neutral_pct = (
    per_date_labels["Neutral"]
    / per_date_cohort
    * 100
)


attractive_pct = (
    per_date_labels["Attractive"]
    / per_date_cohort
    * 100
)


print(
    f"Weak % range:       "
    f"{weak_pct.min():.2f}% → "
    f"{weak_pct.max():.2f}%"
)

print(
    f"Neutral % range:    "
    f"{neutral_pct.min():.2f}% → "
    f"{neutral_pct.max():.2f}%"
)

print(
    f"Attractive % range: "
    f"{attractive_pct.min():.2f}% → "
    f"{attractive_pct.max():.2f}%"
)


# ============================================================
# SOURCE ROW PRESERVATION
# ============================================================

print()
print_separator("-")

print(
    "SOURCE ROW PRESERVATION"
)

print_separator("-")


# ------------------------------------------------------------
# IMPORTANT FIX:
#
# Do NOT initialize label_2y with np.nan.
#
# np.nan makes the column float64.
# Later assigning strings into that column causes:
#
#     TypeError: Invalid value 'Neutral'
#     for dtype 'float64'
#
# Instead, construct target data separately and merge it
# back into the original source.
# ------------------------------------------------------------


target_columns = [
    "return_percentile_2y",
    "risk_percentile_2y",
    "drawdown_percentile_2y",
    "composite_score_2y",
    "score_percentile_2y",
    "label_2y",
]


# Only retain the key + target columns from scoring.
#
# This is safe because ticker/date was already validated
# as unique.

scored_targets = scoring[
    [
        "ticker",
        "date",
    ]
    + target_columns
].copy()


# Make label explicitly string/object.

scored_targets["label_2y"] = (
    scored_targets["label_2y"]
    .astype("object")
)


# ------------------------------------------------------------
# Validate scored target keys.
# ------------------------------------------------------------

target_duplicate_count = (
    scored_targets
    .duplicated(
        subset=[
            "ticker",
            "date",
        ]
    )
    .sum()
)


print(
    f"Scored target duplicate keys: "
    f"{target_duplicate_count}"
)


if target_duplicate_count != 0:
    raise ValueError(
        "Duplicate keys found in scored target layer."
    )


# ------------------------------------------------------------
# Merge back into ORIGINAL source.
#
# left join guarantees that all original rows remain.
# Rows without a valid 2Y outcome remain unlabeled.
# ------------------------------------------------------------

output = df.merge(
    scored_targets,
    on=[
        "ticker",
        "date",
    ],
    how="left",
    validate="one_to_one",
)


print(
    f"Source rows: "
    f"{len(df):,}"
)

print(
    f"Output rows: "
    f"{len(output):,}"
)


if len(output) != len(df):
    raise ValueError(
        "Output row count changed after merge."
    )


print(
    "Source row preservation: PASS"
)


# ============================================================
# OUTPUT COLUMN ORDER
# ============================================================

preferred_column_order = [
    "ticker",
    "date",
    "adj_close",
    "target_date_2y",
    "future_date_2y",
    "future_adj_close_2y",
    "actual_years_2y",
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",

    "return_percentile_2y",
    "risk_percentile_2y",
    "drawdown_percentile_2y",
    "composite_score_2y",
    "score_percentile_2y",
    "label_2y",
]


remaining_columns = [
    column
    for column in output.columns
    if column not in preferred_column_order
]


output = output[
    preferred_column_order
    +
    remaining_columns
]


# ============================================================
# OUTPUT INTEGRITY
# ============================================================

print()
print_separator()

print(
    "OUTPUT INTEGRITY VALIDATION"
)

print_separator()


output_duplicates = (
    output
    .duplicated(
        subset=[
            "ticker",
            "date",
        ]
    )
    .sum()
)


print(
    f"Output rows: "
    f"{len(output):,}"
)

print(
    f"Output duplicate ticker/date rows: "
    f"{output_duplicates:,}"
)


if output_duplicates != 0:
    raise ValueError(
        "Output contains duplicate ticker/date rows."
    )


# ============================================================
# RAW METRIC PRESERVATION
# ============================================================

print()
print(
    "Checking raw 2Y metric preservation..."
)


raw_metrics = [
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
    "risk_adjusted_return_2y",
]


for column in raw_metrics:

    source_values = (
        df[column]
        .reset_index(drop=True)
    )

    output_values = (
        output[column]
        .reset_index(drop=True)
    )


    identical = (
        source_values.equals(
            output_values
        )
    )


    print(
        f"  {column:32s}"
        f"{'PASS' if identical else 'FAIL'}"
    )


    if not identical:
        raise ValueError(
            f"Raw metric changed: {column}"
        )


# ============================================================
# TARGET COVERAGE
# ============================================================

print()
print_separator()

print(
    "TARGET COVERAGE"
)

print_separator()


scored_rows = (
    output[
        "composite_score_2y"
    ]
    .notna()
    .sum()
)


unscored_rows = (
    len(output)
    - scored_rows
)


print(
    f"Scored rows: "
    f"{scored_rows:,}"
)

print(
    f"Unscored rows: "
    f"{unscored_rows:,}"
)

print(
    f"Expected valid 2Y rows: "
    f"{len(valid):,}"
)


if scored_rows != len(scoring):
    raise ValueError(
        "Scored row count mismatch."
    )


# ============================================================
# SCORE/LABEL CONSISTENCY
# ============================================================

print()
print(
    "Checking score/label consistency..."
)


label_without_score = (
    output["label_2y"].notna()
    &
    output["score_percentile_2y"].isna()
).sum()


score_without_label = (
    output["score_percentile_2y"].notna()
    &
    output["label_2y"].isna()
).sum()


print(
    f"Label without score: "
    f"{label_without_score}"
)

print(
    f"Score without label: "
    f"{score_without_label}"
)


if (
    label_without_score
    or score_without_label
):
    raise ValueError(
        "Score/label consistency failed."
    )


# ============================================================
# LABEL BOUNDARY VALIDATION
# ============================================================

print()
print(
    "Checking label boundaries..."
)


weak_boundary_errors = (
    (
        output["label_2y"]
        == "Weak"
    )
    &
    (
        output["score_percentile_2y"]
        > 0.30
    )
).sum()


neutral_boundary_errors = (
    (
        output["label_2y"]
        == "Neutral"
    )
    &
    (
        (
            output["score_percentile_2y"]
            <= 0.30
        )
        |
        (
            output["score_percentile_2y"]
            > 0.70
        )
    )
).sum()


attractive_boundary_errors = (
    (
        output["label_2y"]
        == "Attractive"
    )
    &
    (
        output["score_percentile_2y"]
        <= 0.70
    )
).sum()


print(
    f"Weak boundary errors: "
    f"{weak_boundary_errors}"
)

print(
    f"Neutral boundary errors: "
    f"{neutral_boundary_errors}"
)

print(
    f"Attractive boundary errors: "
    f"{attractive_boundary_errors}"
)


if (
    weak_boundary_errors
    or neutral_boundary_errors
    or attractive_boundary_errors
):
    raise ValueError(
        "Label boundary validation failed."
    )


# ============================================================
# TICKER/DATE KEY PRESERVATION
# ============================================================

print()
print(
    "Checking ticker/date preservation..."
)


source_keys = set(
    zip(
        df["ticker"],
        df["date"],
    )
)


output_keys = set(
    zip(
        output["ticker"],
        output["date"],
    )
)


missing_keys = (
    source_keys
    - output_keys
)


extra_keys = (
    output_keys
    - source_keys
)


print(
    f"Missing source keys: "
    f"{len(missing_keys)}"
)

print(
    f"Extra output keys: "
    f"{len(extra_keys)}"
)


if missing_keys or extra_keys:
    raise ValueError(
        "Ticker/date keys changed."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


output.to_parquet(
    OUTPUT_FILE,
    index=False,
)


print()
print_separator()

print(
    "OUTPUT SAVED"
)

print_separator()


print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(output):,}"
)

print(
    f"Stocks: {output['ticker'].nunique():,}"
)


# ============================================================
# RELOAD VALIDATION
# ============================================================

print()
print_separator()

print(
    "RELOAD VALIDATION"
)

print_separator()


reloaded = pd.read_parquet(
    OUTPUT_FILE
)


print(
    f"Reloaded rows: "
    f"{len(reloaded):,}"
)

print(
    f"Reloaded stocks: "
    f"{reloaded['ticker'].nunique():,}"
)


if len(reloaded) != len(output):
    raise ValueError(
        "Reloaded row count mismatch."
    )


reload_duplicates = (
    reloaded
    .duplicated(
        subset=[
            "ticker",
            "date",
        ]
    )
    .sum()
)


print(
    f"Reload duplicate ticker/date rows: "
    f"{reload_duplicates}"
)


if reload_duplicates != 0:
    raise ValueError(
        "Reloaded output contains duplicates."
    )


# ============================================================
# RELOAD LABEL DISTRIBUTION
# ============================================================

reload_label_counts = (
    reloaded[
        "label_2y"
    ]
    .dropna()
    .value_counts()
    .reindex(
        [
            "Weak",
            "Neutral",
            "Attractive",
        ],
        fill_value=0,
    )
)


print()
print(
    "Reloaded label distribution:"
)


for label in [
    "Weak",
    "Neutral",
    "Attractive",
]:

    count = (
        reload_label_counts[label]
    )

    percentage = (
        count
        / reload_label_counts.sum()
        * 100
    )


    print(
        f"  {label:12s} "
        f"{count:>10,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print_separator()

print(
    "FINAL VALIDATION"
)

print_separator()


checks = {

    "output_rows_preserved":
        len(output) == len(df),

    "stocks_preserved":
        output["ticker"].nunique()
        == df["ticker"].nunique(),

    "minimum_cohort_respected":
        actual_min_cohort
        >= MIN_COHORT,

    "components_valid":
        scoring[
            component_columns
        ]
        .apply(
            lambda column:
            column.between(
                0,
                1
            ).all()
        )
        .all(),

    "composite_score_valid":
        output[
            "composite_score_2y"
        ]
        .dropna()
        .between(
            0,
            1
        )
        .all(),

    "score_percentile_valid":
        output[
            "score_percentile_2y"
        ]
        .dropna()
        .between(
            0,
            1
        )
        .all(),

    "labels_valid":
        output[
            "label_2y"
        ]
        .dropna()
        .isin(
            [
                "Weak",
                "Neutral",
                "Attractive",
            ]
        )
        .all(),

    "no_duplicate_keys":
        output_duplicates == 0,

    "no_label_without_score":
        label_without_score == 0,

    "no_score_without_label":
        score_without_label == 0,

    "label_boundaries_valid":
        (
            weak_boundary_errors == 0
            and
            neutral_boundary_errors == 0
            and
            attractive_boundary_errors == 0
        ),

    "ticker_date_keys_preserved":
        not missing_keys
        and not extra_keys,

    "reload_row_count":
        len(reloaded)
        == len(output),

    "reload_duplicates_zero":
        reload_duplicates == 0,
}


for name, passed in checks.items():

    print(
        f"{name:42s} "
        f"{'PASS' if passed else 'FAIL'}"
    )


if not all(
    checks.values()
):
    raise ValueError(
        "One or more final validation checks failed."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print_separator()

print(
    "FINAL STATUS"
)

print_separator()


print(
    "PASS — Stock 2Y target layer generated successfully."
)


print()
print(
    "Locked formula:"
)

print(
    "  50% Return Quality"
)

print(
    "  25% Risk Quality"
)

print(
    "  25% Drawdown Quality"
)


print()
print(
    "Labels:"
)

print(
    "  Bottom 30%  -> Weak"
)

print(
    "  Middle 40%  -> Neutral"
)

print(
    "  Top 30%     -> Attractive"
)


print()
print(
    "Risk-adjusted return remains diagnostic-only "
    "and is NOT part of the composite score."
)


print()
print(
    f"Final output: {OUTPUT_FILE}"
)


print_separator()