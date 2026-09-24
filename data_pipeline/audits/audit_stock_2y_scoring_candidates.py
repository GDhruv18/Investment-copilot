"""
Stock 2Y Scoring Candidate Audit
================================

Purpose:
    Compare candidate composite scoring formulas for the Stock 2Y target.

IMPORTANT:
    This script DOES NOT generate the final Stock 2Y target file.

Current diagnostic input schema:
    ticker
    date
    adj_close
    target_date_2y
    future_date_2y
    future_adj_close_2y
    actual_years_2y
    annualized_return_2y
    annualized_volatility_2y
    max_drawdown_2y
    risk_adjusted_return_2y

Method:
    1. Load Stock 2Y diagnostics.
    2. Keep dates with minimum cohort >= 80 stocks.
    3. Compute cross-sectional percentiles within each date:
         - Return percentile: higher return is better
         - Risk-quality percentile: lower volatility is better
         - Drawdown-quality percentile: smaller drawdown is better
    4. Calculate candidate composite scores.
    5. Rank each composite score cross-sectionally within each date.
    6. Assign labels from the ranked composite score:
         Bottom 30%  -> Weak
         Middle 40%  -> Neutral
         Top 30%     -> Attractive
    7. Compare:
         - label distributions
         - per-date label balance
         - candidate correlations
         - top-30% overlap
         - per-stock median score
         - per-stock Attractive rate
         - component contributions

Candidate formulas:

    A = 60% Return + 20% Risk + 20% Drawdown
    B = 50% Return + 25% Risk + 25% Drawdown
    C = 50% Return + 30% Risk + 20% Drawdown
    D = 40% Return + 30% Risk + 30% Drawdown
    E = 70% Return + 15% Risk + 15% Drawdown

Risk-adjusted return is intentionally NOT used as a
separate composite component because the previous diagnostic
showed it is overwhelmingly redundant with annualized return.

This script is an AUDIT ONLY.
It does not create stock_2y_targets.parquet.
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

MIN_COHORT = 80


CANDIDATES = {
    "A": {
        "return": 0.60,
        "risk": 0.20,
        "drawdown": 0.20,
    },
    "B": {
        "return": 0.50,
        "risk": 0.25,
        "drawdown": 0.25,
    },
    "C": {
        "return": 0.50,
        "risk": 0.30,
        "drawdown": 0.20,
    },
    "D": {
        "return": 0.40,
        "risk": 0.30,
        "drawdown": 0.30,
    },
    "E": {
        "return": 0.70,
        "risk": 0.15,
        "drawdown": 0.15,
    },
}


# ============================================================
# HELPERS
# ============================================================

def print_separator(char="=", width=78):
    print(char * width)


def percentile_rank(series):
    """
    Cross-sectional percentile rank.

    Higher value receives higher percentile.

    Ties use average ranking.
    """
    return series.rank(
        method="average",
        pct=True,
    )


def label_from_percentile(value):
    """
    Convert composite-score percentile into label.

    Bottom 30%:
        Weak

    Middle 40%:
        Neutral

    Top 30%:
        Attractive
    """

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
    "STOCK 2Y SCORING CANDIDATE AUDIT — CORRECTED LABELING"
)

print_separator()


# ============================================================
# LOAD INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_parquet(INPUT_FILE)


print(
    f"Input file: {INPUT_FILE}"
)

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "ticker",
    "date",
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
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


print(
    f"Stocks: {df['ticker'].nunique():,}"
)


# ============================================================
# DATE NORMALIZATION
# ============================================================

df["date"] = pd.to_datetime(
    df["date"]
)


print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# BASIC INPUT VALIDATION
# ============================================================

print()
print_separator("-")
print("INPUT VALIDATION")
print_separator("-")


duplicate_count = (
    df.duplicated(
        subset=["ticker", "date"]
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
    "Input validation: PASS"
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


missing_2y = (
    len(df)
    - len(valid)
)


print()
print_separator("-")
print("2Y VALID OBSERVATIONS")
print_separator("-")


print(
    f"Valid 2Y observations: "
    f"{len(valid):,}"
)


print(
    f"Missing/incomplete 2Y observations: "
    f"{missing_2y:,}"
)


# ============================================================
# CROSS-SECTIONAL COHORT SIZES
# ============================================================

cohort_sizes = (
    valid
    .groupby("date")["ticker"]
    .nunique()
)


print()
print_separator("-")
print("COHORT ANALYSIS")
print_separator("-")


print(
    f"Total valid cohort dates: "
    f"{len(cohort_sizes):,}"
)


print(
    f"Minimum cohort required: "
    f"{MIN_COHORT}"
)


print(
    f"Minimum observed cohort: "
    f"{cohort_sizes.min()}"
)


print(
    f"Maximum observed cohort: "
    f"{cohort_sizes.max()}"
)


# ============================================================
# ELIGIBLE DATES
# ============================================================

eligible_dates = cohort_sizes[
    cohort_sizes >= MIN_COHORT
].index


scoring = valid[
    valid["date"].isin(
        eligible_dates
    )
].copy()


print(
    f"Eligible dates: "
    f"{len(eligible_dates):,}"
)


print(
    f"Rows entering scoring: "
    f"{len(scoring):,}"
)


# ============================================================
# ACTUAL COHORT RANGE AFTER FILTER
# ============================================================

scoring_cohorts = (
    scoring
    .groupby("date")["ticker"]
    .nunique()
)


actual_min_cohort = (
    scoring_cohorts.min()
)


actual_max_cohort = (
    scoring_cohorts.max()
)


print(
    f"Actual scoring cohort range: "
    f"{actual_min_cohort} → "
    f"{actual_max_cohort}"
)


if actual_min_cohort < MIN_COHORT:
    raise ValueError(
        "Minimum cohort validation failed."
    )


# ============================================================
# STEP 1
# COMPONENT PERCENTILES
# ============================================================

print()
print_separator("-")

print(
    "STEP 1 — CROSS-SECTIONAL COMPONENT PERCENTILES"
)

print_separator("-")


# ------------------------------------------------------------
# Return percentile
# ------------------------------------------------------------
#
# Higher annualized return = better.
#

scoring["return_percentile"] = (
    scoring
    .groupby("date")[
        "annualized_return_2y"
    ]
    .transform(percentile_rank)
)


# ------------------------------------------------------------
# Risk-quality percentile
# ------------------------------------------------------------
#
# Lower volatility = better.
#
# percentile_rank:
#     lowest volatility -> smallest percentile
#
# We invert the rank so:
#     lowest volatility -> highest risk-quality percentile
#
# Formula:
#
#     1 - percentile + 1/n
#
# This keeps the lowest-volatility observation at 1/n
# distance from the upper boundary and avoids creating
# an artificial value greater than 1.
# ------------------------------------------------------------

def inverse_percentile_rank(series):
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


scoring["risk_percentile"] = (
    scoring
    .groupby("date")[
        "annualized_volatility_2y"
    ]
    .transform(inverse_percentile_rank)
)


# ------------------------------------------------------------
# Drawdown-quality percentile
# ------------------------------------------------------------
#
# max_drawdown is negative.
#
# Example:
#
#     -0.10 = better
#     -0.50 = worse
#
# Therefore normal ascending percentile ranking already
# gives the better / less-negative drawdown a higher score.
#

scoring["drawdown_percentile"] = (
    scoring
    .groupby("date")[
        "max_drawdown_2y"
    ]
    .transform(percentile_rank)
)


# ============================================================
# COMPONENT VALIDATION
# ============================================================

print()
print(
    "Component percentile validation:"
)


component_columns = [
    "return_percentile",
    "risk_percentile",
    "drawdown_percentile",
]


for column in component_columns:

    invalid = (
        scoring[column].isna()
        |
        (scoring[column] < 0)
        |
        (scoring[column] > 1)
    ).sum()


    print(
        f"  {column:25s} "
        f"min={scoring[column].min():.6f} "
        f"max={scoring[column].max():.6f} "
        f"invalid={invalid}"
    )


if (
    scoring[
        component_columns
    ]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        "Component percentile calculation "
        "produced NaN values."
    )


if not (
    scoring[
        component_columns
    ]
    .apply(
        lambda column:
        column.between(0, 1).all()
    )
    .all()
):
    raise ValueError(
        "Component percentiles outside [0,1]."
    )


print(
    "Component percentile validation: PASS"
)


# ============================================================
# STEP 2
# COMPOSITE SCORES
# ============================================================

print()
print_separator("-")

print(
    "STEP 2 — COMPOSITE SCORING"
)

print_separator("-")


for candidate, weights in CANDIDATES.items():

    score_column = (
        f"score_{candidate}"
    )


    scoring[score_column] = (
        weights["return"]
        * scoring["return_percentile"]

        +

        weights["risk"]
        * scoring["risk_percentile"]

        +

        weights["drawdown"]
        * scoring["drawdown_percentile"]
    )


    weight_sum = (
        weights["return"]
        +
        weights["risk"]
        +
        weights["drawdown"]
    )


    if not np.isclose(
        weight_sum,
        1.0,
    ):
        raise ValueError(
            f"Candidate {candidate} "
            f"weights do not sum to 1."
        )


    print(
        f"Candidate {candidate}: "
        f"Return={weights['return']:.0%}, "
        f"Risk={weights['risk']:.0%}, "
        f"Drawdown={weights['drawdown']:.0%}"
    )


    print(
        f"  Score range: "
        f"{scoring[score_column].min():.6f}"
        f" → "
        f"{scoring[score_column].max():.6f}"
    )


# ============================================================
# STEP 3
# CORRECTED COMPOSITE-SCORE RANKING
# ============================================================

print()
print_separator("-")

print(
    "STEP 3 — CROSS-SECTIONAL COMPOSITE SCORE RANKING"
)

print_separator("-")


print(
    "IMPORTANT:"
)

print(
    "Raw composite score thresholds are NOT used."
)

print(
    "Each composite score is ranked within its "
    "observation date."
)

print(
    "The ranked composite percentile is then used "
    "for the 30/40/30 labels."
)


for candidate in CANDIDATES:

    score_column = (
        f"score_{candidate}"
    )

    percentile_column = (
        f"score_percentile_{candidate}"
    )

    label_column = (
        f"label_{candidate}"
    )


    scoring[percentile_column] = (
        scoring
        .groupby("date")[
            score_column
        ]
        .transform(percentile_rank)
    )


    scoring[label_column] = (
        scoring[percentile_column]
        .map(label_from_percentile)
    )


# ============================================================
# COMPOSITE SCORE VALIDATION
# ============================================================

print()
print_separator("-")

print(
    "COMPOSITE SCORE VALIDATION"
)

print_separator("-")


for candidate in CANDIDATES:

    score_column = (
        f"score_{candidate}"
    )

    percentile_column = (
        f"score_percentile_{candidate}"
    )

    label_column = (
        f"label_{candidate}"
    )


    invalid_score = (
        scoring[score_column].isna()
        |
        (scoring[score_column] < 0)
        |
        (scoring[score_column] > 1)
    ).sum()


    invalid_percentile = (
        scoring[percentile_column].isna()
        |
        (scoring[percentile_column] < 0)
        |
        (scoring[percentile_column] > 1)
    ).sum()


    valid_labels = {
        "Weak",
        "Neutral",
        "Attractive",
    }


    invalid_labels = (
        scoring[label_column].notna()
        &
        ~scoring[label_column].isin(
            valid_labels
        )
    ).sum()


    print(
        f"Candidate {candidate}: "
        f"invalid_score={invalid_score}, "
        f"invalid_percentile={invalid_percentile}, "
        f"invalid_labels={invalid_labels}"
    )


    if (
        invalid_score
        or invalid_percentile
        or invalid_labels
    ):
        raise ValueError(
            f"Candidate {candidate} "
            f"failed validation."
        )


print(
    "Composite score validation: PASS"
)


# ============================================================
# CORRECTED LABEL DISTRIBUTIONS
# ============================================================

print()
print_separator()

print(
    "CORRECTED AGGREGATE LABEL DISTRIBUTIONS"
)

print_separator()


label_summary = []


for candidate in CANDIDATES:

    label_column = (
        f"label_{candidate}"
    )


    counts = (
        scoring[label_column]
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


    total = counts.sum()


    weak_pct = (
        counts["Weak"]
        / total
        * 100
    )


    neutral_pct = (
        counts["Neutral"]
        / total
        * 100
    )


    attractive_pct = (
        counts["Attractive"]
        / total
        * 100
    )


    label_summary.append(
        {
            "candidate": candidate,
            "weak": counts["Weak"],
            "neutral": counts["Neutral"],
            "attractive": counts["Attractive"],
            "weak_pct": weak_pct,
            "neutral_pct": neutral_pct,
            "attractive_pct": attractive_pct,
        }
    )


    print()
    print(
        f"Candidate {candidate}"
    )


    print(
        f"  Weak:       "
        f"{counts['Weak']:>8,} "
        f"({weak_pct:6.2f}%)"
    )


    print(
        f"  Neutral:    "
        f"{counts['Neutral']:>8,} "
        f"({neutral_pct:6.2f}%)"
    )


    print(
        f"  Attractive: "
        f"{counts['Attractive']:>8,} "
        f"({attractive_pct:6.2f}%)"
    )


# ============================================================
# PER-DATE LABEL BALANCE
# ============================================================

print()
print_separator()

print(
    "PER-DATE LABEL BALANCE"
)

print_separator()


for candidate in CANDIDATES:

    label_column = (
        f"label_{candidate}"
    )


    per_date = (
        scoring
        .groupby("date")[label_column]
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


    cohort = (
        per_date.sum(axis=1)
    )


    weak_pct = (
        per_date["Weak"]
        / cohort
        * 100
    )


    neutral_pct = (
        per_date["Neutral"]
        / cohort
        * 100
    )


    attractive_pct = (
        per_date["Attractive"]
        / cohort
        * 100
    )


    print()
    print(
        f"Candidate {candidate}"
    )


    print(
        f"  Weak % range:       "
        f"{weak_pct.min():.2f}% → "
        f"{weak_pct.max():.2f}%"
    )


    print(
        f"  Neutral % range:    "
        f"{neutral_pct.min():.2f}% → "
        f"{neutral_pct.max():.2f}%"
    )


    print(
        f"  Attractive % range: "
        f"{attractive_pct.min():.2f}% → "
        f"{attractive_pct.max():.2f}%"
    )


# ============================================================
# COMPOSITE SCORE DISTRIBUTIONS
# ============================================================

print()
print_separator()

print(
    "COMPOSITE SCORE DISTRIBUTIONS"
)

print_separator()


for candidate in CANDIDATES:

    score_column = (
        f"score_{candidate}"
    )


    desc = scoring[
        score_column
    ].describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )


    print()
    print(
        f"Candidate {candidate}"
    )


    print(
        f"  Mean:   {desc['mean']:.6f}"
    )

    print(
        f"  Std:    {desc['std']:.6f}"
    )

    print(
        f"  Min:    {desc['min']:.6f}"
    )

    print(
        f"  1%:     {desc['1%']:.6f}"
    )

    print(
        f"  5%:     {desc['5%']:.6f}"
    )

    print(
        f"  10%:    {desc['10%']:.6f}"
    )

    print(
        f"  25%:    {desc['25%']:.6f}"
    )

    print(
        f"  Median: {desc['50%']:.6f}"
    )

    print(
        f"  75%:    {desc['75%']:.6f}"
    )

    print(
        f"  90%:    {desc['90%']:.6f}"
    )

    print(
        f"  95%:    {desc['95%']:.6f}"
    )

    print(
        f"  99%:    {desc['99%']:.6f}"
    )

    print(
        f"  Max:    {desc['max']:.6f}"
    )


# ============================================================
# CANDIDATE SCORE CORRELATIONS
# ============================================================

print()
print_separator()

print(
    "CANDIDATE SCORE CORRELATIONS"
)

print_separator()


candidate_score_columns = [
    f"score_{candidate}"
    for candidate in CANDIDATES
]


pearson = (
    scoring[
        candidate_score_columns
    ]
    .corr(
        method="pearson"
    )
)


spearman = (
    scoring[
        candidate_score_columns
    ]
    .corr(
        method="spearman"
    )
)


print()
print(
    "PEARSON"
)

print(
    pearson.round(6).to_string()
)


print()
print(
    "SPEARMAN"
)

print(
    spearman.round(6).to_string()
)


# ============================================================
# TOP 30% OVERLAP
# ============================================================

print()
print_separator()

print(
    "TOP-30% ATTRACTIVE OVERLAP"
)

print_separator()


for candidate in CANDIDATES:

    percentile_column = (
        f"score_percentile_{candidate}"
    )


    scoring[
        f"top30_{candidate}"
    ] = (
        scoring[percentile_column]
        > 0.70
    )


candidate_names = list(
    CANDIDATES.keys()
)


for i in range(
    len(candidate_names)
):

    for j in range(
        i + 1,
        len(candidate_names)
    ):

        a = candidate_names[i]
        b = candidate_names[j]


        set_a = set(
            scoring.loc[
                scoring[f"top30_{a}"],
                ["date", "ticker"],
            ]
            .apply(
                tuple,
                axis=1,
            )
        )


        set_b = set(
            scoring.loc[
                scoring[f"top30_{b}"],
                ["date", "ticker"],
            ]
            .apply(
                tuple,
                axis=1,
            )
        )


        intersection = (
            len(
                set_a & set_b
            )
        )


        union = (
            len(
                set_a | set_b
            )
        )


        jaccard = (
            intersection / union
            if union
            else np.nan
        )


        print(
            f"{a} vs {b}: "
            f"intersection={intersection:,}, "
            f"union={union:,}, "
            f"Jaccard={jaccard:.6f}"
        )


# ============================================================
# ATTRACTIVE LABEL OVERLAP
# ============================================================

print()
print_separator()

print(
    "ATTRACTIVE LABEL OVERLAP"
)

print_separator()


for i in range(
    len(candidate_names)
):

    for j in range(
        i + 1,
        len(candidate_names)
    ):

        a = candidate_names[i]
        b = candidate_names[j]


        both_attractive = (
            (
                scoring[
                    f"label_{a}"
                ]
                == "Attractive"
            )
            &
            (
                scoring[
                    f"label_{b}"
                ]
                == "Attractive"
            )
        ).sum()


        print(
            f"{a} vs {b}: "
            f"both Attractive = "
            f"{both_attractive:,}"
        )


# ============================================================
# PER-STOCK MEDIAN COMPOSITE PERCENTILE
# ============================================================

print()
print_separator()

print(
    "PER-STOCK MEDIAN COMPOSITE SCORE PERCENTILES"
)

print_separator()


for candidate in CANDIDATES:

    percentile_column = (
        f"score_percentile_{candidate}"
    )


    stock_median = (
        scoring
        .groupby("ticker")[
            percentile_column
        ]
        .median()
        .sort_values(
            ascending=False
        )
    )


    print()
    print(
        f"Candidate {candidate}"
    )


    print(
        "  Top 15 stocks:"
    )


    for ticker, value in (
        stock_median
        .head(15)
        .items()
    ):

        print(
            f"    {ticker:15s} "
            f"{value:.6f}"
        )


    print(
        "  Bottom 15 stocks:"
    )


    for ticker, value in (
        stock_median
        .tail(15)
        .items()
    ):

        print(
            f"    {ticker:15s} "
            f"{value:.6f}"
        )


# ============================================================
# PER-STOCK ATTRACTIVE RATE
# ============================================================

print()
print_separator()

print(
    "PER-STOCK ATTRACTIVE LABEL RATE"
)

print_separator()


for candidate in CANDIDATES:

    label_column = (
        f"label_{candidate}"
    )


    attractive_rate = (
        scoring[label_column]
        .eq("Attractive")
        .groupby(
            scoring["ticker"]
        )
        .mean()
        .sort_values(
            ascending=False
        )
    )


    print()
    print(
        f"Candidate {candidate}"
    )


    print(
        "  Highest Attractive rates:"
    )


    for ticker, value in (
        attractive_rate
        .head(10)
        .items()
    ):

        print(
            f"    {ticker:15s} "
            f"{value * 100:6.2f}%"
        )


    print(
        "  Lowest Attractive rates:"
    )


    for ticker, value in (
        attractive_rate
        .tail(10)
        .items()
    ):

        print(
            f"    {ticker:15s} "
            f"{value * 100:6.2f}%"
        )


# ============================================================
# COMPONENT CONTRIBUTIONS
# ============================================================

print()
print_separator()

print(
    "WEIGHTED COMPONENT CONTRIBUTIONS"
)

print_separator()


component_means = {
    "return":
        scoring[
            "return_percentile"
        ].mean(),

    "risk":
        scoring[
            "risk_percentile"
        ].mean(),

    "drawdown":
        scoring[
            "drawdown_percentile"
        ].mean(),
}


print(
    f"Mean return percentile:   "
    f"{component_means['return']:.6f}"
)


print(
    f"Mean risk percentile:     "
    f"{component_means['risk']:.6f}"
)


print(
    f"Mean drawdown percentile: "
    f"{component_means['drawdown']:.6f}"
)


print()


for candidate, weights in (
    CANDIDATES.items()
):

    return_contribution = (
        weights["return"]
        * component_means["return"]
    )


    risk_contribution = (
        weights["risk"]
        * component_means["risk"]
    )


    drawdown_contribution = (
        weights["drawdown"]
        * component_means["drawdown"]
    )


    total = (
        return_contribution
        +
        risk_contribution
        +
        drawdown_contribution
    )


    print(
        f"Candidate {candidate}:"
    )


    print(
        f"  Return contribution:   "
        f"{return_contribution:.6f}"
    )


    print(
        f"  Risk contribution:     "
        f"{risk_contribution:.6f}"
    )


    print(
        f"  Drawdown contribution: "
        f"{drawdown_contribution:.6f}"
    )


    print(
        f"  Total:                 "
        f"{total:.6f}"
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


checks = {}


checks[
    "scoring_rows_present"
] = (
    len(scoring) > 0
)


checks[
    "minimum_cohort_respected"
] = (
    actual_min_cohort
    >= MIN_COHORT
)


checks[
    "component_percentiles_valid"
] = (
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
    .all()
)


for candidate in CANDIDATES:

    score_column = (
        f"score_{candidate}"
    )

    percentile_column = (
        f"score_percentile_{candidate}"
    )

    label_column = (
        f"label_{candidate}"
    )


    checks[
        f"{candidate}_score_valid"
    ] = (
        scoring[score_column]
        .between(
            0,
            1
        )
        .all()
    )


    checks[
        f"{candidate}_score_percentile_valid"
    ] = (
        scoring[percentile_column]
        .between(
            0,
            1
        )
        .all()
    )


    checks[
        f"{candidate}_labels_valid"
    ] = (
        scoring[label_column]
        .isin(
            [
                "Weak",
                "Neutral",
                "Attractive",
            ]
        )
        .all()
    )


for name, passed in (
    checks.items()
):

    print(
        f"{name:45s} "
        f"{'PASS' if passed else 'FAIL'}"
    )


if not all(
    checks.values()
):

    raise ValueError(
        "One or more final validation "
        "checks failed."
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
    "PASS — corrected Stock 2Y "
    "scoring candidate audit completed."
)


print()


print(
    "No Stock 2Y target file was generated."
)


print(
    "No candidate was selected."
)


print(
    "Composite scores were ranked "
    "cross-sectionally by date before "
    "the 30/40/30 labels were assigned."
)


print_separator()