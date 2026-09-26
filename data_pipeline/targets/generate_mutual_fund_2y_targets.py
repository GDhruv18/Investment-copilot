from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/targets/mutual_fund_2y_diagnostics.parquet"
)

OUTPUT_FILE = Path(
    "data/targets/mutual_fund_2y_targets.parquet"
)

EXPECTED_SCHEMES = 100

MIN_COHORT = 30


# ============================================================
# LOCKED 2Y METHODOLOGY
# ============================================================

RETURN_WEIGHT = 0.50
RISK_WEIGHT = 0.25
DRAWDOWN_WEIGHT = 0.25


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD SOURCE
# ============================================================

print_section("MUTUAL FUND 2Y TARGET GENERATION")

print(
    f"Input: {INPUT_FILE}"
)

print(
    f"Output: {OUTPUT_FILE}"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_parquet(
    INPUT_FILE
)


print(
    f"Loaded rows: {len(df):,}"
)

print(
    f"Schemes: "
    f"{df['scheme_code'].nunique():,}"
)

print(
    f"Date range: "
    f"{df['nav_date'].min()} → "
    f"{df['nav_date'].max()}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print_section("BASIC VALIDATION")


required_columns = [
    "scheme_code",
    "nav_date",
    "annualized_return_2y",
    "annualized_volatility_2y",
    "max_drawdown_2y",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


if (
    df["scheme_code"].nunique()
    != EXPECTED_SCHEMES
):

    raise RuntimeError(
        "Unexpected mutual-fund universe size."
    )


duplicate_count = df.duplicated(
    ["scheme_code", "nav_date"]
).sum()


if duplicate_count != 0:

    raise RuntimeError(
        "Duplicate scheme/date rows detected."
    )


print(
    "Universe: PASS"
)

print(
    "Duplicate scheme/date rows: PASS"
)


# ============================================================
# VALID 2Y OBSERVATIONS
# ============================================================

valid_mask = (
    df["annualized_return_2y"].notna()
    & df["annualized_volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
)


valid = df.loc[
    valid_mask
].copy()


print()
print(
    f"Valid 2Y rows: {len(valid):,}"
)

print(
    f"Missing 2Y rows: "
    f"{len(df) - len(valid):,}"
)

print(
    f"Valid coverage: "
    f"{len(valid) / len(df) * 100:.2f}%"
)


# ============================================================
# COHORT ANALYSIS
# ============================================================

print_section("COHORT FILTER")


cohort_sizes = (
    valid
    .groupby("nav_date")["scheme_code"]
    .nunique()
)


eligible_dates = cohort_sizes[
    cohort_sizes >= MIN_COHORT
].index


target = valid[
    valid["nav_date"].isin(
        eligible_dates
    )
].copy()


print(
    f"Minimum cohort required: "
    f"{MIN_COHORT}"
)

print(
    f"Eligible cohort dates: "
    f"{len(eligible_dates):,}"
)

print(
    f"Eligible rows: "
    f"{len(target):,}"
)

print(
    f"Retention from valid 2Y rows: "
    f"{len(target) / len(valid) * 100:.2f}%"
)


eligible_cohort_sizes = (
    target
    .groupby("nav_date")["scheme_code"]
    .nunique()
)


print(
    f"Actual cohort range: "
    f"{eligible_cohort_sizes.min():,} → "
    f"{eligible_cohort_sizes.max():,}"
)


if (
    eligible_cohort_sizes.min()
    < MIN_COHORT
):

    raise RuntimeError(
        "Minimum cohort filter failed."
    )


# ============================================================
# CROSS-SECTIONAL COMPONENT RANKING
# ============================================================

print_section(
    "CROSS-SECTIONAL COMPONENT RANKING"
)


# ------------------------------------------------------------
# RETURN QUALITY
# ------------------------------------------------------------
# Higher annualized return = better.
# ------------------------------------------------------------

target["return_pct"] = (
    target
    .groupby("nav_date")[
        "annualized_return_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# ------------------------------------------------------------
# RISK QUALITY
# ------------------------------------------------------------
# Lower volatility = better.
# ------------------------------------------------------------

volatility_rank = (
    target
    .groupby("nav_date")[
        "annualized_volatility_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


target["risk_pct"] = (
    1.0 - volatility_rank
)


# ------------------------------------------------------------
# DRAWDOWN QUALITY
# ------------------------------------------------------------
# Higher / less-negative drawdown = better.
# ------------------------------------------------------------

target["drawdown_pct"] = (
    target
    .groupby("nav_date")[
        "max_drawdown_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# ============================================================
# COMPOSITE SCORE
# ============================================================

print_section("COMPOSITE SCORE")


target["score_2y"] = (
    RETURN_WEIGHT
    * target["return_pct"]
    +
    RISK_WEIGHT
    * target["risk_pct"]
    +
    DRAWDOWN_WEIGHT
    * target["drawdown_pct"]
)


print(
    "Weights:"
)

print(
    f"  Return:    {RETURN_WEIGHT:.0%}"
)

print(
    f"  Risk:      {RISK_WEIGHT:.0%}"
)

print(
    f"  Drawdown:  {DRAWDOWN_WEIGHT:.0%}"
)


# ============================================================
# COMPOSITE SCORE RANK
# ============================================================

print_section(
    "CROSS-SECTIONAL COMPOSITE RANK"
)


target["rank_2y"] = (
    target
    .groupby("nav_date")[
        "score_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# ============================================================
# LABELS
# ============================================================

print_section("2Y LABEL GENERATION")


target["label_2y"] = np.select(
    [
        target["rank_2y"] <= 0.30,
        target["rank_2y"] <= 0.70,
    ],
    [
        0,
        1,
    ],
    default=2
)


target["label_2y_name"] = (
    target["label_2y"]
    .map(
        {
            0: "Weak",
            1: "Neutral",
            2: "Attractive",
        }
    )
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print_section("LABEL DISTRIBUTION")


label_counts = (
    target["label_2y"]
    .value_counts()
    .sort_index()
)


for label, name in [
    (0, "Weak"),
    (1, "Neutral"),
    (2, "Attractive"),
]:

    count = label_counts.get(
        label,
        0
    )

    print(
        f"{name:<10}: "
        f"{count:,} "
        f"({count / len(target) * 100:.2f}%)"
    )


# ============================================================
# SCORE / RANK SUMMARY
# ============================================================

print_section("SCORE SUMMARY")


print(
    f"Score range: "
    f"{target['score_2y'].min():.6f} → "
    f"{target['score_2y'].max():.6f}"
)

print(
    f"Rank range: "
    f"{target['rank_2y'].min():.6f} → "
    f"{target['rank_2y'].max():.6f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print_section("FINAL VALIDATION")


checks = []


# ------------------------------------------------------------
# Universe
# ------------------------------------------------------------

checks.append(
    (
        "100 schemes in source",
        df["scheme_code"].nunique()
        == EXPECTED_SCHEMES
    )
)


checks.append(
    (
        "100 schemes in target",
        target["scheme_code"].nunique()
        <= EXPECTED_SCHEMES
    )
)


# ------------------------------------------------------------
# Duplicates
# ------------------------------------------------------------

checks.append(
    (
        "no duplicate scheme/date rows",
        target.duplicated(
            ["scheme_code", "nav_date"]
        ).sum() == 0
    )
)


# ------------------------------------------------------------
# Cohort
# ------------------------------------------------------------

checks.append(
    (
        "minimum cohort satisfied",
        target
        .groupby("nav_date")[
            "scheme_code"
        ]
        .nunique()
        .min()
        >= MIN_COHORT
    )
)


# ------------------------------------------------------------
# Score
# ------------------------------------------------------------

checks.append(
    (
        "score finite",
        np.isfinite(
            target["score_2y"]
        ).all()
    )
)


# ------------------------------------------------------------
# Rank
# ------------------------------------------------------------

checks.append(
    (
        "rank finite",
        np.isfinite(
            target["rank_2y"]
        ).all()
    )
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

checks.append(
    (
        "labels valid",
        target["label_2y"].isin(
            [0, 1, 2]
        ).all()
    )
)


# ------------------------------------------------------------
# Label names
# ------------------------------------------------------------

checks.append(
    (
        "label names valid",
        target["label_2y_name"].isin(
            [
                "Weak",
                "Neutral",
                "Attractive",
            ]
        ).all()
    )
)


# ------------------------------------------------------------
# Component percentiles
# ------------------------------------------------------------

checks.append(
    (
        "return percentile finite",
        np.isfinite(
            target["return_pct"]
        ).all()
    )
)


checks.append(
    (
        "risk percentile finite",
        np.isfinite(
            target["risk_pct"]
        ).all()
    )
)


checks.append(
    (
        "drawdown percentile finite",
        np.isfinite(
            target["drawdown_pct"]
        ).all()
    )
)


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

for name, passed in checks:

    print(
        f"{'PASS' if passed else 'FAIL'} - {name}"
    )


if not all(
    passed
    for _, passed in checks
):

    raise RuntimeError(
        "One or more final target validation checks failed."
    )


# ============================================================
# SAVE
# ============================================================

print_section("SAVING TARGET DATASET")


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


target.to_parquet(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(target):,}"
)

print(
    f"Schemes: "
    f"{target['scheme_code'].nunique():,}"
)


# ============================================================
# RELOAD VALIDATION
# ============================================================

print_section("RELOAD VALIDATION")


reloaded = pd.read_parquet(
    OUTPUT_FILE
)


reload_checks = [
    (
        "row count preserved",
        len(reloaded) == len(target)
    ),
    (
        "scheme count preserved",
        reloaded["scheme_code"].nunique()
        == target["scheme_code"].nunique()
    ),
    (
        "duplicate scheme/date rows",
        reloaded.duplicated(
            ["scheme_code", "nav_date"]
        ).sum() == 0
    ),
    (
        "labels valid",
        reloaded["label_2y"].isin(
            [0, 1, 2]
        ).all()
    ),
    (
        "score finite",
        np.isfinite(
            reloaded["score_2y"]
        ).all()
    ),
    (
        "rank finite",
        np.isfinite(
            reloaded["rank_2y"]
        ).all()
    ),
]


for name, passed in reload_checks:

    print(
        f"{'PASS' if passed else 'FAIL'} - {name}"
    )


if not all(
    passed
    for _, passed in reload_checks
):

    raise RuntimeError(
        "Reload validation failed."
    )


# ============================================================
# FINAL STATUS
# ============================================================

print_section("FINAL STATUS")

print(
    "MUTUAL FUND 2Y TARGET GENERATION: PASS"
)

print(
    f"Rows: {len(target):,}"
)

print(
    f"Schemes: "
    f"{target['scheme_code'].nunique():,}"
)

print(
    f"Eligible dates: "
    f"{len(eligible_dates):,}"
)

print(
    f"Minimum cohort: {MIN_COHORT}"
)

print(
    "Composite: "
    "50% Return + 25% Risk + 25% Drawdown"
)

print(
    "Labels: "
    "30% Weak / 40% Neutral / 30% Attractive"
)

print(
    f"Output: {OUTPUT_FILE}"
)