from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/targets/mutual_fund_2y_diagnostics.parquet"
)

EXPECTED_SCHEMES = 100

MIN_COHORT_OPTIONS = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
]


# ============================================================
# CANDIDATE WEIGHTS
# ============================================================

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
        "return": 0.50,
        "risk": 0.20,
        "drawdown": 0.30,
    },
    "E": {
        "return": 0.333,
        "risk": 0.333,
        "drawdown": 0.334,
    },
    "F": {
        "return": 0.70,
        "risk": 0.15,
        "drawdown": 0.15,
    },
}


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_corr(series_a, series_b, method="pearson"):

    temp = pd.concat(
        [series_a, series_b],
        axis=1
    ).dropna()

    if len(temp) < 2:
        return np.nan

    return temp.iloc[:, 0].corr(
        temp.iloc[:, 1],
        method=method
    )


def percentage(value):

    return f"{value * 100:.2f}%"


# ============================================================
# LOAD DATA
# ============================================================

print_section("MUTUAL FUND 2Y CANDIDATE AUDIT")

print(f"Input: {INPUT_FILE}")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_parquet(INPUT_FILE)


print(
    f"Loaded rows: {len(df):,}"
)

print(
    f"Schemes: {df['scheme_code'].nunique():,}"
)

print(
    f"Date range: "
    f"{df['nav_date'].min()} → {df['nav_date'].max()}"
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


duplicate_count = df.duplicated(
    ["scheme_code", "nav_date"]
).sum()


print(
    f"Duplicate scheme/date rows: "
    f"{duplicate_count:,}"
)


if duplicate_count != 0:

    raise RuntimeError(
        "Duplicate scheme/date rows detected."
    )


# ============================================================
# VALID 2Y OUTCOMES
# ============================================================

valid_mask = (
    df["annualized_return_2y"].notna()
    & df["annualized_volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
)


valid = df.loc[
    valid_mask
].copy()


print(
    f"Valid 2Y rows: {len(valid):,}"
)

print(
    f"Missing 2Y rows: "
    f"{len(df) - len(valid):,}"
)

print(
    f"2Y coverage: "
    f"{len(valid) / len(df) * 100:.2f}%"
)


# ============================================================
# COHORT ANALYSIS
# ============================================================

print_section("COHORT ANALYSIS")


cohort_sizes = (
    valid
    .groupby("nav_date")["scheme_code"]
    .nunique()
)


print(
    f"Cohort dates: {len(cohort_sizes):,}"
)

print(
    f"Minimum cohort: {cohort_sizes.min():,}"
)

print(
    f"Median cohort: {cohort_sizes.median():.0f}"
)

print(
    f"Mean cohort: {cohort_sizes.mean():.2f}"
)

print(
    f"Maximum cohort: {cohort_sizes.max():,}"
)


print()
print(
    "Minimum cohort retention:"
)

for minimum_cohort in MIN_COHORT_OPTIONS:

    eligible_dates = (
        cohort_sizes[
            cohort_sizes >= minimum_cohort
        ]
    )

    eligible_rows = valid[
        valid["nav_date"].isin(
            eligible_dates.index
        )
    ]

    print(
        f"  >= {minimum_cohort:>2}: "
        f"{len(eligible_dates):,} dates, "
        f"{len(eligible_rows):,} rows"
    )


# ============================================================
# VOLATILITY ANALYSIS
# ============================================================

print_section("LOW-VOLATILITY ANALYSIS")


volatility = valid[
    "annualized_volatility_2y"
]


for threshold in [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]:

    count = (
        volatility < threshold
    ).sum()

    print(
        f"Volatility < {threshold * 100:.1f}%: "
        f"{count:,} rows "
        f"({count / len(valid) * 100:.2f}%)"
    )


scheme_volatility = (
    valid
    .groupby("scheme_code")[
        "annualized_volatility_2y"
    ]
    .median()
    .sort_values()
)


print()
print("Lowest median-volatility schemes:")

print(
    scheme_volatility.head(10).to_string()
)


print()
print("Highest median-volatility schemes:")

print(
    scheme_volatility.tail(10).sort_values(
        ascending=False
    ).to_string()
)


# ============================================================
# DRAWDOWN ANALYSIS
# ============================================================

print_section("DRAWDOWN ANALYSIS")


drawdown = valid[
    "max_drawdown_2y"
]


for threshold in [
    -0.01,
    -0.05,
    -0.10,
    -0.25,
    -0.50,
]:

    count = (
        drawdown < threshold
    ).sum()

    print(
        f"Drawdown < {threshold * 100:.0f}%: "
        f"{count:,} rows "
        f"({count / len(valid) * 100:.2f}%)"
    )


scheme_drawdown = (
    valid
    .groupby("scheme_code")[
        "max_drawdown_2y"
    ]
    .median()
    .sort_values(
        ascending=False
    )
)


print()
print("Best median drawdown schemes:")

print(
    scheme_drawdown.head(10).to_string()
)


print()
print("Worst median drawdown schemes:")

print(
    scheme_drawdown.tail(10).sort_values().to_string()
)


# ============================================================
# CROSS-SECTIONAL PERCENTILE RANKS
# ============================================================

print_section(
    "CROSS-SECTIONAL COMPONENT RANKS"
)


# Higher annualized return = better
valid["return_pct"] = (
    valid
    .groupby("nav_date")[
        "annualized_return_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# Lower volatility = better
volatility_rank = (
    valid
    .groupby("nav_date")[
        "annualized_volatility_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)

valid["risk_pct"] = (
    1.0 - volatility_rank
)


# Less-negative / higher drawdown = better
valid["drawdown_pct"] = (
    valid
    .groupby("nav_date")[
        "max_drawdown_2y"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


print(
    "Return percentile range: "
    f"{valid['return_pct'].min():.6f} → "
    f"{valid['return_pct'].max():.6f}"
)

print(
    "Risk percentile range: "
    f"{valid['risk_pct'].min():.6f} → "
    f"{valid['risk_pct'].max():.6f}"
)

print(
    "Drawdown percentile range: "
    f"{valid['drawdown_pct'].min():.6f} → "
    f"{valid['drawdown_pct'].max():.6f}"
)


# ============================================================
# COMPONENT CORRELATIONS
# ============================================================

print_section(
    "COMPONENT RANK CORRELATIONS"
)


components = [
    "return_pct",
    "risk_pct",
    "drawdown_pct",
]


print(
    "Return vs Risk: "
    f"{safe_corr(valid['return_pct'], valid['risk_pct']):.4f}"
)

print(
    "Return vs Drawdown: "
    f"{safe_corr(valid['return_pct'], valid['drawdown_pct']):.4f}"
)

print(
    "Risk vs Drawdown: "
    f"{safe_corr(valid['risk_pct'], valid['drawdown_pct']):.4f}"
)


print()
print("Raw component correlations:")

print(
    "Return vs Volatility: "
    f"{safe_corr(valid['annualized_return_2y'], valid['annualized_volatility_2y']):.4f}"
)

print(
    "Return vs Drawdown: "
    f"{safe_corr(valid['annualized_return_2y'], valid['max_drawdown_2y']):.4f}"
)

print(
    "Volatility vs Drawdown: "
    f"{safe_corr(valid['annualized_volatility_2y'], valid['max_drawdown_2y']):.4f}"
)


# ============================================================
# CANDIDATE SCORE CALCULATION
# ============================================================

print_section("CANDIDATE SCORES")


for name, weights in CANDIDATES.items():

    valid[f"score_{name}"] = (
        weights["return"]
        * valid["return_pct"]
        +
        weights["risk"]
        * valid["risk_pct"]
        +
        weights["drawdown"]
        * valid["drawdown_pct"]
    )


for name in CANDIDATES:

    print(
        f"{name}: "
        f"{valid[f'score_{name}'].min():.6f} → "
        f"{valid[f'score_{name}'].max():.6f}"
    )


# ============================================================
# SCORE CORRELATIONS
# ============================================================

print_section("CANDIDATE SCORE CORRELATIONS")


candidate_names = list(CANDIDATES.keys())


for i, name_a in enumerate(candidate_names):

    for name_b in candidate_names[i + 1:]:

        correlation = safe_corr(
            valid[f"score_{name_a}"],
            valid[f"score_{name_b}"]
        )

        print(
            f"{name_a} vs {name_b}: "
            f"{correlation:.4f}"
        )


# ============================================================
# SCORE RANKS AND LABELS
# ============================================================

print_section(
    "CROSS-SECTIONAL SCORE RANKS AND LABELS"
)


for name in candidate_names:

    valid[f"rank_{name}"] = (
        valid
        .groupby("nav_date")[
            f"score_{name}"
        ]
        .rank(
            pct=True,
            method="average"
        )
    )

    valid[f"label_{name}"] = np.select(
        [
            valid[f"rank_{name}"] <= 0.30,
            valid[f"rank_{name}"] <= 0.70,
        ],
        [
            0,
            1,
        ],
        default=2
    )


# ============================================================
# LABEL DISTRIBUTIONS
# ============================================================

print_section("LABEL DISTRIBUTIONS")


for name in candidate_names:

    counts = (
        valid[f"label_{name}"]
        .value_counts()
        .sort_index()
    )

    print()
    print(f"Candidate {name}:")

    for label, label_name in [
        (0, "Weak"),
        (1, "Neutral"),
        (2, "Attractive"),
    ]:

        count = counts.get(
            label,
            0
        )

        print(
            f"  {label_name:<10}: "
            f"{count:,} "
            f"({count / len(valid) * 100:.2f}%)"
        )


# ============================================================
# TOP-30% OVERLAP
# ============================================================

print_section(
    "TOP-30% CANDIDATE OVERLAP"
)


top_sets = {}


for name in candidate_names:

    top_sets[name] = set(
        valid.index[
            valid[f"rank_{name}"] > 0.70
        ]
    )


for i, name_a in enumerate(candidate_names):

    for name_b in candidate_names[i + 1:]:

        intersection = (
            top_sets[name_a]
            & top_sets[name_b]
        )

        union = (
            top_sets[name_a]
            | top_sets[name_b]
        )

        if len(union) == 0:

            jaccard = np.nan

        else:

            jaccard = (
                len(intersection)
                / len(union)
            )

        print(
            f"{name_a} vs {name_b}: "
            f"{jaccard:.6f}"
        )


# ============================================================
# PER-SCHEME ATTRACTIVE RATES
# ============================================================

print_section(
    "PER-SCHEME ATTRACTIVE RATES"
)


attractive_rates = pd.DataFrame(
    index=sorted(
        valid["scheme_code"].unique()
    )
)


for name in candidate_names:

    attractive_rates[name] = (
        valid
        .groupby("scheme_code")[
            f"label_{name}"
        ]
        .apply(
            lambda x: (
                x == 2
            ).mean()
        )
    )


for name in candidate_names:

    print()
    print(
        f"Candidate {name} "
        f"highest attractive rates:"
    )

    print(
        attractive_rates[name]
        .sort_values(
            ascending=False
        )
        .head(10)
        .to_string()
    )


# ============================================================
# PER-SCHEME MEDIAN SCORES
# ============================================================

print_section(
    "PER-SCHEME MEDIAN SCORES"
)


median_scores = pd.DataFrame(
    index=sorted(
        valid["scheme_code"].unique()
    )
)


for name in candidate_names:

    median_scores[name] = (
        valid
        .groupby("scheme_code")[
            f"score_{name}"
        ]
        .median()
    )


for name in candidate_names:

    print()
    print(
        f"Candidate {name} "
        f"highest median scores:"
    )

    print(
        median_scores[name]
        .sort_values(
            ascending=False
        )
        .head(10)
        .to_string()
    )


# ============================================================
# FINAL AUDIT
# ============================================================

print_section("FINAL AUDIT")


checks = []


# ------------------------------------------------------------
# SOURCE UNIVERSE CHECK
# ------------------------------------------------------------

checks.append(
    (
        "100 schemes",
        df["scheme_code"].nunique()
        == EXPECTED_SCHEMES
    )
)


# ------------------------------------------------------------
# VALID DATA CHECK
# ------------------------------------------------------------

checks.append(
    (
        "no duplicate scheme/date rows",
        valid.duplicated(
            ["scheme_code", "nav_date"]
        ).sum() == 0
    )
)


# ------------------------------------------------------------
# CANDIDATE CHECKS
# ------------------------------------------------------------

for name in candidate_names:

    checks.append(
        (
            f"{name} score finite",
            np.isfinite(
                valid[f"score_{name}"]
            ).all()
        )
    )

    checks.append(
        (
            f"{name} rank finite",
            np.isfinite(
                valid[f"rank_{name}"]
            ).all()
        )
    )

    checks.append(
        (
            f"{name} labels valid",
            valid[f"label_{name}"].isin(
                [0, 1, 2]
            ).all()
        )
    )


# ------------------------------------------------------------
# PRINT CHECK RESULTS
# ------------------------------------------------------------

for name, passed in checks:

    print(
        f"{'PASS' if passed else 'FAIL'} - {name}"
    )


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

if not all(
    passed
    for _, passed in checks
):

    raise RuntimeError(
        "One or more audit checks failed."
    )


print()
print("FINAL STATUS: PASS")

print()
print(
    "No final 2Y target file was generated. "
    "This script is diagnostic only."
)