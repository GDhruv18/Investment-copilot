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

# We will compare these thresholds.
MIN_COHORTS = [10, 20, 30, 40, 50]


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


def safe_corr(a, b):

    temp = pd.concat(
        [a, b],
        axis=1
    ).dropna()

    if len(temp) < 2:
        return np.nan

    return temp.iloc[:, 0].corr(
        temp.iloc[:, 1]
    )


# ============================================================
# LOAD
# ============================================================

print_section("MUTUAL FUND 2Y COHORT AUDIT")

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


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
    )


duplicates = df.duplicated(
    ["scheme_code", "nav_date"]
).sum()


print(
    f"Duplicate scheme/date rows: "
    f"{duplicates:,}"
)


if duplicates != 0:

    raise RuntimeError(
        "Duplicate scheme/date rows detected."
    )


print(
    f"Universe check: "
    f"{'PASS' if df['scheme_code'].nunique() == EXPECTED_SCHEMES else 'FAIL'}"
)


# ============================================================
# VALID 2Y OBSERVATIONS
# ============================================================

valid = df[
    df["annualized_return_2y"].notna()
    & df["annualized_volatility_2y"].notna()
    & df["max_drawdown_2y"].notna()
].copy()


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
# ORIGINAL COHORT STRUCTURE
# ============================================================

print_section("COHORT STRUCTURE")


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
    f"Maximum cohort: {cohort_sizes.max():,}"
)


# ============================================================
# COMPONENT RANKING FUNCTION
# ============================================================

def prepare_components(data):

    data = data.copy()

    # Higher return = better
    data["return_pct"] = (
        data
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
        data
        .groupby("nav_date")[
            "annualized_volatility_2y"
        ]
        .rank(
            pct=True,
            method="average"
        )
    )

    data["risk_pct"] = (
        1.0 - volatility_rank
    )

    # Less-negative drawdown = better
    data["drawdown_pct"] = (
        data
        .groupby("nav_date")[
            "max_drawdown_2y"
        ]
        .rank(
            pct=True,
            method="average"
        )
    )

    return data


# ============================================================
# COHORT-BY-COHORT ANALYSIS
# ============================================================

results = []


for minimum_cohort in MIN_COHORTS:

    eligible_dates = cohort_sizes[
        cohort_sizes >= minimum_cohort
    ].index

    cohort_data = valid[
        valid["nav_date"].isin(
            eligible_dates
        )
    ].copy()

    cohort_data = prepare_components(
        cohort_data
    )

    # --------------------------------------------------------
    # CANDIDATE SCORES
    # --------------------------------------------------------

    for name, weights in CANDIDATES.items():

        cohort_data[
            f"score_{name}"
        ] = (
            weights["return"]
            * cohort_data["return_pct"]
            +
            weights["risk"]
            * cohort_data["risk_pct"]
            +
            weights["drawdown"]
            * cohort_data["drawdown_pct"]
        )

        # Rank composite score within each date.
        cohort_data[
            f"rank_{name}"
        ] = (
            cohort_data
            .groupby("nav_date")[
                f"score_{name}"
            ]
            .rank(
                pct=True,
                method="average"
            )
        )

        # 30 / 40 / 30 labels.
        cohort_data[
            f"label_{name}"
        ] = np.select(
            [
                cohort_data[
                    f"rank_{name}"
                ] <= 0.30,

                cohort_data[
                    f"rank_{name}"
                ] <= 0.70,
            ],
            [
                0,
                1,
            ],
            default=2
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    row = {
        "minimum_cohort": minimum_cohort,
        "eligible_dates": len(
            eligible_dates
        ),
        "eligible_rows": len(
            cohort_data
        ),
        "row_retention_pct": (
            len(cohort_data)
            / len(valid)
            * 100
        ),
        "min_actual_cohort": (
            cohort_data
            .groupby("nav_date")[
                "scheme_code"
            ]
            .nunique()
            .min()
        ),
        "median_actual_cohort": (
            cohort_data
            .groupby("nav_date")[
                "scheme_code"
            ]
            .nunique()
            .median()
        ),
        "max_actual_cohort": (
            cohort_data
            .groupby("nav_date")[
                "scheme_code"
            ]
            .nunique()
            .max()
        ),
    }

    # --------------------------------------------------------
    # LABEL DISTRIBUTIONS
    # --------------------------------------------------------

    for name in CANDIDATES:

        attractive = (
            cohort_data[
                f"label_{name}"
            ] == 2
        ).sum()

        neutral = (
            cohort_data[
                f"label_{name}"
            ] == 1
        ).sum()

        weak = (
            cohort_data[
                f"label_{name}"
            ] == 0
        ).sum()

        total = len(
            cohort_data
        )

        row[
            f"{name}_weak_pct"
        ] = weak / total * 100

        row[
            f"{name}_neutral_pct"
        ] = neutral / total * 100

        row[
            f"{name}_attractive_pct"
        ] = attractive / total * 100

    results.append(row)


# ============================================================
# RETENTION SUMMARY
# ============================================================

print_section("COHORT RETENTION SUMMARY")


results_df = pd.DataFrame(
    results
)


for _, row in results_df.iterrows():

    print(
        f"Min cohort {int(row['minimum_cohort']):>2}: "
        f"{int(row['eligible_dates']):,} dates | "
        f"{int(row['eligible_rows']):,} rows | "
        f"{row['row_retention_pct']:.2f}% retained | "
        f"actual cohort "
        f"{int(row['min_actual_cohort'])}"
        f"–"
        f"{int(row['max_actual_cohort'])}"
    )


# ============================================================
# LABEL DISTRIBUTION SUMMARY
# ============================================================

print_section("LABEL DISTRIBUTIONS BY COHORT")


for _, row in results_df.iterrows():

    minimum_cohort = int(
        row["minimum_cohort"]
    )

    print()
    print(
        f"Minimum cohort = "
        f"{minimum_cohort}"
    )

    for name in CANDIDATES:

        print(
            f"  {name}: "
            f"Weak "
            f"{row[f'{name}_weak_pct']:.2f}% | "
            f"Neutral "
            f"{row[f'{name}_neutral_pct']:.2f}% | "
            f"Attractive "
            f"{row[f'{name}_attractive_pct']:.2f}%"
        )


# ============================================================
# CANDIDATE RANK STABILITY ACROSS COHORT THRESHOLDS
# ============================================================

print_section(
    "CANDIDATE RANK STABILITY"
)


cohort_candidate_scores = {}


for minimum_cohort in MIN_COHORTS:

    eligible_dates = cohort_sizes[
        cohort_sizes >= minimum_cohort
    ].index

    cohort_data = valid[
        valid["nav_date"].isin(
            eligible_dates
        )
    ].copy()

    cohort_data = prepare_components(
        cohort_data
    )

    for name, weights in CANDIDATES.items():

        cohort_data[
            f"score_{name}"
        ] = (
            weights["return"]
            * cohort_data["return_pct"]
            +
            weights["risk"]
            * cohort_data["risk_pct"]
            +
            weights["drawdown"]
            * cohort_data["drawdown_pct"]
        )

    cohort_candidate_scores[
        minimum_cohort
    ] = cohort_data


for minimum_cohort in MIN_COHORTS:

    print()
    print(
        f"Minimum cohort = "
        f"{minimum_cohort}"
    )

    data = cohort_candidate_scores[
        minimum_cohort
    ]

    for i, name_a in enumerate(
        CANDIDATES
    ):

        for name_b in list(
            CANDIDATES
        )[i + 1:]:

            correlation = safe_corr(
                data[f"score_{name_a}"],
                data[f"score_{name_b}"]
            )

            print(
                f"  {name_a} vs {name_b}: "
                f"{correlation:.4f}"
            )


# ============================================================
# FINAL VALIDATION
# ============================================================

print_section("FINAL AUDIT")


checks = []


checks.append(
    (
        "100 schemes",
        df["scheme_code"].nunique()
        == EXPECTED_SCHEMES
    )
)


checks.append(
    (
        "no duplicate scheme/date rows",
        df.duplicated(
            ["scheme_code", "nav_date"]
        ).sum() == 0
    )
)


checks.append(
    (
        "valid 2Y rows exist",
        len(valid) > 0
    )
)


for minimum_cohort in MIN_COHORTS:

    eligible_dates = cohort_sizes[
        cohort_sizes >= minimum_cohort
    ]

    checks.append(
        (
            f"cohort {minimum_cohort} has dates",
            len(eligible_dates) > 0
        )
    )


for name, passed in checks:

    print(
        f"{'PASS' if passed else 'FAIL'} - {name}"
    )


if not all(
    passed
    for _, passed in checks
):

    raise RuntimeError(
        "One or more cohort audit checks failed."
    )


print()
print("FINAL STATUS: PASS")

print()
print(
    "No final 2Y target file was generated."
)

print(
    "This script is diagnostic only."
)