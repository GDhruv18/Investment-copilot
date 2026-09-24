from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# FINAL ETF 6M / 1Y TARGET GENERATION
# ============================================================
#
# 6M:
#   Raw forward 6-calendar-month return
#
# 1Y:
#   Raw forward 1-calendar-year return
#
# For each horizon:
#   - rank ETFs cross-sectionally on the same date
#   - ETF vs ETF only
#   - minimum cohort size = 10
#   - bottom 30%  -> Weak
#   - middle 40% -> Neutral
#   - top 30%    -> Attractive
#
# Raw forward returns are preserved.
# No source price data is modified.
# ============================================================


INPUT_FILE = Path(
    "data/targets/etf_forward_outcomes_clean.parquet"
)

OUTPUT_FILE = Path(
    "data/targets/etf_6m_1y_targets.parquet"
)

MIN_COHORT_SIZE = 10


# ============================================================
# EXPECTED ETF UNIVERSE
# ============================================================

EXPECTED_ETFS = {
    "ALPHA.NS",
    "AUTOBEES.NS",
    "BANKBEES.NS",
    "BSE500IETF.NS",
    "CONSUMBEES.NS",
    "CPSEETF.NS",
    "ESG.NS",
    "GOLDBEES.NS",
    "HDFCGOLD.NS",
    "HDFCNEXT50.NS",
    "HDFCNIFTY.NS",
    "HDFCSENSEX.NS",
    "HNGSNGBEES.NS",
    "ICICIB22.NS",
    "ITBEES.NS",
    "JUNIORBEES.NS",
    "LIQUIDBEES.NS",
    "LIQUIDCASE.NS",
    "LOWVOL1.NS",
    "MAFANG.NS",
    "METAL.NS",
    "MID150BEES.NS",
    "MOM100.NS",
    "MOM30IETF.NS",
    "MOMENTUM.NS",
    "MON100.NS",
    "MONQ50.NS",
    "NIFTY1.NS",
    "NIFTYBEES.NS",
    "NIFTYETF.NS",
    "NIFTYIETF.NS",
    "NIFTYQLITY.NS",
    "PHARMABEES.NS",
    "PSUBANK.NS",
    "QUAL30IETF.NS",
    "SETFGOLD.NS",
    "SETFNIF50.NS",
    "SETFNIFBK.NS",
    "SETFNN50.NS",
    "SILVERBEES.NS",
    "TATAGOLD.NS",
}


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FINAL ETF 6M / 1Y TARGET GENERATION")
print("=" * 80)


# ============================================================
# LOAD INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# INPUT VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("INPUT VALIDATION")
print("=" * 80)

print(f"Input rows: {len(df):,}")
print(f"ETFs: {df['Ticker'].nunique()}")

print(
    f"Date range: "
    f"{df['Date'].min().date()} -> "
    f"{df['Date'].max().date()}"
)


# IMPORTANT:
# The clean forward-outcomes file uses these names:
#
#   forward_return_6m
#   forward_return_1y
#
required_columns = [
    "Ticker",
    "Date",
    "forward_return_6m",
    "forward_return_1y",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# ETF UNIVERSE VALIDATION
# ============================================================

actual_etfs = set(
    df["Ticker"].dropna().unique()
)

unexpected = actual_etfs - EXPECTED_ETFS
missing_etfs = EXPECTED_ETFS - actual_etfs

if unexpected:
    raise ValueError(
        f"Unexpected ETFs found: {sorted(unexpected)}"
    )

if missing_etfs:
    raise ValueError(
        f"Expected ETFs missing: {sorted(missing_etfs)}"
    )

if "DYNAMIC.NS" in actual_etfs:
    raise ValueError(
        "DYNAMIC.NS must not be present in the final clean ETF universe."
    )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicates = df.duplicated(
    subset=["Ticker", "Date"]
).sum()

print(
    f"Duplicate ticker/date rows: {duplicates}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate ticker/date rows detected."
    )


# ============================================================
# HORIZON TARGET GENERATOR
# ============================================================

def generate_horizon_targets(
    data,
    return_column,
    horizon_name,
    min_cohort,
):
    """
    Generate cross-sectional percentile and labels
    for one forward-return horizon.

    Higher forward return = better.

    Bottom 30%  -> Weak
    Middle 40% -> Neutral
    Top 30%    -> Attractive
    """

    percentile_column = (
        f"return_percentile_{horizon_name}"
    )

    score_column = (
        f"score_{horizon_name}"
    )

    label_column = (
        f"label_{horizon_name}"
    )

    cohort_column = (
        f"cohort_size_{horizon_name}"
    )

    eligible_column = (
        f"eligible_for_{horizon_name}"
    )

    result = data.copy()

    # --------------------------------------------------------
    # Initialize columns
    # --------------------------------------------------------

    result[percentile_column] = np.nan
    result[score_column] = np.nan

    result[label_column] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result[cohort_column] = 0
    result[eligible_column] = False

    # --------------------------------------------------------
    # Valid forward returns
    # --------------------------------------------------------

    valid_mask = result[
        return_column
    ].notna()

    # --------------------------------------------------------
    # Cross-sectional cohort size
    # --------------------------------------------------------

    cohort_sizes = (
        result.loc[valid_mask]
        .groupby("Date")["Ticker"]
        .nunique()
        .rename(cohort_column)
    )

    result = (
        result.drop(
            columns=[cohort_column]
        )
        .join(
            cohort_sizes,
            on="Date"
        )
    )

    result[cohort_column] = (
        result[cohort_column]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Eligible observations
    # --------------------------------------------------------

    result[eligible_column] = (
        valid_mask
        & (
            result[cohort_column]
            >= min_cohort
        )
    )

    eligible = result[
        result[eligible_column]
    ].copy()

    # --------------------------------------------------------
    # Cross-sectional percentile
    # --------------------------------------------------------
    #
    # Higher return = higher percentile
    #
    # pct=True gives values between 0 and 1.
    #
    # IMPORTANT:
    # Ranking is performed separately for each Date.
    # Therefore ETFs are compared only against ETFs
    # available on the same historical date.
    # --------------------------------------------------------

    eligible[percentile_column] = (
        eligible
        .groupby("Date")[return_column]
        .rank(
            method="average",
            pct=True
        )
    )

    # For 6M / 1Y the score is simply
    # the cross-sectional return percentile.
    eligible[score_column] = (
        eligible[percentile_column]
    )

    # --------------------------------------------------------
    # Assign labels
    # --------------------------------------------------------

    def assign_labels(group):

        labels = pd.Series(
            pd.NA,
            index=group.index,
            dtype="string"
        )

        rank_pct = (
            group[score_column]
            .rank(
                method="first",
                pct=True
            )
        )

        labels.loc[
            rank_pct <= 0.30
        ] = "Weak"

        labels.loc[
            (rank_pct > 0.30)
            & (rank_pct <= 0.70)
        ] = "Neutral"

        labels.loc[
            rank_pct > 0.70
        ] = "Attractive"

        return labels

    eligible[label_column] = (
        eligible
        .groupby(
            "Date",
            group_keys=False
        )
        .apply(
            assign_labels,
            include_groups=False
        )
    )

    # --------------------------------------------------------
    # Copy results back
    # --------------------------------------------------------

    result.loc[
        eligible.index,
        percentile_column
    ] = eligible[
        percentile_column
    ]

    result.loc[
        eligible.index,
        score_column
    ] = eligible[
        score_column
    ]

    result.loc[
        eligible.index,
        label_column
    ] = eligible[
        label_column
    ]

    result[label_column] = (
        result[label_column]
        .astype("string")
    )

    return result


# ============================================================
# GENERATE 6M TARGET
# ============================================================

print("\n" + "=" * 80)
print("GENERATING 6M TARGET")
print("=" * 80)

df = generate_horizon_targets(
    df,
    "forward_return_6m",
    "6m",
    MIN_COHORT_SIZE,
)

print(
    f"Valid 6M returns: "
    f"{df['forward_return_6m'].notna().sum():,}"
)

print(
    f"Eligible 6M rows: "
    f"{df['eligible_for_6m'].sum():,}"
)

print(
    f"Labeled 6M rows: "
    f"{df['label_6m'].notna().sum():,}"
)


# ============================================================
# GENERATE 1Y TARGET
# ============================================================

print("\n" + "=" * 80)
print("GENERATING 1Y TARGET")
print("=" * 80)

df = generate_horizon_targets(
    df,
    "forward_return_1y",
    "1y",
    MIN_COHORT_SIZE,
)

print(
    f"Valid 1Y returns: "
    f"{df['forward_return_1y'].notna().sum():,}"
)

print(
    f"Eligible 1Y rows: "
    f"{df['eligible_for_1y'].sum():,}"
)

print(
    f"Labeled 1Y rows: "
    f"{df['label_1y'].notna().sum():,}"
)


# ============================================================
# LABEL DISTRIBUTIONS
# ============================================================

for horizon in ["6m", "1y"]:

    print("\n" + "=" * 80)
    print(f"{horizon.upper()} LABEL DISTRIBUTION")
    print("=" * 80)

    labels = df[
        f"label_{horizon}"
    ].dropna()

    counts = labels.value_counts()

    total = len(labels)

    for label in [
        "Weak",
        "Neutral",
        "Attractive"
    ]:

        count = counts.get(
            label,
            0
        )

        pct = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{label:<12}: "
            f"{count:>8,} "
            f"({pct:>6.2f}%)"
        )


# ============================================================
# COHORT VALIDATION
# ============================================================

for horizon in ["6m", "1y"]:

    print("\n" + "=" * 80)
    print(f"{horizon.upper()} COHORT VALIDATION")
    print("=" * 80)

    eligible = df[
        df[f"eligible_for_{horizon}"]
    ]

    cohort_column = (
        f"cohort_size_{horizon}"
    )

    print(
        f"Eligible dates: "
        f"{eligible['Date'].nunique():,}"
    )

    print(
        f"Eligible observations: "
        f"{len(eligible):,}"
    )

    print(
        f"Minimum cohort: "
        f"{eligible[cohort_column].min()}"
    )

    print(
        f"Maximum cohort: "
        f"{eligible[cohort_column].max()}"
    )

    if (
        not eligible.empty
        and eligible[cohort_column].min()
        < MIN_COHORT_SIZE
    ):
        raise ValueError(
            f"{horizon}: cohort below minimum."
        )


# ============================================================
# SCORE VALIDATION
# ============================================================

for horizon in ["6m", "1y"]:

    score_column = (
        f"score_{horizon}"
    )

    labeled = df[
        df[f"label_{horizon}"].notna()
    ]

    if labeled.empty:
        raise ValueError(
            f"No labeled {horizon} observations."
        )

    min_score = labeled[
        score_column
    ].min()

    max_score = labeled[
        score_column
    ].max()

    print(
        f"\n{horizon.upper()} score range: "
        f"{min_score:.6f} -> "
        f"{max_score:.6f}"
    )

    invalid = (
        (labeled[score_column] < 0)
        | (labeled[score_column] > 1)
    ).sum()

    print(
        f"{horizon.upper()} invalid scores: "
        f"{invalid}"
    )

    if invalid > 0:
        raise ValueError(
            f"Invalid {horizon} scores."
        )


# ============================================================
# DYNAMIC CHECK
# ============================================================

dynamic_count = (
    df["Ticker"]
    .eq("DYNAMIC.NS")
    .sum()
)

print(
    f"\nDYNAMIC.NS rows: {dynamic_count}"
)

if dynamic_count > 0:
    raise ValueError(
        "DYNAMIC.NS found in final target."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RELOAD VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("RELOAD VALIDATION")
print("=" * 80)

reloaded = pd.read_parquet(
    OUTPUT_FILE
)

print(
    f"Saved rows: "
    f"{len(reloaded):,}"
)

print(
    f"Saved ETFs: "
    f"{reloaded['Ticker'].nunique()}"
)

print(
    f"Saved 6M labels: "
    f"{reloaded['label_6m'].notna().sum():,}"
)

print(
    f"Saved 1Y labels: "
    f"{reloaded['label_1y'].notna().sum():,}"
)

print(
    f"Saved date range: "
    f"{reloaded['Date'].min().date()} -> "
    f"{reloaded['Date'].max().date()}"
)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

if len(reloaded) != len(df):
    raise ValueError(
        "Reloaded row count mismatch."
    )

if reloaded["Ticker"].nunique() != 41:
    raise ValueError(
        "Expected exactly 41 ETFs."
    )

if reloaded["Ticker"].eq(
    "DYNAMIC.NS"
).any():
    raise ValueError(
        "DYNAMIC.NS found."
    )

duplicates = reloaded.duplicated(
    subset=["Ticker", "Date"]
).sum()

if duplicates > 0:
    raise ValueError(
        "Duplicate ticker/date rows found."
    )


# Every eligible row must have a label
for horizon in ["6m", "1y"]:

    bad = (
        reloaded[
            f"eligible_for_{horizon}"
        ]
        & reloaded[
            f"label_{horizon}"
        ].isna()
    ).sum()

    print(
        f"{horizon.upper()} eligible rows "
        f"without labels: {bad}"
    )

    if bad > 0:
        raise ValueError(
            f"{horizon}: eligible rows missing labels."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"Output file: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(reloaded):,}"
)

print(
    f"ETFs: {reloaded['Ticker'].nunique()}"
)

print(
    f"Minimum cohort: {MIN_COHORT_SIZE}"
)

print(
    "6M: 30% Weak / 40% Neutral / 30% Attractive"
)

print(
    "1Y: 30% Weak / 40% Neutral / 30% Attractive"
)

print(
    "\nRaw prices and source outcome data were not modified."
)

print(
    "ETF 6M / 1Y target generation complete."
)

print("=" * 80)