from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# FINAL ETF 2Y TARGET GENERATION
# ============================================================
#
# Methodology:
#
#   Horizon:
#       2 calendar years
#
#   Components:
#       1. Annualized 2Y return       -> higher is better
#       2. Annualized volatility      -> lower is better
#       3. Maximum drawdown           -> less negative is better
#
#   Cross-sectional scoring:
#       60% Return
#       20% Risk
#       20% Drawdown
#
#   Minimum cohort:
#       15 ETFs with valid 2Y outcomes
#
#   Labels:
#       Bottom 30%  -> Weak
#       Middle 40%  -> Neutral
#       Top 30%     -> Attractive
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/targets/etf_2y_diagnostics.parquet"
)

OUTPUT_FILE = Path(
    "data/targets/etf_2y_targets.parquet"
)


# ============================================================
# LOCKED METHODOLOGY
# ============================================================

MIN_COHORT_SIZE = 15

RETURN_WEIGHT = 0.60
RISK_WEIGHT = 0.20
DRAWDOWN_WEIGHT = 0.20


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
# LOAD DIAGNOSTICS
# ============================================================

print("=" * 80)
print("FINAL ETF 2Y TARGET GENERATION")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

required_columns = [
    "Ticker",
    "Date",
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("INPUT VALIDATION")
print("=" * 80)

print(f"Input rows: {len(df):,}")
print(f"Input ETFs: {df['Ticker'].nunique()}")
print(
    f"Date range: "
    f"{df['Date'].min().date()} -> "
    f"{df['Date'].max().date()}"
)

actual_etfs = set(df["Ticker"].unique())

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
        "DYNAMIC.NS must not be present in the clean ETF target universe."
    )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicates = df.duplicated(
    subset=["Ticker", "Date"]
).sum()

print(f"Duplicate ticker/date rows: {duplicates}")

if duplicates > 0:
    raise ValueError(
        "Duplicate ticker/date rows detected."
    )


# ============================================================
# VALID 2Y OBSERVATIONS
# ============================================================

component_columns = [
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",
]

valid_mask = df[component_columns].notna().all(axis=1)

df["eligible_2y"] = False

df.loc[
    valid_mask,
    "eligible_2y"
] = True


# ============================================================
# CROSS-SECTIONAL COHORT SIZE
# ============================================================

cohort_size = (
    df.loc[valid_mask]
    .groupby("Date")["Ticker"]
    .nunique()
    .rename("cohort_size")
)

df = df.join(
    cohort_size,
    on="Date"
)

df["cohort_size"] = (
    df["cohort_size"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# ELIGIBLE FOR FINAL LABEL
# ============================================================

df["eligible_for_label"] = (
    df["eligible_2y"]
    & (df["cohort_size"] >= MIN_COHORT_SIZE)
)


# ============================================================
# INITIALIZE SCORE COLUMNS
# ============================================================

df["return_percentile_2y"] = np.nan
df["risk_percentile_2y"] = np.nan
df["drawdown_percentile_2y"] = np.nan
df["score_2y"] = np.nan


# ============================================================
# CROSS-SECTIONAL PERCENTILES
# ============================================================

eligible = df[
    df["eligible_for_label"]
].copy()


# Higher return = better
eligible["return_percentile_2y"] = (
    eligible
    .groupby("Date")["annualized_return_2y"]
    .rank(
        method="average",
        pct=True
    )
)


# Lower volatility = better
volatility_percentile = (
    eligible
    .groupby("Date")["volatility_2y"]
    .rank(
        method="average",
        pct=True
    )
)

eligible["risk_percentile_2y"] = (
    1.0 - volatility_percentile
)


# Less negative drawdown = better
eligible["drawdown_percentile_2y"] = (
    eligible
    .groupby("Date")["max_drawdown_2y"]
    .rank(
        method="average",
        pct=True
    )
)


# ============================================================
# FINAL COMPOSITE SCORE
# ============================================================

eligible["score_2y"] = (
    RETURN_WEIGHT
    * eligible["return_percentile_2y"]
    +
    RISK_WEIGHT
    * eligible["risk_percentile_2y"]
    +
    DRAWDOWN_WEIGHT
    * eligible["drawdown_percentile_2y"]
)


# ============================================================
# FINAL 30 / 40 / 30 LABEL
# ============================================================

def assign_label(group):
    """
    Assign labels based on cross-sectional score.

    Bottom 30%  -> Weak
    Middle 40%  -> Neutral
    Top 30%     -> Attractive
    """

    scores = group["score_2y"]

    labels = pd.Series(
        pd.NA,
        index=group.index,
        dtype="object"
    )

    if len(group) < MIN_COHORT_SIZE:
        return labels

    rank_pct = scores.rank(
        method="first",
        pct=True
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


eligible["label_2y"] = (
    eligible
    .groupby("Date", group_keys=False)
    .apply(
        assign_label,
        include_groups=False
    )
)


# ============================================================
# COPY FINAL VALUES BACK
# ============================================================

df.loc[
    eligible.index,
    "return_percentile_2y"
] = eligible["return_percentile_2y"]

df.loc[
    eligible.index,
    "risk_percentile_2y"
] = eligible["risk_percentile_2y"]

df.loc[
    eligible.index,
    "drawdown_percentile_2y"
] = eligible["drawdown_percentile_2y"]

df.loc[
    eligible.index,
    "score_2y"
] = eligible["score_2y"]

df["label_2y"] = pd.NA

df.loc[
    eligible.index,
    "label_2y"
] = eligible["label_2y"]


# ============================================================
# LABEL TYPE
# ============================================================

df["label_2y"] = df["label_2y"].astype("string")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL TARGET VALIDATION")
print("=" * 80)

valid_2y_count = (
    df["eligible_2y"].sum()
)

eligible_label_count = (
    df["eligible_for_label"].sum()
)

labeled_count = (
    df["label_2y"].notna().sum()
)

unlabeled_count = (
    df["label_2y"].isna().sum()
)

print(
    f"Valid 2Y observations: "
    f"{valid_2y_count:,}"
)

print(
    f"Eligible for labeling: "
    f"{eligible_label_count:,}"
)

print(
    f"Labeled observations: "
    f"{labeled_count:,}"
)

print(
    f"Unlabeled observations: "
    f"{unlabeled_count:,}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("FINAL LABEL DISTRIBUTION")
print("=" * 80)

label_counts = (
    df["label_2y"]
    .value_counts(dropna=False)
)

print(label_counts)


labeled = df[
    df["label_2y"].notna()
].copy()


label_counts_non_null = (
    labeled["label_2y"]
    .value_counts()
)

label_total = len(labeled)

print("\nPercentages:")

for label in [
    "Weak",
    "Neutral",
    "Attractive"
]:

    count = label_counts_non_null.get(
        label,
        0
    )

    pct = (
        count / label_total * 100
        if label_total > 0
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

print("\n" + "=" * 80)
print("COHORT VALIDATION")
print("=" * 80)

labeled_cohort_min = (
    labeled
    .groupby("Date")["Ticker"]
    .nunique()
    .min()
)

labeled_cohort_max = (
    labeled
    .groupby("Date")["Ticker"]
    .nunique()
    .max()
)

labeled_dates = (
    labeled["Date"].nunique()
)

print(
    f"Labeled dates: "
    f"{labeled_dates:,}"
)

print(
    f"Minimum labeled cohort: "
    f"{labeled_cohort_min}"
)

print(
    f"Maximum labeled cohort: "
    f"{labeled_cohort_max}"
)

if labeled_cohort_min < MIN_COHORT_SIZE:
    raise ValueError(
        "A labeled date has fewer than the minimum cohort size."
    )


# ============================================================
# LABEL BALANCE PER DATE
# ============================================================

print("\n" + "=" * 80)
print("PER-DATE LABEL COUNTS")
print("=" * 80)

per_date_counts = (
    labeled
    .groupby(["Date", "label_2y"])
    .size()
    .unstack(fill_value=0)
)

for label in [
    "Weak",
    "Neutral",
    "Attractive"
]:

    if label not in per_date_counts.columns:
        per_date_counts[label] = 0

weak_counts = per_date_counts["Weak"]
neutral_counts = per_date_counts["Neutral"]
attractive_counts = per_date_counts["Attractive"]

print(
    f"Weak per date: "
    f"min={weak_counts.min()}, "
    f"median={weak_counts.median()}, "
    f"max={weak_counts.max()}"
)

print(
    f"Neutral per date: "
    f"min={neutral_counts.min()}, "
    f"median={neutral_counts.median()}, "
    f"max={neutral_counts.max()}"
)

print(
    f"Attractive per date: "
    f"min={attractive_counts.min()}, "
    f"median={attractive_counts.median()}, "
    f"max={attractive_counts.max()}"
)


# ============================================================
# SCORE VALIDATION
# ============================================================

score_columns = [
    "return_percentile_2y",
    "risk_percentile_2y",
    "drawdown_percentile_2y",
    "score_2y",
]

print("\n" + "=" * 80)
print("SCORE VALIDATION")
print("=" * 80)

for column in score_columns:

    values = labeled[column]

    print(
        f"{column:<28} "
        f"min={values.min():.6f} "
        f"max={values.max():.6f}"
    )

    if values.isna().any():
        raise ValueError(
            f"NaN values found in labeled column: {column}"
        )


# Score should be between 0 and 1
invalid_scores = (
    (labeled["score_2y"] < 0)
    | (labeled["score_2y"] > 1)
).sum()

print(
    f"Invalid score values: "
    f"{invalid_scores}"
)

if invalid_scores > 0:
    raise ValueError(
        "2Y score outside [0, 1] detected."
    )


# ============================================================
# UNLABELED ROW VALIDATION
# ============================================================

unlabeled = df[
    df["label_2y"].isna()
]

invalid_unlabeled = (
    unlabeled["eligible_for_label"]
).sum()

print(
    f"Rows incorrectly left unlabeled: "
    f"{invalid_unlabeled}"
)

if invalid_unlabeled > 0:
    raise ValueError(
        "Some rows eligible for labeling have no label."
    )


# ============================================================
# DYNAMIC VALIDATION
# ============================================================

dynamic_count = (
    df["Ticker"]
    .eq("DYNAMIC.NS")
    .sum()
)

print(
    f"DYNAMIC.NS rows: "
    f"{dynamic_count}"
)

if dynamic_count > 0:
    raise ValueError(
        "DYNAMIC.NS must not appear in final ETF targets."
    )


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "Ticker",
    "Date",

    # Original 2Y diagnostics
    "target_date_2y",
    "future_date_2y",
    "future_adj_close_2y",
    "annualized_return_2y",
    "volatility_2y",
    "max_drawdown_2y",

    # Cross-sectional information
    "cohort_size",
    "eligible_2y",
    "eligible_for_label",

    # Scoring components
    "return_percentile_2y",
    "risk_percentile_2y",
    "drawdown_percentile_2y",

    # Final score
    "score_2y",

    # Final target
    "label_2y",
]


missing_output_columns = [
    column
    for column in output_columns
    if column not in df.columns
]

if missing_output_columns:
    raise ValueError(
        "Missing output columns: "
        f"{missing_output_columns}"
    )


final_df = df[
    output_columns
].copy()


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_parquet(
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
    f"Saved date range: "
    f"{reloaded['Date'].min().date()} -> "
    f"{reloaded['Date'].max().date()}"
)

print(
    f"Saved labeled rows: "
    f"{reloaded['label_2y'].notna().sum():,}"
)

print(
    f"Saved unlabeled rows: "
    f"{reloaded['label_2y'].isna().sum():,}"
)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

if len(reloaded) != len(df):
    raise ValueError(
        "Reloaded row count does not match input."
    )

if reloaded["Ticker"].nunique() != 41:
    raise ValueError(
        "Final target must contain exactly 41 ETF universe tickers."
    )

if reloaded["Ticker"].eq("DYNAMIC.NS").any():
    raise ValueError(
        "DYNAMIC.NS found in final target."
    )

duplicate_output = reloaded.duplicated(
    subset=["Ticker", "Date"]
).sum()

if duplicate_output > 0:
    raise ValueError(
        "Duplicate ticker/date rows in final target."
    )


# Every labeled row must satisfy minimum cohort
bad_labeled_rows = reloaded.loc[
    reloaded["label_2y"].notna(),
    "cohort_size"
] < MIN_COHORT_SIZE

if bad_labeled_rows.any():
    raise ValueError(
        "Labeled row found below minimum cohort size."
    )


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
    f"Return weight: {RETURN_WEIGHT:.0%}"
)

print(
    f"Risk weight: {RISK_WEIGHT:.0%}"
)

print(
    f"Drawdown weight: {DRAWDOWN_WEIGHT:.0%}"
)

print(
    "Labels: Weak / Neutral / Attractive"
)

print(
    "Label split: 30% / 40% / 30%"
)

print(
    "\nRaw prices and diagnostic files were not modified."
)

print(
    "Final 2Y target generation complete."
)

print("=" * 80)