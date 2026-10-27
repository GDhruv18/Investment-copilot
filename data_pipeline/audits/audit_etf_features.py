from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DIR = PROJECT_ROOT / "data" / "clean" / "etfs"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "etf_features.parquet"


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

EXPECTED_ETFS = 41

EXPECTED_START = pd.Timestamp("2014-01-01")
EXPECTED_END = pd.Timestamp("2026-10-23")

REQUIRED_FEATURES = [
    # Momentum
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",

    # Volatility
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",

    # Drawdown
    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",

    # Trend
    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_sma_200",

    # Volume
    "volume_ratio_20d",
    "volume_ratio_60d",
    "volume_change_20d",

    # Trading range
    "range_20d",
    "range_60d",

    # Cross-sectional percentiles
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]

PERCENTILE_FEATURES = [
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_calendar_dates(values):
    """
    Normalize timestamps to calendar dates while preserving
    the actual calendar day.

    Avoids accidental timezone conversion shifting dates.
    """

    if isinstance(values, pd.Series):

        dt = pd.to_datetime(values, errors="coerce")

        if isinstance(dt.dtype, pd.DatetimeTZDtype):
            return dt.dt.tz_localize(None).dt.normalize()

        return dt.dt.normalize()

    if isinstance(values, pd.DatetimeIndex):

        if values.tz is not None:
            return values.tz_localize(None).normalize()

        return values.normalize()

    dt = pd.to_datetime(values, errors="coerce")

    if isinstance(dt, pd.DatetimeIndex):

        if dt.tz is not None:
            return dt.tz_localize(None).normalize()

        return dt.normalize()

    return pd.Series(dt).dt.normalize()


def extract_source_dates(df, file_name):
    """
    Extract the date series from a clean ETF dataframe.

    Clean ETF files may store dates either:
      1. as a Date/date column, or
      2. as the DataFrame index.

    The generator successfully processed these files, so the audit
    supports both layouts.
    """

    # Case 1: explicit date column
    for column in ["Date", "date"]:

        if column in df.columns:
            dates = normalize_calendar_dates(df[column])

            if dates.isna().any():
                raise ValueError(
                    f"{file_name}: null dates detected."
                )

            return dates

    # Case 2: date stored in index
    if isinstance(df.index, pd.DatetimeIndex):

        dates = normalize_calendar_dates(df.index)

        if dates.isna().any():
            raise ValueError(
                f"{file_name}: null dates detected in index."
            )

        return pd.Series(
            dates,
            index=df.index
        ).reset_index(drop=True)

    # Case 3: attempt to interpret index as dates
    try:

        dates = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        if not dates.isna().any():

            dates = normalize_calendar_dates(dates)

            return pd.Series(
                dates,
                index=df.index
            ).reset_index(drop=True)

    except Exception:
        pass

    raise ValueError(
        f"{file_name}: could not identify a date column or "
        f"datetime index."
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("ETF FEATURE AUDIT")
print("=" * 70)

print("Clean ETF directory:")
print(CLEAN_DIR)

print()
print("Feature file:")
print(FEATURE_FILE)


# ============================================================
# 1. FILE EXISTENCE
# ============================================================

print()
print("=" * 70)
print("1. FILE EXISTENCE")
print("=" * 70)

if not CLEAN_DIR.exists():
    raise FileNotFoundError(
        f"Clean ETF directory does not exist:\n{CLEAN_DIR}"
    )

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"ETF feature file does not exist:\n{FEATURE_FILE}"
    )

print("Clean ETF directory: PASS")
print("Feature file: PASS")


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print()
print("=" * 70)
print("2. LOAD FEATURE DATA")
print("=" * 70)

features = pd.read_parquet(FEATURE_FILE)

print(f"Feature rows: {len(features):,}")
print(f"Feature columns: {len(features.columns)}")


# ============================================================
# 3. COLUMN VALIDATION
# ============================================================

print()
print("=" * 70)
print("3. COLUMN VALIDATION")
print("=" * 70)

required_columns = [
    "ticker",
    "date",
] + REQUIRED_FEATURES

missing_columns = [
    col
    for col in required_columns
    if col not in features.columns
]

if missing_columns:

    print("Missing required columns:")

    for col in missing_columns:
        print(f"  - {col}")

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: missing columns."
    )

print("All required columns present: PASS")

actual_feature_columns = [
    c
    for c in features.columns
    if c not in ["ticker", "date"]
]

print()
print("Feature count:")
print(f"Expected: {len(REQUIRED_FEATURES)}")
print(
    f"Actual feature columns: "
    f"{len(actual_feature_columns)}"
)

if len(actual_feature_columns) != 27:
    raise ValueError(
        f"Expected 27 feature columns, "
        f"found {len(actual_feature_columns)}"
    )

if len(features.columns) != 29:
    raise ValueError(
        f"Expected 29 total columns, "
        f"found {len(features.columns)}"
    )

print("Total column count: PASS")


# ============================================================
# 4. DATE VALIDATION
# ============================================================

print()
print("=" * 70)
print("4. DATE VALIDATION")
print("=" * 70)

features["date"] = normalize_calendar_dates(
    features["date"]
)

if features["date"].isna().any():

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: null dates detected."
    )

print("Null dates: 0")

min_date = features["date"].min()
max_date = features["date"].max()

print(f"Min date: {min_date.date()}")
print(f"Max date: {max_date.date()}")

if min_date < EXPECTED_START:

    raise ValueError(
        f"Rows exist before expected start date: "
        f"{EXPECTED_START.date()}"
    )

if max_date > EXPECTED_END:

    raise ValueError(
        f"Rows exist after expected end date: "
        f"{EXPECTED_END.date()}"
    )

print("Date boundary validation: PASS")


# ============================================================
# 5. ETF UNIVERSE VALIDATION
# ============================================================

print()
print("=" * 70)
print("5. ETF UNIVERSE VALIDATION")
print("=" * 70)

feature_etfs = sorted(
    features["ticker"]
    .dropna()
    .astype(str)
    .unique()
)

print(
    f"ETFs in feature file: "
    f"{len(feature_etfs)}"
)

if len(feature_etfs) != EXPECTED_ETFS:

    print("ETF universe:")

    for ticker in feature_etfs:
        print(f"  {ticker}")

    raise ValueError(
        f"Expected {EXPECTED_ETFS} ETFs, "
        f"found {len(feature_etfs)}"
    )

print("ETF count: PASS")

if "DYNAMIC_NS" in feature_etfs:

    raise ValueError(
        "DYNAMIC_NS is present in ETF features."
    )

print("DYNAMIC_NS exclusion: PASS")


# ============================================================
# 6. DUPLICATE KEY CHECK
# ============================================================

print()
print("=" * 70)
print("6. DUPLICATE KEY VALIDATION")
print("=" * 70)

duplicate_keys = features.duplicated(
    subset=["ticker", "date"]
).sum()

print(
    f"Duplicate ticker/date keys: "
    f"{duplicate_keys}"
)

if duplicate_keys != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "duplicate ticker/date keys."
    )

print("Duplicate key validation: PASS")


# ============================================================
# 7. NULL KEY CHECK
# ============================================================

print()
print("=" * 70)
print("7. NULL KEY VALIDATION")
print("=" * 70)

null_tickers = features["ticker"].isna().sum()
null_dates = features["date"].isna().sum()

print(f"Null tickers: {null_tickers}")
print(f"Null dates: {null_dates}")

if null_tickers != 0 or null_dates != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "null key values detected."
    )

print("Null key validation: PASS")


# ============================================================
# 8. INFINITE VALUE CHECK
# ============================================================

print()
print("=" * 70)
print("8. INFINITE VALUE VALIDATION")
print("=" * 70)

numeric_features = features[
    REQUIRED_FEATURES
]

infinite_count = np.isinf(
    numeric_features.to_numpy(
        dtype=float
    )
).sum()

print(
    f"Infinite values: "
    f"{infinite_count}"
)

if infinite_count != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "infinite values detected."
    )

print("Infinite value validation: PASS")


# ============================================================
# 9. SOURCE KEY PRESERVATION
# ============================================================

print()
print("=" * 70)
print("9. SOURCE KEY PRESERVATION")
print("=" * 70)

clean_files = sorted(
    CLEAN_DIR.glob("*.parquet")
)

print(
    f"Clean source files: "
    f"{len(clean_files)}"
)

if len(clean_files) != EXPECTED_ETFS:

    raise ValueError(
        f"Expected {EXPECTED_ETFS} clean ETF files, "
        f"found {len(clean_files)}"
    )


source_frames = []


for file in clean_files:

    ticker = file.stem

    if ticker == "DYNAMIC_NS":

        raise ValueError(
            "DYNAMIC_NS found in clean ETF directory."
        )

    df = pd.read_parquet(file)

    dates = extract_source_dates(
        df,
        file.name
    )

    temp = pd.DataFrame({
        "ticker": ticker,
        "date": dates,
    })

    source_frames.append(temp)


source_keys = pd.concat(
    source_frames,
    ignore_index=True
)

source_keys = source_keys.drop_duplicates(
    subset=["ticker", "date"]
)

feature_keys = features[
    ["ticker", "date"]
].drop_duplicates()


# Source keys missing from feature data
source_compare = source_keys.merge(
    feature_keys,
    on=["ticker", "date"],
    how="left",
    indicator=True
)

source_missing = (
    source_compare["_merge"] == "left_only"
).sum()


# Feature keys absent from source
feature_compare = feature_keys.merge(
    source_keys,
    on=["ticker", "date"],
    how="left",
    indicator=True
)

feature_extra = (
    feature_compare["_merge"] == "left_only"
).sum()


print(
    f"Source rows: "
    f"{len(source_keys):,}"
)

print(
    f"Feature rows: "
    f"{len(feature_keys):,}"
)

print(
    f"Source keys missing from features: "
    f"{source_missing}"
)

print(
    f"Feature keys absent from source: "
    f"{feature_extra}"
)

if source_missing != 0 or feature_extra != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "source/feature key mismatch."
    )

print("Source key preservation: PASS")


# ============================================================
# 10. PER-ETF ROW COUNT VALIDATION
# ============================================================

print()
print("=" * 70)
print("10. PER-ETF ROW COUNT VALIDATION")
print("=" * 70)

source_counts = (
    source_keys
    .groupby("ticker")
    .size()
    .sort_index()
)

feature_counts = (
    feature_keys
    .groupby("ticker")
    .size()
    .sort_index()
)

count_compare = pd.concat(
    [
        source_counts.rename("source"),
        feature_counts.rename("features"),
    ],
    axis=1
).fillna(0)

count_compare["difference"] = (
    count_compare["features"]
    -
    count_compare["source"]
)

print(
    count_compare.to_string()
)

count_mismatches = (
    count_compare["difference"] != 0
).sum()

print()
print(
    f"Per-ETF row count mismatches: "
    f"{count_mismatches}"
)

if count_mismatches != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "per-ETF row count mismatch."
    )

print("Per-ETF row counts: PASS")


# ============================================================
# 11. TEMPORAL ORDER CHECK
# ============================================================

print()
print("=" * 70)
print("11. TEMPORAL ORDER VALIDATION")
print("=" * 70)

sorted_features = features.sort_values(
    ["ticker", "date"]
).copy()

date_diff = (
    sorted_features
    .groupby("ticker")["date"]
    .diff()
)

temporal_violations = (
    date_diff.dropna()
    <= pd.Timedelta(0)
).sum()

print(
    f"Temporal ordering violations: "
    f"{temporal_violations}"
)

if temporal_violations != 0:

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "temporal ordering violation."
    )

print("Temporal ordering: PASS")


# ============================================================
# 12. CROSS-SECTIONAL PERCENTILE VALIDATION
# ============================================================

print()
print("=" * 70)
print("12. CROSS-SECTIONAL PERCENTILE VALIDATION")
print("=" * 70)

percentile_failures = []

for col in PERCENTILE_FEATURES:

    series = features[col].dropna()

    if len(series) == 0:

        percentile_failures.append(
            f"{col}: all values are NaN"
        )

        continue

    min_value = series.min()
    max_value = series.max()
    unique_values = series.nunique()

    print(
        f"{col:<25} "
        f"min={min_value:.6f} "
        f"max={max_value:.6f} "
        f"unique={unique_values}"
    )

    if min_value < 0 or max_value > 1:

        percentile_failures.append(
            f"{col}: values outside [0,1]"
        )

    if unique_values <= 1:

        percentile_failures.append(
            f"{col}: only one unique value"
        )


if percentile_failures:

    print()
    print("Percentile validation failures:")

    for failure in percentile_failures:
        print(f"  - {failure}")

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "percentile validation."
    )

print()
print("Percentile range validation: PASS")


# ============================================================
# 13. CROSS-SECTIONAL VARIATION CHECK
# ============================================================

print()
print("=" * 70)
print("13. CROSS-SECTIONAL VARIATION CHECK")
print("=" * 70)

for col in PERCENTILE_FEATURES:

    grouped = (
        features
        .groupby("date")[col]
        .nunique()
    )

    grouped = grouped[grouped > 0]

    if grouped.empty:

        raise ValueError(
            f"{col}: no valid cross-sectional dates."
        )

    max_unique = grouped.max()

    dates_with_multiple = (
        grouped > 1
    ).sum()

    print(
        f"{col:<25} "
        f"max_unique_per_date={max_unique:<3} "
        f"dates_with_multiple_values="
        f"{dates_with_multiple:,}"
    )

    if max_unique <= 1:

        raise ValueError(
            f"{col}: no meaningful "
            f"cross-sectional variation."
        )

print()
print("Cross-sectional variation: PASS")


# ============================================================
# 14. ROLLING WARM-UP VALIDATION
# ============================================================

print()
print("=" * 70)
print("14. ROLLING WARM-UP VALIDATION")
print("=" * 70)

warmup_checks = {

    "return_5d": 5,
    "return_20d": 20,
    "return_60d": 60,
    "return_120d": 120,
    "return_252d": 252,

    "volatility_20d": 20,
    "volatility_60d": 60,
    "volatility_120d": 120,
    "volatility_252d": 252,

    "price_vs_sma_20": 20,
    "price_vs_sma_50": 50,
    "price_vs_sma_200": 200,

    "range_20d": 20,
    "range_60d": 60,
}


warmup_failures = []


for ticker, group in (
    features
    .sort_values(
        ["ticker", "date"]
    )
    .groupby("ticker")
):

    group = group.reset_index(
        drop=True
    )

    for feature, required_history in (
        warmup_checks.items()
    ):

        if feature not in group.columns:
            continue

        if len(group) <= required_history:
            continue

        first_valid = (
            group[feature]
            .first_valid_index()
        )

        if first_valid is None:
            continue

        if first_valid < required_history - 1:

            warmup_failures.append(
                (
                    ticker,
                    feature,
                    first_valid,
                    required_history
                )
            )


print(
    f"Warm-up violations: "
    f"{len(warmup_failures)}"
)

if warmup_failures:

    print()
    print("First warm-up violations:")

    for item in warmup_failures[:20]:

        print(
            f"  {item[0]} | "
            f"{item[1]} | "
            f"first_valid={item[2]} | "
            f"required_history={item[3]}"
        )

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "rolling warm-up validation."
    )

print(
    "Rolling warm-up validation: PASS"
)


# ============================================================
# 15. NAN SUMMARY
# ============================================================

print()
print("=" * 70)
print("15. NaN SUMMARY")
print("=" * 70)

nan_summary = (
    features[REQUIRED_FEATURES]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

total_rows = len(features)

for feature, count in nan_summary.items():

    if count > 0:

        percentage = (
            count / total_rows
        ) * 100

        print(
            f"{feature:<25} "
            f"{count:>8,} "
            f"({percentage:>6.3f}%)"
        )

print()
print(
    "NaNs are reported only; "
    "they are not automatically filled "
    "or modified."
)


# ============================================================
# 16. FEATURE VARIANCE CHECK
# ============================================================

print()
print("=" * 70)
print("16. FEATURE VARIANCE CHECK")
print("=" * 70)

near_zero_variance = []


for feature in REQUIRED_FEATURES:

    valid = (
        features[feature]
        .dropna()
    )

    if valid.empty:
        continue

    unique_count = valid.nunique()

    if unique_count <= 1:

        near_zero_variance.append(
            (
                feature,
                unique_count
            )
        )


print(
    f"Near-zero/constant features: "
    f"{len(near_zero_variance)}"
)

if near_zero_variance:

    for feature, unique_count in (
        near_zero_variance
    ):

        print(
            f"  {feature}: "
            f"{unique_count} unique value(s)"
        )

    raise ValueError(
        "ETF FEATURE AUDIT FAILED: "
        "constant feature detected."
    )

print(
    "Feature variance validation: PASS"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL ETF FEATURE AUDIT SUMMARY")
print("=" * 70)

print(
    f"Rows:               {len(features):,}"
)

print(
    f"ETFs:               "
    f"{features['ticker'].nunique()}"
)

print(
    f"Features:           "
    f"{len(REQUIRED_FEATURES)}"
)

print(
    f"Total columns:      "
    f"{len(features.columns)}"
)

print(
    f"Min date:           "
    f"{min_date.date()}"
)

print(
    f"Max date:           "
    f"{max_date.date()}"
)

print(
    f"Duplicate keys:     "
    f"{duplicate_keys}"
)

print(
    f"Infinite values:    "
    f"{infinite_count}"
)

print(
    f"Source key missing: "
    f"{source_missing}"
)

print(
    f"Feature key extra:  "
    f"{feature_extra}"
)

print(
    f"Count mismatches:   "
    f"{count_mismatches}"
)

print(
    f"Temporal errors:    "
    f"{temporal_violations}"
)

print(
    f"Warm-up errors:     "
    f"{len(warmup_failures)}"
)

print(
    f"Constant features:  "
    f"{len(near_zero_variance)}"
)

print()
print("=" * 70)
print("ETF FEATURE AUDIT: PASS")
print("=" * 70)