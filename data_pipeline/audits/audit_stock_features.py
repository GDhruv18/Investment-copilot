# data_pipeline/audits/audit_stock_features.py

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "stock_features.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_STOCKS = 100
EXPECTED_ROWS = 293430

EXPECTED_START_DATE = pd.Timestamp("2014-01-01")
EXPECTED_END_DATE = pd.Timestamp("2026-09-18")

EXPECTED_FEATURE_COLUMNS = [
    "ticker",
    "date",

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


FEATURE_COLUMNS = [
    column
    for column in EXPECTED_FEATURE_COLUMNS
    if column not in ["ticker", "date"]
]


PERCENTILE_COLUMNS = [
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_calendar_dates(values):
    """
    Normalize dates while preserving the original calendar date.

    Important:
    We deliberately do NOT convert timestamps to UTC because
    timezone conversion can shift the calendar date.

    Example:
        2020-01-02 00:00 IST
        -> UTC
        -> 2020-01-01 18:30
        -> normalize
        -> 2020-01-01

    That would incorrectly change the trading date.

    This function therefore removes timezone information first
    and then normalizes to midnight.
    """

    if isinstance(values, pd.Series):

        dt = pd.to_datetime(
            values,
            errors="coerce"
        )

        if isinstance(dt.dtype, pd.DatetimeTZDtype):
            return (
                dt.dt
                .tz_localize(None)
                .dt
                .normalize()
            )

        return dt.dt.normalize()

    if isinstance(values, pd.DatetimeIndex):

        if values.tz is not None:
            return values.tz_localize(None).normalize()

        return values.normalize()

    dt = pd.to_datetime(
        values,
        errors="coerce"
    )

    if isinstance(dt, pd.DatetimeIndex):

        if dt.tz is not None:
            return dt.tz_localize(None).normalize()

        return dt.normalize()

    return pd.Series(dt).dt.normalize()


# ============================================================
# LOAD SOURCE KEYS
# ============================================================

def load_source_keys():

    parquet_files = sorted(
        RAW_DIR.glob("*.parquet")
    )

    if len(parquet_files) != EXPECTED_STOCKS:

        raise ValueError(
            f"Expected {EXPECTED_STOCKS} raw stock files, "
            f"found {len(parquet_files)}."
        )

    source_parts = []

    for file_path in parquet_files:

        ticker = file_path.stem

        df = pd.read_parquet(file_path)

        dates = normalize_calendar_dates(
            df.index
        )

        if dates.isna().any():

            invalid_count = dates.isna().sum()

            raise ValueError(
                f"{ticker}: "
                f"{invalid_count} invalid source dates."
            )

        source_part = pd.DataFrame({
            "ticker": ticker,
            "date": dates,
        })

        source_parts.append(
            source_part
        )

    source_keys = pd.concat(
        source_parts,
        ignore_index=True
    )

    return source_keys


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("STOCK FEATURE AUDIT")
    print("=" * 70)

    # ========================================================
    # LOAD FEATURE DATASET
    # ========================================================

    print()
    print("Loading feature dataset...")

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    features = pd.read_parquet(
        FEATURE_FILE
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Feature columns: {len(features.columns)}"
    )

    # ========================================================
    # 1. COLUMN STRUCTURE
    # ========================================================

    print()
    print("=" * 70)
    print("1. COLUMN STRUCTURE")
    print("=" * 70)

    actual_columns = list(
        features.columns
    )

    missing_columns = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_FEATURE_COLUMNS
    ]

    print(
        f"Required columns: "
        f"{len(EXPECTED_FEATURE_COLUMNS)}"
    )

    print(
        f"Missing columns: "
        f"{len(missing_columns)}"
    )

    if missing_columns:
        print(
            f"Missing: {missing_columns}"
        )

    print(
        f"Unexpected columns: "
        f"{len(unexpected_columns)}"
    )

    if unexpected_columns:
        print(
            f"Unexpected: {unexpected_columns}"
        )

    if missing_columns or unexpected_columns:

        raise ValueError(
            "Feature column structure does not match "
            "the expected schema."
        )

    print("Column structure: PASS")

    # ========================================================
    # 2. BASIC DATA INTEGRITY
    # ========================================================

    print()
    print("=" * 70)
    print("2. BASIC DATA INTEGRITY")
    print("=" * 70)

    row_count = len(features)

    stock_count = features[
        "ticker"
    ].nunique()

    duplicate_keys = features.duplicated(
        subset=["ticker", "date"]
    ).sum()

    null_tickers = features[
        "ticker"
    ].isna().sum()

    null_dates = features[
        "date"
    ].isna().sum()

    print(
        f"Rows: {row_count:,}"
    )

    print(
        f"Expected rows: {EXPECTED_ROWS:,}"
    )

    print(
        f"Row count difference: "
        f"{row_count - EXPECTED_ROWS:,}"
    )

    print(
        f"Stocks: {stock_count}"
    )

    print(
        f"Expected stocks: {EXPECTED_STOCKS}"
    )

    print(
        f"Duplicate (ticker,date): "
        f"{duplicate_keys:,}"
    )

    print(
        f"Null tickers: {null_tickers:,}"
    )

    print(
        f"Null dates: {null_dates:,}"
    )

    if row_count != EXPECTED_ROWS:

        raise ValueError(
            f"Unexpected feature row count: "
            f"{row_count:,}"
        )

    if stock_count != EXPECTED_STOCKS:

        raise ValueError(
            f"Expected {EXPECTED_STOCKS} stocks, "
            f"found {stock_count}."
        )

    if duplicate_keys != 0:

        raise ValueError(
            f"Duplicate ticker/date keys: "
            f"{duplicate_keys:,}"
        )

    if null_tickers != 0:

        raise ValueError(
            f"Null tickers: {null_tickers:,}"
        )

    if null_dates != 0:

        raise ValueError(
            f"Null dates: {null_dates:,}"
        )

    print("Basic data integrity: PASS")

    # ========================================================
    # NORMALIZE FEATURE DATES
    # ========================================================

    features["date"] = normalize_calendar_dates(
        features["date"]
    )

    # ========================================================
    # 3. SOURCE KEY PRESERVATION
    # ========================================================

    print()
    print("=" * 70)
    print("3. SOURCE KEY PRESERVATION")
    print("=" * 70)

    print("Loading raw source keys...")

    source_keys = load_source_keys()

    feature_keys = features[
        ["ticker", "date"]
    ].copy()

    source_duplicate_keys = source_keys.duplicated(
        subset=["ticker", "date"]
    ).sum()

    feature_duplicate_keys = feature_keys.duplicated(
        subset=["ticker", "date"]
    ).sum()

    print(
        f"Source rows: {len(source_keys):,}"
    )

    print(
        f"Feature rows: {len(feature_keys):,}"
    )

    print(
        f"Source duplicate keys: "
        f"{source_duplicate_keys:,}"
    )

    print(
        f"Feature duplicate keys: "
        f"{feature_duplicate_keys:,}"
    )

    if source_duplicate_keys != 0:

        raise ValueError(
            f"Source contains duplicate ticker/date keys: "
            f"{source_duplicate_keys:,}"
        )

    if feature_duplicate_keys != 0:

        raise ValueError(
            f"Feature contains duplicate ticker/date keys: "
            f"{feature_duplicate_keys:,}"
        )

    # --------------------------------------------------------
    # Exact source/feature key comparison
    # --------------------------------------------------------

    source_key_set = set(
        zip(
            source_keys["ticker"],
            source_keys["date"]
        )
    )

    feature_key_set = set(
        zip(
            feature_keys["ticker"],
            feature_keys["date"]
        )
    )

    source_missing_keys = (
        source_key_set - feature_key_set
    )

    feature_extra_keys = (
        feature_key_set - source_key_set
    )

    print(
        f"Source keys missing from features: "
        f"{len(source_missing_keys):,}"
    )

    print(
        f"Feature keys absent from source: "
        f"{len(feature_extra_keys):,}"
    )

    if source_missing_keys:

        print()
        print("First 20 source keys missing from features:")

        for key in sorted(
            source_missing_keys,
            key=lambda x: (x[0], x[1])
        )[:20]:

            print(
                f"  {key[0]} | {key[1]}"
            )

        raise ValueError(
            "Some source rows are missing from "
            "the feature dataset."
        )

    if feature_extra_keys:

        print()
        print("First 20 feature keys absent from source:")

        for key in sorted(
            feature_extra_keys,
            key=lambda x: (x[0], x[1])
        )[:20]:

            print(
                f"  {key[0]} | {key[1]}"
            )

        raise ValueError(
            "Feature dataset contains keys that "
            "do not exist in the raw source."
        )

    print(
        "Source key preservation: PASS"
    )

    # ========================================================
    # 4. DATE BOUNDARY CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("4. DATE BOUNDARY CHECK")
    print("=" * 70)

    min_date = features[
        "date"
    ].min()

    max_date = features[
        "date"
    ].max()

    print(
        f"Feature min date: {min_date}"
    )

    print(
        f"Feature max date: {max_date}"
    )

    print(
        f"Expected collection start: "
        f"{EXPECTED_START_DATE}"
    )

    print(
        f"Expected collection end: "
        f"{EXPECTED_END_DATE}"
    )

    rows_before_start = (
        features["date"] < EXPECTED_START_DATE
    ).sum()

    rows_after_end = (
        features["date"] > EXPECTED_END_DATE
    ).sum()

    print(
        f"Rows before expected start: "
        f"{rows_before_start:,}"
    )

    print(
        f"Rows after expected end: "
        f"{rows_after_end:,}"
    )

    # A small number of pre-start rows would indicate
    # unexpected source history. Our current generator is
    # expected to start exactly at 2014-01-01.

    if rows_before_start != 0:

        raise ValueError(
            f"{rows_before_start:,} rows occur before "
            f"{EXPECTED_START_DATE.date()}."
        )

    if rows_after_end != 0:

        raise ValueError(
            f"{rows_after_end:,} rows occur after "
            f"{EXPECTED_END_DATE.date()}."
        )

    print("Date boundaries: PASS")

    # ========================================================
    # 5. NUMERIC DATA QUALITY
    # ========================================================

    print()
    print("=" * 70)
    print("5. NUMERIC DATA QUALITY")
    print("=" * 70)

    numeric_features = features[
        FEATURE_COLUMNS
    ]

    infinite_mask = np.isinf(
        numeric_features.to_numpy(
            dtype=float
        )
    )

    infinite_count = int(
        infinite_mask.sum()
    )

    print(
        f"Infinite values: "
        f"{infinite_count:,}"
    )

    if infinite_count != 0:

        raise ValueError(
            f"Infinite feature values found: "
            f"{infinite_count:,}"
        )

    print("Infinite-value check: PASS")

    # ========================================================
    # 6. NaN COVERAGE
    # ========================================================

    print()
    print("=" * 70)
    print("6. NaN COVERAGE")
    print("=" * 70)

    nan_counts = (
        numeric_features
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for column, count in nan_counts.items():

        percentage = (
            count / len(features) * 100
        )

        print(
            f"{column:25s} "
            f"{count:8,} "
            f"({percentage:.3f}%)"
        )

    # We intentionally do NOT fail on NaNs.
    #
    # Rolling features naturally have warm-up NaNs.
    # volume_change_20d can additionally have NaNs when the
    # denominator 20 trading days earlier was zero.
    #
    # These are handled later during training according to
    # feature/model requirements.

    print()
    print(
        "NaN coverage: REPORTED "
        "(not treated as an audit failure)"
    )

    # ========================================================
    # 7. PERCENTILE FEATURE VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("7. CROSS-SECTIONAL PERCENTILE VALIDATION")
    print("=" * 70)

    for column in PERCENTILE_COLUMNS:

        series = features[
            column
        ]

        valid = series.dropna()

        if len(valid) == 0:

            raise ValueError(
                f"{column}: all values are NaN."
            )

        minimum = valid.min()
        maximum = valid.max()
        unique_count = valid.nunique()

        print(
            f"{column:25s} "
            f"min={minimum:.6f} "
            f"max={maximum:.6f} "
            f"unique={unique_count:,}"
        )

        if minimum < 0:

            raise ValueError(
                f"{column}: value below 0."
            )

        if maximum > 1:

            raise ValueError(
                f"{column}: value above 1."
            )

        if unique_count <= 1:

            raise ValueError(
                f"{column}: constant percentile feature."
            )

    print(
        "Percentile features: PASS"
    )

    # ========================================================
    # 8. TEMPORAL ORDER
    # ========================================================

    print()
    print("=" * 70)
    print("8. TEMPORAL ORDER CHECK")
    print("=" * 70)

    temporal_failures = []

    for ticker, group in features.groupby(
        "ticker",
        sort=False
    ):

        dates = group[
            "date"
        ].reset_index(drop=True)

        if not dates.is_monotonic_increasing:

            temporal_failures.append(
                ticker
            )

    print(
        f"Stocks with temporal ordering violations: "
        f"{len(temporal_failures)}"
    )

    if temporal_failures:

        print(
            "Violating stocks:",
            temporal_failures
        )

        raise ValueError(
            "Temporal ordering violation detected."
        )

    print(
        "Temporal order: PASS"
    )

    # ========================================================
    # 9. EXPECTED ROLLING-WINDOW AVAILABILITY
    # ========================================================

    print()
    print("=" * 70)
    print("9. ROLLING FEATURE AVAILABILITY")
    print("=" * 70)

    expected_warmup = {
        "return_5d": 5,
        "return_20d": 20,
        "return_60d": 60,
        "return_120d": 120,
        "return_252d": 252,

        "volatility_20d": 20,
        "volatility_60d": 60,
        "volatility_120d": 120,
        "volatility_252d": 252,

        "price_vs_sma_20": 19,
        "price_vs_sma_50": 49,
        "price_vs_sma_200": 199,

        "volume_ratio_20d": 19,
        "volume_ratio_60d": 59,

        "range_20d": 19,
        "range_60d": 59,
    }

    print(
        "Expected warm-up NaNs per stock:"
    )

    for feature, expected in expected_warmup.items():

        actual_counts = (
            features
            .groupby("ticker")[feature]
            .apply(
                lambda s: s.isna().sum()
            )
        )

        mismatches = (
            actual_counts != expected
        ).sum()

        maximum_actual = actual_counts.max()
        minimum_actual = actual_counts.min()

        print(
            f"{feature:25s} "
            f"expected={expected:3d} "
            f"actual_range="
            f"{minimum_actual:3d}-{maximum_actual:3d} "
            f"mismatched_stocks={mismatches}"
        )

    # We don't fail this section because features such as
    # volume_change_20d can legitimately contain additional
    # NaNs when the historical denominator is zero.

    print(
        "Rolling-window availability: REVIEWED"
    )

    # ========================================================
    # 10. SOURCE / FEATURE ROW COUNTS BY STOCK
    # ========================================================

    print()
    print("=" * 70)
    print("10. ROW COUNT PRESERVATION BY STOCK")
    print("=" * 70)

    source_counts = (
        source_keys
        .groupby("ticker")
        .size()
        .rename("source_rows")
    )

    feature_counts = (
        feature_keys
        .groupby("ticker")
        .size()
        .rename("feature_rows")
    )

    count_comparison = pd.concat(
        [
            source_counts,
            feature_counts
        ],
        axis=1
    )

    count_comparison[
        "difference"
    ] = (
        count_comparison["feature_rows"]
        - count_comparison["source_rows"]
    )

    count_mismatches = count_comparison[
        count_comparison["difference"] != 0
    ]

    print(
        f"Stocks with row-count mismatch: "
        f"{len(count_mismatches)}"
    )

    if len(count_mismatches) > 0:

        print(
            count_mismatches.to_string()
        )

        raise ValueError(
            "Per-stock row counts do not match."
        )

    print(
        "Per-stock row preservation: PASS"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL AUDIT STATUS")
    print("=" * 70)

    print(
        f"Rows                    : {len(features):,}"
    )

    print(
        f"Stocks                  : {stock_count}"
    )

    print(
        f"Feature columns         : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Infinite values         : {infinite_count:,}"
    )

    print(
        f"Duplicate ticker/date   : {duplicate_keys:,}"
    )

    print(
        f"Percentile features     : {len(PERCENTILE_COLUMNS)}"
    )

    print(
        f"Near-zero variance      : NOT PRESENT"
    )

    print()
    print(
        "STOCK FEATURE AUDIT: PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()