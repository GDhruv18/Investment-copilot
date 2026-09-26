from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# Clean ETF files are stored inside data/clean/etfs/
CLEAN_DIR = PROJECT_ROOT / "data" / "clean" / "etfs"

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"

OUTPUT_FILE = OUTPUT_DIR / "etf_features.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_ETFS = 41

EXPECTED_START_DATE = pd.Timestamp("2014-01-01")
EXPECTED_END_DATE = pd.Timestamp("2026-09-18")

REQUIRED_RAW_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_calendar_dates(values):
    """
    Normalize timestamps while preserving the original
    calendar date.

    We deliberately do not convert timestamps to UTC because
    timezone conversion can shift a calendar date.
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
# SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator):
    """
    Convert zero denominators to NaN before division.
    """

    denominator = denominator.replace(
        0,
        np.nan
    )

    return numerator / denominator


# ============================================================
# ETF FEATURE CALCULATION
# ============================================================

def calculate_etf_features(df, ticker):
    """
    Calculate ETF-level historical features.

    Cross-sectional percentile features are NOT calculated here.
    They require the complete ETF universe and are calculated
    only after all 41 ETFs have been combined.
    """

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = normalize_calendar_dates(
        df.index
    )

    if df["date"].isna().any():

        invalid_count = df["date"].isna().sum()

        raise ValueError(
            f"{ticker}: {invalid_count} invalid dates."
        )

    df = (
        df
        .reset_index(drop=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # DUPLICATE DATE CHECK
    # --------------------------------------------------------

    if df["date"].duplicated().any():

        duplicate_dates = (
            df.loc[
                df["date"].duplicated(keep=False),
                "date"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{ticker}: duplicate dates found: "
            f"{duplicate_dates[:10]}"
        )

    # --------------------------------------------------------
    # RAW SERIES
    # --------------------------------------------------------

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    adj_close = pd.to_numeric(
        df["Adj Close"],
        errors="coerce"
    )

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce"
    )

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # FEATURE DATAFRAME
    # --------------------------------------------------------

    features = pd.DataFrame(index=df.index)

    features["ticker"] = ticker
    features["date"] = df["date"]

    # ========================================================
    # 1. MOMENTUM
    # ========================================================

    features["return_5d"] = (
        adj_close / adj_close.shift(5) - 1
    )

    features["return_20d"] = (
        adj_close / adj_close.shift(20) - 1
    )

    features["return_60d"] = (
        adj_close / adj_close.shift(60) - 1
    )

    features["return_120d"] = (
        adj_close / adj_close.shift(120) - 1
    )

    features["return_252d"] = (
        adj_close / adj_close.shift(252) - 1
    )

    # ========================================================
    # 2. DAILY RETURN
    # ========================================================

    daily_return = adj_close.pct_change()

    # ========================================================
    # 3. VOLATILITY
    # ========================================================

    features["volatility_20d"] = (
        daily_return
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    features["volatility_60d"] = (
        daily_return
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    features["volatility_120d"] = (
        daily_return
        .rolling(120)
        .std()
        * np.sqrt(252)
    )

    features["volatility_252d"] = (
        daily_return
        .rolling(252)
        .std()
        * np.sqrt(252)
    )

    # ========================================================
    # 4. DRAWDOWN
    # ========================================================

    rolling_max_60 = (
        adj_close
        .rolling(60)
        .max()
    )

    rolling_max_120 = (
        adj_close
        .rolling(120)
        .max()
    )

    rolling_max_252 = (
        adj_close
        .rolling(252)
        .max()
    )

    features["drawdown_60d"] = (
        safe_divide(
            adj_close,
            rolling_max_60
        ) - 1
    )

    features["drawdown_120d"] = (
        safe_divide(
            adj_close,
            rolling_max_120
        ) - 1
    )

    features["drawdown_252d"] = (
        safe_divide(
            adj_close,
            rolling_max_252
        ) - 1
    )

    # ========================================================
    # 5. TREND
    # ========================================================

    sma_20 = (
        adj_close
        .rolling(20)
        .mean()
    )

    sma_50 = (
        adj_close
        .rolling(50)
        .mean()
    )

    sma_200 = (
        adj_close
        .rolling(200)
        .mean()
    )

    features["price_vs_sma_20"] = (
        safe_divide(
            adj_close,
            sma_20
        ) - 1
    )

    features["price_vs_sma_50"] = (
        safe_divide(
            adj_close,
            sma_50
        ) - 1
    )

    features["price_vs_sma_200"] = (
        safe_divide(
            adj_close,
            sma_200
        ) - 1
    )

    # ========================================================
    # 6. VOLUME
    # ========================================================

    volume_mean_20 = (
        volume
        .rolling(20)
        .mean()
    )

    volume_mean_60 = (
        volume
        .rolling(60)
        .mean()
    )

    features["volume_ratio_20d"] = (
        safe_divide(
            volume,
            volume_mean_20
        )
    )

    features["volume_ratio_60d"] = (
        safe_divide(
            volume,
            volume_mean_60
        )
    )

    # --------------------------------------------------------
    # Volume change
    # --------------------------------------------------------

    volume_20d_ago = (
        volume
        .shift(20)
        .replace(
            0,
            np.nan
        )
    )

    features["volume_change_20d"] = (
        volume / volume_20d_ago - 1
    )

    # ========================================================
    # 7. TRADING RANGE
    # ========================================================

    safe_close = (
        close
        .replace(
            0,
            np.nan
        )
    )

    daily_range = (
        (high - low)
        / safe_close
    )

    features["range_20d"] = (
        daily_range
        .rolling(20)
        .mean()
    )

    features["range_60d"] = (
        daily_range
        .rolling(60)
        .mean()
    )

    return features


# ============================================================
# CROSS-SECTIONAL FEATURES
# ============================================================

def calculate_cross_sectional_features(features):
    """
    Calculate cross-sectional percentile ranks across all ETFs
    for each observation date.

    This must happen AFTER all 41 ETFs are combined.
    """

    features = features.copy()

    features["return_20d_pct"] = (
        features
        .groupby("date")["return_20d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["return_60d_pct"] = (
        features
        .groupby("date")["return_60d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["return_120d_pct"] = (
        features
        .groupby("date")["return_120d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["return_252d_pct"] = (
        features
        .groupby("date")["return_252d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["volatility_60d_pct"] = (
        features
        .groupby("date")["volatility_60d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["drawdown_120d_pct"] = (
        features
        .groupby("date")["drawdown_120d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    features["volume_ratio_20d_pct"] = (
        features
        .groupby("date")["volume_ratio_20d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ETF FEATURE GENERATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check clean ETF directory
    # --------------------------------------------------------

    if not CLEAN_DIR.exists():

        raise FileNotFoundError(
            f"Clean ETF directory does not exist:\n"
            f"{CLEAN_DIR}"
        )

    print(
        f"Clean ETF directory:\n"
        f"{CLEAN_DIR}"
    )

    # --------------------------------------------------------
    # Find clean ETF files
    # --------------------------------------------------------

    parquet_files = sorted(
        CLEAN_DIR.glob("*.parquet")
    )

    print()
    print(
        f"Clean ETF files found: "
        f"{len(parquet_files)}"
    )

    if len(parquet_files) != EXPECTED_ETFS:

        print()
        print(
            "Files found:"
        )

        for file_path in parquet_files:
            print(
                f"  {file_path.name}"
            )

        raise ValueError(
            f"Expected {EXPECTED_ETFS} clean ETF files, "
            f"found {len(parquet_files)}.\n"
            f"Expected directory:\n"
            f"{CLEAN_DIR}"
        )

    # --------------------------------------------------------
    # Extract tickers
    # --------------------------------------------------------

    tickers = [
        file_path.stem
        for file_path in parquet_files
    ]

    # --------------------------------------------------------
    # DYNAMIC check
    # --------------------------------------------------------

    if "DYNAMIC_NS" in tickers:

        raise ValueError(
            "DYNAMIC_NS.parquet is present in the clean ETF "
            "directory. DYNAMIC was removed from the ETF "
            "universe and must not be included."
        )

    # ========================================================
    # STEP 1
    # Individual ETF features
    # ========================================================

    print()
    print(
        "STEP 1: Calculating ETF-level features..."
    )

    all_features = []

    for i, file_path in enumerate(
        parquet_files,
        start=1
    ):

        ticker = file_path.stem

        print(
            f"[{i:3d}/{len(parquet_files)}] "
            f"{ticker}",
            end=": "
        )

        df = pd.read_parquet(
            file_path
        )

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in REQUIRED_RAW_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{ticker}: missing required columns: "
                f"{missing_columns}"
            )

        etf_features = calculate_etf_features(
            df,
            ticker
        )

        all_features.append(
            etf_features
        )

        print(
            f"{len(etf_features):,} rows"
        )

    # ========================================================
    # STEP 2
    # Combine complete ETF universe
    # ========================================================

    print()
    print(
        "STEP 2: Combining all ETFs..."
    )

    features = pd.concat(
        all_features,
        ignore_index=True
    )

    features = (
        features
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Combined rows: "
        f"{len(features):,}"
    )

    print(
        f"ETFs: "
        f"{features['ticker'].nunique()}"
    )

    # ========================================================
    # STEP 3
    # Cross-sectional percentile features
    # ========================================================

    print()
    print(
        "STEP 3: Calculating cross-sectional "
        "percentile features..."
    )

    features = calculate_cross_sectional_features(
        features
    )

    # ========================================================
    # STEP 4
    # FINAL SCHEMA
    # ========================================================

    expected_columns = [
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

        # Range
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

    if list(features.columns) != expected_columns:

        raise ValueError(
            "Final ETF feature columns do not match "
            "the expected schema.\n\n"
            f"Expected:\n{expected_columns}\n\n"
            f"Actual:\n{list(features.columns)}"
        )

    # ========================================================
    # STEP 5
    # FINAL VALIDATION
    # ========================================================

    print()
    print(
        "STEP 4: Final validation..."
    )

    # --------------------------------------------------------
    # ETF count
    # --------------------------------------------------------

    etf_count = (
        features["ticker"]
        .nunique()
    )

    if etf_count != EXPECTED_ETFS:

        raise ValueError(
            f"Expected {EXPECTED_ETFS} ETFs, "
            f"found {etf_count}."
        )

    # --------------------------------------------------------
    # DYNAMIC
    # --------------------------------------------------------

    if (
        features["ticker"]
        .eq("DYNAMIC_NS")
        .any()
    ):

        raise ValueError(
            "DYNAMIC_NS found in ETF feature dataset."
        )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_keys = (
        features
        .duplicated(
            ["ticker", "date"]
        )
        .sum()
    )

    if duplicate_keys != 0:

        raise ValueError(
            f"Duplicate ticker/date keys: "
            f"{duplicate_keys}"
        )

    print(
        f"Duplicate ticker/date keys: "
        f"{duplicate_keys}"
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in features.columns
        if column not in ["ticker", "date"]
    ]

    infinite_count = np.isinf(
        features[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).sum()

    print(
        f"Infinite values: "
        f"{infinite_count:,}"
    )

    if infinite_count != 0:

        raise ValueError(
            f"Infinite values found: "
            f"{infinite_count:,}"
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    min_date = features[
        "date"
    ].min()

    max_date = features[
        "date"
    ].max()

    print(
        f"Min date: "
        f"{min_date.date()}"
    )

    print(
        f"Max date: "
        f"{max_date.date()}"
    )

    rows_before_start = (
        features["date"]
        < EXPECTED_START_DATE
    ).sum()

    rows_after_end = (
        features["date"]
        > EXPECTED_END_DATE
    ).sum()

    print(
        f"Rows before expected start: "
        f"{rows_before_start:,}"
    )

    print(
        f"Rows after expected end: "
        f"{rows_after_end:,}"
    )

    if rows_before_start != 0:

        raise ValueError(
            "ETF feature rows exist before "
            f"{EXPECTED_START_DATE.date()}."
        )

    if rows_after_end != 0:

        raise ValueError(
            "ETF feature rows exist after "
            f"{EXPECTED_END_DATE.date()}."
        )

    # ========================================================
    # PERCENTILE VALIDATION
    # ========================================================

    percentile_columns = [
        "return_20d_pct",
        "return_60d_pct",
        "return_120d_pct",
        "return_252d_pct",
        "volatility_60d_pct",
        "drawdown_120d_pct",
        "volume_ratio_20d_pct",
    ]

    print()
    print(
        "Cross-sectional percentile validation:"
    )

    for column in percentile_columns:

        valid = (
            features[column]
            .dropna()
        )

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

        if minimum < 0 or maximum > 1:

            raise ValueError(
                f"{column}: percentile outside [0,1]."
            )

        if unique_count <= 1:

            raise ValueError(
                f"{column}: percentile feature is constant."
            )

    # ========================================================
    # NaN SUMMARY
    # ========================================================

    print()
    print(
        "NaN summary:"
    )

    nan_summary = (
        features[numeric_columns]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for column, count in nan_summary.items():

        if count > 0:

            percentage = (
                count
                / len(features)
                * 100
            )

            print(
                f"{column:25s} "
                f"{count:8,} "
                f"({percentage:.3f}%)"
            )

    # ========================================================
    # SAVE
    # ========================================================

    features.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("ETF FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Rows:          {len(features):,}"
    )

    print(
        f"ETFs:          {etf_count}"
    )

    print(
        f"Features:      {len(numeric_columns)}"
    )

    print(
        f"Total columns: {len(features.columns)}"
    )

    print(
        f"Output:        {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()