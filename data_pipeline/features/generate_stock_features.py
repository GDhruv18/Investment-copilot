# data_pipeline/features/generate_stock_features.py

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"

OUTPUT_FILE = OUTPUT_DIR / "stock_features.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_STOCKS = 100
START_DATE = pd.Timestamp("2014-01-01")

REQUIRED_RAW_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_calendar_dates(values):
    """
    Normalize dates while preserving the actual calendar date.

    We intentionally do NOT convert to UTC because that can shift
    calendar dates depending on timezone representation.
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


def safe_divide(numerator, denominator):
    """
    Division that converts zero denominators into NaN.
    """

    denominator = denominator.replace(0, np.nan)

    return numerator / denominator


# ============================================================
# STOCK FEATURE CALCULATION
# ============================================================

def calculate_stock_features(df, ticker):
    """
    Calculate all stock-level features for one stock.

    Cross-sectional percentile features are intentionally NOT
    calculated here. They require the complete universe.
    """

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = normalize_calendar_dates(df.index)

    if df["date"].isna().any():
        raise ValueError(
            f"{ticker}: invalid dates found in raw data."
        )

    df = df.reset_index(drop=True)

    df = df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------------
    # DUPLICATE DATE CHECK
    # --------------------------------------------------------

    if df["date"].duplicated().any():
        duplicate_dates = (
            df.loc[df["date"].duplicated(keep=False), "date"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{ticker}: duplicate dates found: {duplicate_dates[:10]}"
        )

    # --------------------------------------------------------
    # RAW SERIES
    # --------------------------------------------------------

    close = pd.to_numeric(df["Close"], errors="coerce")
    adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    # --------------------------------------------------------
    # FEATURE CONTAINER
    # --------------------------------------------------------

    features = pd.DataFrame(index=df.index)

    features["ticker"] = ticker
    features["date"] = df["date"]

    # ========================================================
    # MOMENTUM
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
    # DAILY RETURN
    # ========================================================

    daily_return = adj_close.pct_change()

    # ========================================================
    # VOLATILITY
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
    # DRAWDOWN
    # ========================================================

    rolling_max_60 = adj_close.rolling(60).max()
    rolling_max_120 = adj_close.rolling(120).max()
    rolling_max_252 = adj_close.rolling(252).max()

    features["drawdown_60d"] = safe_divide(
        adj_close,
        rolling_max_60
    ) - 1

    features["drawdown_120d"] = safe_divide(
        adj_close,
        rolling_max_120
    ) - 1

    features["drawdown_252d"] = safe_divide(
        adj_close,
        rolling_max_252
    ) - 1

    # ========================================================
    # TREND
    # ========================================================

    sma_20 = adj_close.rolling(20).mean()
    sma_50 = adj_close.rolling(50).mean()
    sma_200 = adj_close.rolling(200).mean()

    features["price_vs_sma_20"] = safe_divide(
        adj_close,
        sma_20
    ) - 1

    features["price_vs_sma_50"] = safe_divide(
        adj_close,
        sma_50
    ) - 1

    features["price_vs_sma_200"] = safe_divide(
        adj_close,
        sma_200
    ) - 1

    # ========================================================
    # VOLUME
    # ========================================================

    volume_mean_20 = volume.rolling(20).mean()
    volume_mean_60 = volume.rolling(60).mean()

    features["volume_ratio_20d"] = safe_divide(
        volume,
        volume_mean_20
    )

    features["volume_ratio_60d"] = safe_divide(
        volume,
        volume_mean_60
    )

    # --------------------------------------------------------
    # Volume change over 20 trading days
    #
    # If volume 20 days ago is zero, the percentage change is
    # mathematically undefined, so retain NaN.
    # --------------------------------------------------------

    volume_20d_ago = volume.shift(20).replace(0, np.nan)

    features["volume_change_20d"] = (
        volume / volume_20d_ago - 1
    )

    # ========================================================
    # TRADING RANGE
    # ========================================================

    safe_close = close.replace(0, np.nan)

    daily_range = (
        (high - low) / safe_close
    )

    features["range_20d"] = (
        daily_range.rolling(20).mean()
    )

    features["range_60d"] = (
        daily_range.rolling(60).mean()
    )

    return features


# ============================================================
# CROSS-SECTIONAL FEATURES
# ============================================================

def calculate_cross_sectional_features(features):
    """
    Calculate cross-sectional percentile ranks across the
    complete stock universe for each date.

    IMPORTANT:
    This must happen AFTER all stocks are concatenated.
    """

    features = features.copy()

    # --------------------------------------------------------
    # Return percentiles
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Volatility percentile
    # --------------------------------------------------------

    features["volatility_60d_pct"] = (
        features
        .groupby("date")["volatility_60d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    # --------------------------------------------------------
    # Drawdown percentile
    # --------------------------------------------------------

    features["drawdown_120d_pct"] = (
        features
        .groupby("date")["drawdown_120d"]
        .rank(
            pct=True,
            method="average"
        )
    )

    # --------------------------------------------------------
    # Volume percentile
    # --------------------------------------------------------

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
    print("STOCK FEATURE GENERATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    parquet_files = sorted(
        RAW_DIR.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {RAW_DIR}"
        )

    print(f"Raw stock files found: {len(parquet_files)}")

    if len(parquet_files) != EXPECTED_STOCKS:
        raise ValueError(
            f"Expected {EXPECTED_STOCKS} stocks, "
            f"found {len(parquet_files)}."
        )

    all_features = []

    # ========================================================
    # STEP 1
    # Calculate stock-level features individually
    # ========================================================

    print()
    print("STEP 1: Calculating stock-level features...")

    for i, file_path in enumerate(parquet_files, start=1):

        ticker = file_path.stem

        df = pd.read_parquet(file_path)

        # ----------------------------------------------------
        # Validate required raw columns
        # ----------------------------------------------------

        missing_columns = [
            col
            for col in REQUIRED_RAW_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{ticker}: missing raw columns: "
                f"{missing_columns}"
            )

        stock_features = calculate_stock_features(
            df,
            ticker
        )

        all_features.append(
            stock_features
        )

        print(
            f"[{i:3d}/{len(parquet_files)}] "
            f"{ticker}: {len(stock_features):,} rows"
        )

    # ========================================================
    # STEP 2
    # Combine complete stock universe
    # ========================================================

    print()
    print("STEP 2: Combining all stocks...")

    features = pd.concat(
        all_features,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Basic ordering
    # --------------------------------------------------------

    features = features.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    print(
        f"Combined rows: {len(features):,}"
    )

    print(
        f"Stocks: {features['ticker'].nunique()}"
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
    # Final ordering
    # ========================================================

    features = features.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # ========================================================
    # EXPECTED COLUMNS
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

    if list(features.columns) != expected_columns:
        raise ValueError(
            "Final feature columns do not match expected schema.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {list(features.columns)}"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("STEP 4: Final validation...")

    # Row count
    if len(features) != 293430:
        raise ValueError(
            f"Unexpected row count: {len(features)}"
        )

    # Stock count
    stock_count = features["ticker"].nunique()

    if stock_count != EXPECTED_STOCKS:
        raise ValueError(
            f"Expected {EXPECTED_STOCKS} stocks, "
            f"found {stock_count}"
        )

    # Duplicate keys
    duplicate_keys = features.duplicated(
        ["ticker", "date"]
    ).sum()

    if duplicate_keys != 0:
        raise ValueError(
            f"Duplicate ticker/date keys: "
            f"{duplicate_keys}"
        )

    # Infinite values
    numeric_columns = [
        col
        for col in features.columns
        if col not in ["ticker", "date"]
    ]

    infinite_count = np.isinf(
        features[numeric_columns].to_numpy()
    ).sum()

    if infinite_count != 0:
        raise ValueError(
            f"Infinite values found: {infinite_count}"
        )

    # Date checks
    min_date = features["date"].min()
    max_date = features["date"].max()

    print(f"Min date: {min_date.date()}")
    print(f"Max date: {max_date.date()}")

    if (features["date"] < START_DATE).any():
        print(
            "WARNING: Rows exist before "
            f"{START_DATE.date()}."
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
    print("Cross-sectional percentile validation:")

    for col in percentile_columns:

        valid = features[col].dropna()

        if len(valid) == 0:
            raise ValueError(
                f"{col}: all values are NaN."
            )

        minimum = valid.min()
        maximum = valid.max()
        unique_count = valid.nunique()

        print(
            f"{col:25s} "
            f"min={minimum:.6f} "
            f"max={maximum:.6f} "
            f"unique={unique_count:,}"
        )

        if minimum < 0 or maximum > 1:
            raise ValueError(
                f"{col}: percentile outside [0,1]."
            )

        if unique_count <= 1:
            raise ValueError(
                f"{col}: percentile feature is constant."
            )

    # ========================================================
    # NaN SUMMARY
    # ========================================================

    print()
    print("NaN summary:")

    nan_summary = (
        features[numeric_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    for column, count in nan_summary.items():

        if count > 0:

            percentage = (
                count / len(features) * 100
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
    print("STOCK FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print(f"Rows:          {len(features):,}")
    print(f"Stocks:        {stock_count}")
    print(f"Features:      {len(numeric_columns)}")
    print(f"Total columns: {len(features.columns)}")
    print(f"Output:        {OUTPUT_FILE}")


if __name__ == "__main__":
    main()