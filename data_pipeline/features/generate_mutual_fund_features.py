import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/clean/mutual_fund_nav_clean.parquet"
OUTPUT_FILE = "data/features/mutual_fund_features.parquet"

MIN_DATE = pd.Timestamp("2013-01-01")
MAX_DATE = pd.Timestamp("2026-10-22")

EXPECTED_FUNDS = 100
EXPECTED_FEATURE_COUNT = 27


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

BASE_FEATURES = [

    # --------------------------------------------------------
    # Momentum - 5
    # --------------------------------------------------------
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",

    # --------------------------------------------------------
    # Volatility - 4
    # --------------------------------------------------------
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",

    # --------------------------------------------------------
    # Drawdown - 3
    # --------------------------------------------------------
    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",

    # --------------------------------------------------------
    # Trend - 3
    # --------------------------------------------------------
    "nav_vs_sma_20",
    "nav_vs_sma_50",
    "nav_vs_sma_200",

    # --------------------------------------------------------
    # NAV-specific risk/activity - 5
    # --------------------------------------------------------
    "nav_range_20d",
    "nav_range_60d",
    "downside_volatility_60d",
    "upside_volatility_60d",
    "rolling_sharpe_60d",
]


PERCENTILE_FEATURES = [

    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",
    "volatility_60d",
    "drawdown_120d",
    "nav_range_60d",
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_calendar_dates(values):
    """
    Convert dates to timezone-naive normalized calendar dates.

    We preserve the actual calendar date and do not convert
    timezone-aware timestamps through UTC because that can
    shift dates by one day.
    """

    if isinstance(values, pd.Series):

        dt = pd.to_datetime(
            values,
            errors="coerce"
        )

        if isinstance(
            dt.dtype,
            pd.DatetimeTZDtype
        ):
            return (
                dt.dt
                .tz_localize(None)
                .dt
                .normalize()
            )

        return dt.dt.normalize()

    if isinstance(values, pd.DatetimeIndex):

        if values.tz is not None:
            return (
                values
                .tz_localize(None)
                .normalize()
            )

        return values.normalize()

    dt = pd.to_datetime(
        values,
        errors="coerce"
    )

    if isinstance(
        dt,
        pd.DatetimeIndex
    ):

        if dt.tz is not None:
            return (
                dt
                .tz_localize(None)
                .normalize()
            )

        return dt.normalize()

    return (
        pd.Series(dt)
        .dt
        .normalize()
    )


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source():

    print("=" * 70)
    print("MUTUAL FUND FEATURE GENERATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"\nLoaded rows: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "scheme_code",
        "date",
        "nav",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Standardize types
    # --------------------------------------------------------

    df["scheme_code"] = (
        df["scheme_code"]
        .astype(str)
    )

    df["date"] = normalize_calendar_dates(
        df["date"]
    )

    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Validate nulls
    # --------------------------------------------------------

    if df["date"].isna().any():

        raise ValueError(
            "Source contains null dates."
        )

    if df["scheme_code"].isna().any():

        raise ValueError(
            "Source contains null scheme codes."
        )

    if df["nav"].isna().any():

        raise ValueError(
            "Source contains null NAV values."
        )

    # --------------------------------------------------------
    # Validate NAV
    # --------------------------------------------------------

    if (df["nav"] <= 0).any():

        raise ValueError(
            "Source contains non-positive NAV values."
        )

    # --------------------------------------------------------
    # Restrict date range
    # --------------------------------------------------------

    df = df[
        (df["date"] >= MIN_DATE)
        &
        (df["date"] <= MAX_DATE)
    ].copy()

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = (
        df
        .duplicated(
            subset=[
                "scheme_code",
                "date"
            ]
        )
        .sum()
    )

    print(
        f"Duplicate scheme/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate scheme/date rows found "
            "in clean MF source."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "scheme_code",
                "date"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Fund count
    # --------------------------------------------------------

    fund_count = (
        df["scheme_code"]
        .nunique()
    )

    print(
        f"Funds: {fund_count}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    if fund_count != EXPECTED_FUNDS:

        raise ValueError(
            f"Expected {EXPECTED_FUNDS} funds, "
            f"but found {fund_count}."
        )

    return df


# ============================================================
# CALCULATE FEATURES FOR ONE FUND
# ============================================================

def calculate_fund_features(group):

    """
    Calculate all 20 non-percentile features for one fund.

    Cross-sectional percentile features are calculated only
    after all funds have been combined.
    """

    group = (
        group
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    ticker = str(
        group["scheme_code"].iloc[0]
    )

    # --------------------------------------------------------
    # Base output
    # --------------------------------------------------------

    features = pd.DataFrame(
        {
            "ticker": np.full(
                len(group),
                ticker,
                dtype=object
            ),

            "date": group[
                "date"
            ].to_numpy()
        }
    )

    nav = (
        group["nav"]
        .astype(float)
        .reset_index(drop=True)
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    features["return_5d"] = (
        nav
        .div(nav.shift(5))
        .sub(1.0)
    )

    features["return_20d"] = (
        nav
        .div(nav.shift(20))
        .sub(1.0)
    )

    features["return_60d"] = (
        nav
        .div(nav.shift(60))
        .sub(1.0)
    )

    features["return_120d"] = (
        nav
        .div(nav.shift(120))
        .sub(1.0)
    )

    features["return_252d"] = (
        nav
        .div(nav.shift(252))
        .sub(1.0)
    )

    # ========================================================
    # DAILY RETURNS
    # ========================================================

    daily_return = nav.pct_change()

    # ========================================================
    # VOLATILITY
    # ========================================================

    features["volatility_20d"] = (
        daily_return
        .rolling(
            20,
            min_periods=20
        )
        .std()
        .mul(np.sqrt(252))
    )

    features["volatility_60d"] = (
        daily_return
        .rolling(
            60,
            min_periods=60
        )
        .std()
        .mul(np.sqrt(252))
    )

    features["volatility_120d"] = (
        daily_return
        .rolling(
            120,
            min_periods=120
        )
        .std()
        .mul(np.sqrt(252))
    )

    features["volatility_252d"] = (
        daily_return
        .rolling(
            252,
            min_periods=252
        )
        .std()
        .mul(np.sqrt(252))
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    rolling_max_60 = (
        nav
        .rolling(
            60,
            min_periods=60
        )
        .max()
    )

    rolling_max_120 = (
        nav
        .rolling(
            120,
            min_periods=120
        )
        .max()
    )

    rolling_max_252 = (
        nav
        .rolling(
            252,
            min_periods=252
        )
        .max()
    )

    features["drawdown_60d"] = (
        nav
        .div(rolling_max_60)
        .sub(1.0)
    )

    features["drawdown_120d"] = (
        nav
        .div(rolling_max_120)
        .sub(1.0)
    )

    features["drawdown_252d"] = (
        nav
        .div(rolling_max_252)
        .sub(1.0)
    )

    # ========================================================
    # TREND
    # ========================================================

    sma_20 = (
        nav
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    sma_50 = (
        nav
        .rolling(
            50,
            min_periods=50
        )
        .mean()
    )

    sma_200 = (
        nav
        .rolling(
            200,
            min_periods=200
        )
        .mean()
    )

    features["nav_vs_sma_20"] = (
        nav
        .div(sma_20)
        .sub(1.0)
    )

    features["nav_vs_sma_50"] = (
        nav
        .div(sma_50)
        .sub(1.0)
    )

    features["nav_vs_sma_200"] = (
        nav
        .div(sma_200)
        .sub(1.0)
    )

    # ========================================================
    # NAV RANGE
    # ========================================================

    rolling_max_20 = (
        nav
        .rolling(
            20,
            min_periods=20
        )
        .max()
    )

    rolling_min_20 = (
        nav
        .rolling(
            20,
            min_periods=20
        )
        .min()
    )

    rolling_max_60 = (
        nav
        .rolling(
            60,
            min_periods=60
        )
        .max()
    )

    rolling_min_60 = (
        nav
        .rolling(
            60,
            min_periods=60
        )
        .min()
    )

    features["nav_range_20d"] = (
        rolling_max_20
        .div(rolling_min_20)
        .sub(1.0)
    )

    features["nav_range_60d"] = (
        rolling_max_60
        .div(rolling_min_60)
        .sub(1.0)
    )

    # ========================================================
    # DOWNSIDE / UPSIDE VOLATILITY
    # ========================================================

    # IMPORTANT:
    #
    # We do NOT use:
    #
    # daily_return.where(daily_return < 0)
    #
    # with min_periods=60.
    #
    # That would require 60 negative-return observations
    # inside every 60-day window and would therefore create
    # excessive NaNs.
    #
    # Instead, positive returns are represented as zero when
    # measuring downside volatility, and negative returns are
    # represented as zero when measuring upside volatility.

    downside_returns = (
        daily_return
        .clip(upper=0)
    )

    upside_returns = (
        daily_return
        .clip(lower=0)
    )

    features["downside_volatility_60d"] = (
        downside_returns
        .rolling(
            60,
            min_periods=60
        )
        .std()
        .mul(np.sqrt(252))
    )

    features["upside_volatility_60d"] = (
        upside_returns
        .rolling(
            60,
            min_periods=60
        )
        .std()
        .mul(np.sqrt(252))
    )

    # ========================================================
    # ROLLING SHARPE
    # ========================================================

    rolling_mean_return = (
        daily_return
        .rolling(
            60,
            min_periods=60
        )
        .mean()
    )

    rolling_std_return = (
        daily_return
        .rolling(
            60,
            min_periods=60
        )
        .std()
    )

    features["rolling_sharpe_60d"] = np.where(
        rolling_std_return > 0,

        (
            rolling_mean_return
            .div(rolling_std_return)
            .mul(np.sqrt(252))
        ),

        np.nan
    )

    return features


# ============================================================
# CROSS-SECTIONAL PERCENTILES
# ============================================================

def calculate_cross_sectional_percentiles(df):

    """
    Calculate percentile ranks across funds for each date.

    This MUST happen after all funds are combined.

    Ranking within individual funds would produce incorrect
    cross-sectional percentile values.
    """

    print(
        "\nCalculating cross-sectional "
        "percentile features..."
    )

    for feature in PERCENTILE_FEATURES:

        percentile_name = (
            f"{feature}_pct"
        )

        df[percentile_name] = (
            df
            .groupby("date")[feature]
            .rank(
                method="average",
                pct=True
            )
        )

        print(
            f"{percentile_name}: "
            f"min={df[percentile_name].min():.6f}, "
            f"max={df[percentile_name].max():.6f}, "
            f"unique={df[percentile_name].nunique():,}"
        )

    return df


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(
    df,
    source
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE VALIDATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Expected columns
    # --------------------------------------------------------

    expected_columns = (
        ["ticker", "date"]
        +
        BASE_FEATURES
        +
        [
            f"{feature}_pct"
            for feature in PERCENTILE_FEATURES
        ]
    )

    actual_feature_columns = [
        column
        for column in df.columns
        if column not in [
            "ticker",
            "date"
        ]
    ]

    print(
        f"\nFeature columns: "
        f"{len(actual_feature_columns)}"
    )

    if (
        len(actual_feature_columns)
        != EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"found "
            f"{len(actual_feature_columns)}."
        )

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing expected columns: "
            f"{missing_columns}"
        )

    extra_columns = [
        column
        for column in df.columns
        if column not in expected_columns
    ]

    if extra_columns:

        raise ValueError(
            f"Unexpected columns found: "
            f"{extra_columns}"
        )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"Source rows:   {len(source):,}"
    )

    print(
        f"Feature rows:  {len(df):,}"
    )

    if len(source) != len(df):

        raise ValueError(
            "Source and feature row counts "
            "do not match."
        )

    # --------------------------------------------------------
    # Fund count
    # --------------------------------------------------------

    fund_count = (
        df["ticker"]
        .nunique()
    )

    print(
        f"Funds: {fund_count}"
    )

    if fund_count != EXPECTED_FUNDS:

        raise ValueError(
            f"Expected {EXPECTED_FUNDS} funds, "
            f"found {fund_count}."
        )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_count = (
        df
        .duplicated(
            subset=[
                "ticker",
                "date"
            ]
        )
        .sum()
    )

    print(
        f"Duplicate ticker/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate ticker/date rows found."
        )

    # --------------------------------------------------------
    # Null keys
    # --------------------------------------------------------

    null_ticker = (
        df["ticker"]
        .isna()
        .sum()
    )

    null_date = (
        df["date"]
        .isna()
        .sum()
    )

    print(
        f"Null ticker: {null_ticker}"
    )

    print(
        f"Null date:   {null_date}"
    )

    if (
        null_ticker > 0
        or
        null_date > 0
    ):

        raise ValueError(
            "Null key values found."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in df.columns
        if column not in [
            "ticker",
            "date"
        ]
    ]

    infinite_count = (
        np.isinf(
            df[numeric_columns]
            .to_numpy(
                dtype=float
            )
        )
        .sum()
    )

    print(
        f"Infinite feature values: "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        raise ValueError(
            "Infinite feature values found."
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    min_date = df["date"].min()
    max_date = df["date"].max()

    print(
        f"Date range: "
        f"{min_date.date()} → "
        f"{max_date.date()}"
    )

    if min_date < MIN_DATE:

        raise ValueError(
            "Feature data contains dates "
            "before MIN_DATE."
        )

    if max_date > MAX_DATE:

        raise ValueError(
            "Feature data contains dates "
            "after MAX_DATE."
        )

    # --------------------------------------------------------
    # Percentile validation
    # --------------------------------------------------------

    print(
        "\nPercentile validation:"
    )

    for feature in PERCENTILE_FEATURES:

        percentile_column = (
            f"{feature}_pct"
        )

        valid_values = (
            df[percentile_column]
            .dropna()
        )

        if len(valid_values) == 0:

            raise ValueError(
                f"{percentile_column} contains "
                f"no valid values."
            )

        minimum = (
            valid_values.min()
        )

        maximum = (
            valid_values.max()
        )

        unique_count = (
            valid_values.nunique()
        )

        print(
            f"  {percentile_column}: "
            f"min={minimum:.6f}, "
            f"max={maximum:.6f}, "
            f"unique={unique_count:,}"
        )

        if (
            minimum < 0
            or
            maximum > 1
        ):

            raise ValueError(
                f"{percentile_column} "
                f"is outside [0, 1]."
            )

        if unique_count <= 1:

            raise ValueError(
                f"{percentile_column} has "
                f"no cross-sectional variation."
            )

    # --------------------------------------------------------
    # Temporal ordering
    # --------------------------------------------------------

    ordering_violations = 0

    for ticker, group in df.groupby(
        "ticker"
    ):

        dates = (
            group["date"]
            .to_numpy()
        )

        if not np.all(
            dates[1:] >= dates[:-1]
        ):

            ordering_violations += 1

    print(
        f"Temporal ordering violations: "
        f"{ordering_violations}"
    )

    if ordering_violations > 0:

        raise ValueError(
            "Temporal ordering violations found."
        )

    # --------------------------------------------------------
    # Source key preservation
    # --------------------------------------------------------

    source_keys = set(
        zip(
            source[
                "scheme_code"
            ].astype(str),

            source[
                "date"
            ]
        )
    )

    feature_keys = set(
        zip(
            df[
                "ticker"
            ].astype(str),

            df[
                "date"
            ]
        )
    )

    source_missing = (
        source_keys - feature_keys
    )

    feature_extra = (
        feature_keys - source_keys
    )

    print(
        f"Source keys missing from features: "
        f"{len(source_missing)}"
    )

    print(
        f"Feature keys absent from source: "
        f"{len(feature_extra)}"
    )

    if (
        source_missing
        or
        feature_extra
    ):

        raise ValueError(
            "Source/feature key preservation "
            "failed."
        )

    print(
        "\nFEATURE GENERATION VALIDATION: PASS"
    )


# ============================================================
# NAN SUMMARY
# ============================================================

def print_nan_summary(df):

    print(
        "\n" + "=" * 70
    )

    print(
        "NaN SUMMARY"
    )

    print(
        "=" * 70
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "ticker",
            "date"
        ]
    ]

    nan_summary = (
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for feature, count in (
        nan_summary.items()
    ):

        if count > 0:

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{feature:35s} "
                f"{count:8,} "
                f"({percentage:6.3f}%)"
            )

    print(
        "\nNaNs are expected during "
        "rolling-window warm-up periods "
        "and are NOT filled."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_directory = (
        os.path.dirname(
            OUTPUT_FILE
        )
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Load source
    # --------------------------------------------------------

    source = load_source()

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING PER-FUND FEATURES"
    )

    print(
        "=" * 70
    )

    feature_frames = []

    fund_groups = (
        source
        .groupby(
            "scheme_code",
            sort=False
        )
    )

    total_funds = (
        source[
            "scheme_code"
        ]
        .nunique()
    )

    for index, (
        scheme_code,
        group
    ) in enumerate(
        fund_groups,
        start=1
    ):

        print(
            f"[{index:3d}/{total_funds}] "
            f"Processing fund "
            f"{scheme_code}"
        )

        fund_features = (
            calculate_fund_features(
                group
            )
        )

        feature_frames.append(
            fund_features
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print(
        "\nCombining fund features..."
    )

    features = pd.concat(
        feature_frames,
        ignore_index=True
    )

    features = (
        features
        .sort_values(
            [
                "ticker",
                "date"
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Combined rows: "
        f"{len(features):,}"
    )

    # --------------------------------------------------------
    # Cross-sectional percentiles
    # --------------------------------------------------------

    features = (
        calculate_cross_sectional_percentiles(
            features
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_features(
        features,
        source
    )

    # --------------------------------------------------------
    # NaN summary
    # --------------------------------------------------------

    print_nan_summary(
        features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving features..."
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MUTUAL FUND FEATURE GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Funds: "
        f"{features['ticker'].nunique()}"
    )

    print(
        f"Feature count: "
        f"{len(features.columns) - 2}"
    )

    print(
        f"Columns: "
        f"{len(features.columns)}"
    )

    print(
        f"Date range: "
        f"{features['date'].min().date()} → "
        f"{features['date'].max().date()}"
    )

    print(
        "\nExpected structure:"
    )

    print(
        "ticker + date + 27 features = 29 columns"
    )

    print(
        "\nNext step: run the dedicated "
        "MF feature audit."
    )


if __name__ == "__main__":
    main()
