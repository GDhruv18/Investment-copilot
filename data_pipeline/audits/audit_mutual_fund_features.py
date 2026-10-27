import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = "data/clean/mutual_fund_nav_clean.parquet"
FEATURE_FILE = "data/features/mutual_fund_features.parquet"

MIN_DATE = pd.Timestamp("2013-01-01")
MAX_DATE = pd.Timestamp("2026-10-22")

EXPECTED_FUNDS = 100
EXPECTED_FEATURE_COUNT = 27


# ============================================================
# EXPECTED FEATURES
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

    Calendar dates are preserved without converting timezone-aware
    timestamps through UTC.
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
    print("MUTUAL FUND FEATURE AUDIT")
    print("=" * 70)

    if not os.path.exists(SOURCE_FILE):

        raise FileNotFoundError(
            f"\nSource file not found:\n{SOURCE_FILE}"
        )

    if not os.path.exists(FEATURE_FILE):

        raise FileNotFoundError(
            f"\nFeature file not found:\n{FEATURE_FILE}"
        )

    source = pd.read_parquet(
        SOURCE_FILE
    )

    features = pd.read_parquet(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Required source columns
    # --------------------------------------------------------

    required_source_columns = [
        "scheme_code",
        "date",
        "nav",
    ]

    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in source.columns
    ]

    if missing_source_columns:

        raise ValueError(
            "Missing source columns: "
            f"{missing_source_columns}"
        )

    # --------------------------------------------------------
    # Standardize source
    # --------------------------------------------------------

    source["scheme_code"] = (
        source["scheme_code"]
        .astype(str)
    )

    source["date"] = normalize_calendar_dates(
        source["date"]
    )

    source["nav"] = pd.to_numeric(
        source["nav"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Standardize features
    # --------------------------------------------------------

    if "ticker" not in features.columns:

        raise ValueError(
            "Feature file is missing 'ticker'."
        )

    if "date" not in features.columns:

        raise ValueError(
            "Feature file is missing 'date'."
        )

    features["ticker"] = (
        features["ticker"]
        .astype(str)
    )

    features["date"] = normalize_calendar_dates(
        features["date"]
    )

    return source, features


# ============================================================
# 1. STRUCTURE AUDIT
# ============================================================

def audit_structure(features):

    print("\n" + "=" * 70)
    print("1. STRUCTURE AUDIT")
    print("=" * 70)

    feature_columns = [
        column
        for column in features.columns
        if column not in [
            "ticker",
            "date",
        ]
    ]

    print(
        f"\nFeature columns: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:

        print(
            f"FAIL: Expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"found {len(feature_columns)}."
        )

        return False

    print(
        f"PASS: {EXPECTED_FEATURE_COUNT} "
        f"features present."
    )

    if len(features.columns) != 29:

        print(
            f"FAIL: Expected 29 total columns, "
            f"found {len(features.columns)}."
        )

        return False

    print(
        "PASS: 29 total columns."
    )

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

    missing_columns = [
        column
        for column in expected_columns
        if column not in features.columns
    ]

    extra_columns = [
        column
        for column in features.columns
        if column not in expected_columns
    ]

    if missing_columns:

        print(
            "FAIL: Missing expected features: "
            f"{missing_columns}"
        )

        return False

    print(
        "PASS: All expected features present."
    )

    if extra_columns:

        print(
            "FAIL: Unexpected features: "
            f"{extra_columns}"
        )

        return False

    print(
        "PASS: No unexpected features."
    )

    if (
        "ticker" in features.columns
        and
        "date" in features.columns
    ):

        print(
            "PASS: Required key columns present."
        )

    else:

        print(
            "FAIL: Required key columns missing."
        )

        return False

    return True


# ============================================================
# 2. FUND UNIVERSE AUDIT
# ============================================================

def audit_fund_universe(
    source,
    features
):

    print("\n" + "=" * 70)
    print("2. FUND UNIVERSE AUDIT")
    print("=" * 70)

    source_funds = set(
        source["scheme_code"]
        .astype(str)
        .unique()
    )

    feature_funds = set(
        features["ticker"]
        .astype(str)
        .unique()
    )

    print(
        f"Source funds:   "
        f"{len(source_funds)}"
    )

    print(
        f"Feature funds:  "
        f"{len(feature_funds)}"
    )

    if len(source_funds) != EXPECTED_FUNDS:

        print(
            f"FAIL: Source contains "
            f"{len(source_funds)} funds."
        )

        return False

    print(
        f"PASS: Source contains "
        f"{EXPECTED_FUNDS} funds."
    )

    if len(feature_funds) != EXPECTED_FUNDS:

        print(
            f"FAIL: Features contain "
            f"{len(feature_funds)} funds."
        )

        return False

    print(
        f"PASS: Features contain "
        f"{EXPECTED_FUNDS} funds."
    )

    missing = source_funds - feature_funds
    extra = feature_funds - source_funds

    print(
        f"Funds missing from features: "
        f"{len(missing)}"
    )

    print(
        f"Extra funds in features: "
        f"{len(extra)}"
    )

    if missing or extra:

        print(
            "FAIL: Fund universe mismatch."
        )

        return False

    print(
        "PASS: Fund universe preserved."
    )

    return True


# ============================================================
# 3. KEY INTEGRITY AUDIT
# ============================================================

def audit_keys(
    source,
    features
):

    print("\n" + "=" * 70)
    print("3. KEY INTEGRITY AUDIT")
    print("=" * 70)

    source_duplicate_count = (
        source
        .duplicated(
            subset=[
                "scheme_code",
                "date",
            ]
        )
        .sum()
    )

    feature_duplicate_count = (
        features
        .duplicated(
            subset=[
                "ticker",
                "date",
            ]
        )
        .sum()
    )

    print(
        f"Source duplicate scheme/date: "
        f"{source_duplicate_count}"
    )

    if source_duplicate_count != 0:

        print(
            "FAIL: Source contains duplicate keys."
        )

        return False

    print(
        "PASS: Source has no duplicate keys."
    )

    print(
        f"Feature duplicate ticker/date: "
        f"{feature_duplicate_count}"
    )

    if feature_duplicate_count != 0:

        print(
            "FAIL: Features contain duplicate keys."
        )

        return False

    print(
        "PASS: Features have no duplicate keys."
    )

    source_null_scheme = (
        source["scheme_code"]
        .isna()
        .sum()
    )

    source_null_date = (
        source["date"]
        .isna()
        .sum()
    )

    feature_null_ticker = (
        features["ticker"]
        .isna()
        .sum()
    )

    feature_null_date = (
        features["date"]
        .isna()
        .sum()
    )

    print(
        f"Source null scheme_code: "
        f"{source_null_scheme}"
    )

    print(
        f"Source null date: "
        f"{source_null_date}"
    )

    print(
        f"Feature null ticker: "
        f"{feature_null_ticker}"
    )

    print(
        f"Feature null date: "
        f"{feature_null_date}"
    )

    if any(
        value > 0
        for value in [
            source_null_scheme,
            source_null_date,
            feature_null_ticker,
            feature_null_date,
        ]
    ):

        print(
            "FAIL: Null key values found."
        )

        return False

    print(
        "PASS: No null key values."
    )

    return True


# ============================================================
# 4. ROW PRESERVATION AUDIT
# ============================================================

def audit_row_preservation(
    source,
    features
):

    print("\n" + "=" * 70)
    print("4. ROW PRESERVATION AUDIT")
    print("=" * 70)

    source_rows = len(source)
    feature_rows = len(features)

    print(
        f"Source rows:   {source_rows:,}"
    )

    print(
        f"Feature rows:  {feature_rows:,}"
    )

    if source_rows != feature_rows:

        print(
            "FAIL: Row counts do not match."
        )

        return False

    print(
        "PASS: Row counts match."
    )

    source_counts = (
        source
        .groupby("scheme_code")
        .size()
    )

    feature_counts = (
        features
        .groupby("ticker")
        .size()
    )

    mismatches = 0

    for ticker in source_counts.index:

        source_count = source_counts.get(
            ticker,
            0
        )

        feature_count = feature_counts.get(
            ticker,
            0
        )

        if source_count != feature_count:

            mismatches += 1

    print(
        f"Per-fund row count mismatches: "
        f"{mismatches}"
    )

    if mismatches > 0:

        print(
            "FAIL: Some funds do not preserve "
            "their source row count."
        )

        return False

    print(
        "PASS: Every fund preserves its "
        "source row count."
    )

    return True


# ============================================================
# 5. SOURCE/FEATURE KEY PRESERVATION
# ============================================================

def audit_key_preservation(
    source,
    features
):

    print("\n" + "=" * 70)
    print("5. SOURCE/FEATURE KEY PRESERVATION")
    print("=" * 70)

    source_keys = set(
        zip(
            source["scheme_code"].astype(str),
            source["date"],
        )
    )

    feature_keys = set(
        zip(
            features["ticker"].astype(str),
            features["date"],
        )
    )

    source_missing = (
        source_keys - feature_keys
    )

    feature_extra = (
        feature_keys - source_keys
    )

    print(
        f"Source keys: "
        f"{len(source_keys):,}"
    )

    print(
        f"Feature keys: "
        f"{len(feature_keys):,}"
    )

    print(
        f"Source keys missing from features: "
        f"{len(source_missing):,}"
    )

    print(
        f"Feature keys absent from source: "
        f"{len(feature_extra):,}"
    )

    if source_missing or feature_extra:

        print(
            "FAIL: Source/feature keys do not "
            "match exactly."
        )

        return False

    print(
        "PASS: Source keys are exactly preserved."
    )

    return True


# ============================================================
# 6. DATE RANGE AUDIT
# ============================================================

def audit_dates(
    source,
    features
):

    print("\n" + "=" * 70)
    print("6. DATE RANGE AUDIT")
    print("=" * 70)

    source_min = source["date"].min()
    source_max = source["date"].max()

    feature_min = features["date"].min()
    feature_max = features["date"].max()

    print(
        f"Source date range: "
        f"{source_min.date()} → "
        f"{source_max.date()}"
    )

    print(
        f"Feature date range: "
        f"{feature_min.date()} → "
        f"{feature_max.date()}"
    )

    if source_min != feature_min:

        print(
            "FAIL: Minimum dates do not match."
        )

        return False

    print(
        "PASS: Minimum dates match."
    )

    if source_max != feature_max:

        print(
            "FAIL: Maximum dates do not match."
        )

        return False

    print(
        "PASS: Maximum dates match."
    )

    if feature_min < MIN_DATE:

        print(
            "FAIL: Feature data contains dates "
            "before MIN_DATE."
        )

        return False

    if feature_max > MAX_DATE:

        print(
            "FAIL: Feature data contains dates "
            "after MAX_DATE."
        )

        return False

    print(
        "PASS: Date boundaries valid."
    )

    return True


# ============================================================
# 7. TEMPORAL ORDER AUDIT
# ============================================================

def audit_temporal_order(
    features
):

    print("\n" + "=" * 70)
    print("7. TEMPORAL ORDER AUDIT")
    print("=" * 70)

    violations = 0

    for ticker, group in features.groupby(
        "ticker"
    ):

        dates = (
            group["date"]
            .to_numpy()
        )

        if len(dates) <= 1:
            continue

        if not np.all(
            dates[1:] >= dates[:-1]
        ):

            violations += 1

    print(
        f"Temporal ordering violations: "
        f"{violations}"
    )

    if violations > 0:

        print(
            "FAIL: Temporal ordering violations found."
        )

        return False

    print(
        "PASS: All funds are temporally ordered."
    )

    return True


# ============================================================
# 8. INFINITE VALUE AUDIT
# ============================================================

def audit_infinite_values(
    features
):

    print("\n" + "=" * 70)
    print("8. INFINITE VALUE AUDIT")
    print("=" * 70)

    numeric_columns = [
        column
        for column in features.columns
        if column not in [
            "ticker",
            "date",
        ]
    ]

    infinite_count = (
        np.isinf(
            features[numeric_columns]
            .to_numpy(dtype=float)
        )
        .sum()
    )

    print(
        f"Total infinite values: "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        print(
            "FAIL: Infinite values found."
        )

        return False

    print(
        "PASS: No infinite values."
    )

    return True


# ============================================================
# 9. CROSS-SECTIONAL PERCENTILE AUDIT
# ============================================================

def audit_percentiles(
    features
):

    print("\n" + "=" * 70)
    print("9. CROSS-SECTIONAL PERCENTILE AUDIT")
    print("=" * 70)

    passed = True

    for feature in PERCENTILE_FEATURES:

        column = f"{feature}_pct"

        valid_values = (
            features[column]
            .dropna()
        )

        if len(valid_values) == 0:

            print(
                f"FAIL: {column} contains no "
                f"valid values."
            )

            passed = False
            continue

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
            f"{column}: "
            f"min={minimum:.6f}, "
            f"max={maximum:.6f}, "
            f"unique={unique_count:,}"
        )

        if minimum < 0 or maximum > 1:

            passed = False

        if unique_count <= 1:

            passed = False

    if not passed:

        print(
            "\nFAIL: Percentile feature validation failed."
        )

        return False

    print(
        "\nPASS: All percentile features are "
        "within [0, 1] and vary."
    )

    return True


# ============================================================
# 10. CROSS-SECTIONAL VARIATION AUDIT
# ============================================================

def audit_cross_sectional_variation(
    features
):

    print("\n" + "=" * 70)
    print("10. CROSS-SECTIONAL VARIATION AUDIT")
    print("=" * 70)

    passed = True

    for feature in PERCENTILE_FEATURES:

        column = f"{feature}_pct"

        max_unique_per_date = (
            features
            .groupby("date")[column]
            .nunique()
            .max()
        )

        print(
            f"{column}: "
            f"max unique values/date = "
            f"{max_unique_per_date}"
        )

        if (
            pd.isna(max_unique_per_date)
            or
            max_unique_per_date <= 1
        ):

            passed = False

    if not passed:

        print(
            "\nFAIL: Percentile features do not "
            "show sufficient cross-sectional variation."
        )

        return False

    print(
        "\nPASS: Percentile features show "
        "cross-sectional variation."
    )

    return True


# ============================================================
# 11. CONSTANT / LOW-VARIANCE FEATURE AUDIT
# ============================================================

def audit_constant_features(
    features
):

    print("\n" + "=" * 70)
    print("11. CONSTANT / LOW-VARIANCE FEATURE AUDIT")
    print("=" * 70)

    numeric_columns = [
        column
        for column in features.columns
        if column not in [
            "ticker",
            "date",
        ]
    ]

    constant_features = []
    zero_variance_features = []

    for column in numeric_columns:

        valid_values = (
            features[column]
            .dropna()
        )

        if len(valid_values) == 0:
            continue

        if valid_values.nunique() <= 1:

            constant_features.append(
                column
            )

        if valid_values.var() == 0:

            zero_variance_features.append(
                column
            )

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Zero-variance features: "
        f"{len(zero_variance_features)}"
    )

    if constant_features:

        print(
            f"FAIL: Constant features: "
            f"{constant_features}"
        )

        return False

    if zero_variance_features:

        print(
            f"FAIL: Zero-variance features: "
            f"{zero_variance_features}"
        )

        return False

    print(
        "PASS: No constant features."
    )

    return True


# ============================================================
# 12. NAN / WARM-UP AUDIT
# ============================================================

def audit_nan_patterns(
    source,
    features
):

    print("\n" + "=" * 70)
    print("12. NAN / WARM-UP AUDIT")
    print("=" * 70)

    passed = True

    # --------------------------------------------------------
    # Expected first-valid positions
    #
    # These are the positions produced by the feature
    # generator when enough history exists.
    #
    # For a fund with insufficient history:
    # all values being NaN is EXPECTED and PASS.
    # --------------------------------------------------------

    expected_first_valid = {

        # Momentum
        "return_5d": 5,
        "return_20d": 20,
        "return_60d": 60,
        "return_120d": 120,
        "return_252d": 252,

        # Volatility
        "volatility_20d": 20,
        "volatility_60d": 60,
        "volatility_120d": 120,
        "volatility_252d": 252,

        # Drawdown
        "drawdown_60d": 59,
        "drawdown_120d": 119,
        "drawdown_252d": 251,

        # Trend
        "nav_vs_sma_20": 19,
        "nav_vs_sma_50": 49,
        "nav_vs_sma_200": 199,

        # NAV range
        "nav_range_20d": 19,
        "nav_range_60d": 59,

        # Downside / upside volatility
        "downside_volatility_60d": 60,
        "upside_volatility_60d": 60,
    }

    # --------------------------------------------------------
    # Standard feature warm-up checks
    # --------------------------------------------------------

    for feature, expected_index in (
        expected_first_valid.items()
    ):

        for ticker, group in source.groupby(
            "scheme_code"
        ):

            group = (
                group
                .sort_values("date")
                .reset_index(drop=True)
            )

            feature_group = (
                features[
                    features["ticker"].astype(str)
                    == str(ticker)
                ]
                .sort_values("date")
                .reset_index(drop=True)
            )

            observation_count = len(group)

            values = (
                feature_group[feature]
                .reset_index(drop=True)
            )

            valid_indices = np.flatnonzero(
                values.notna().to_numpy()
            )

            # ------------------------------------------------
            # Insufficient history
            #
            # If there are not enough observations for the
            # expected first-valid position, all NaN is correct.
            # ------------------------------------------------

            if observation_count <= expected_index:

                if len(valid_indices) != 0:

                    print(
                        f"FAIL: {ticker} / {feature}: "
                        f"fund has only "
                        f"{observation_count} observations "
                        f"but contains valid values."
                    )

                    passed = False

                continue

            # ------------------------------------------------
            # Sufficient history
            # ------------------------------------------------

            if len(valid_indices) == 0:

                print(
                    f"FAIL: {ticker} / {feature}: "
                    f"has {observation_count} observations "
                    f"but no valid values."
                )

                passed = False

                continue

            first_valid_index = (
                valid_indices[0]
            )

            if first_valid_index != expected_index:

                print(
                    f"FAIL: {ticker} / {feature}: "
                    f"first valid index = "
                    f"{first_valid_index}, "
                    f"expected = "
                    f"{expected_index}"
                )

                passed = False

    # --------------------------------------------------------
    # Rolling Sharpe validation
    #
    # Sharpe is reconstructed directly from NAV.
    #
    # First valid index is NOT assumed to be 60 because:
    #
    #     Sharpe = mean / std
    #
    # and when std == 0 the generator intentionally returns NaN.
    # --------------------------------------------------------

    sharpe_validation_failures = 0
    sharpe_post_warmup_missing = 0

    for ticker, source_group in source.groupby(
        "scheme_code"
    ):

        source_group = (
            source_group
            .sort_values("date")
            .reset_index(drop=True)
        )

        nav = (
            source_group["nav"]
            .astype(float)
            .reset_index(drop=True)
        )

        daily_return = nav.pct_change()

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

        expected_sharpe = np.where(
            rolling_std_return > 0,
            (
                rolling_mean_return
                .div(rolling_std_return)
                .mul(np.sqrt(252))
            ),
            np.nan
        )

        feature_group = (
            features[
                features["ticker"].astype(str)
                == str(ticker)
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        actual_sharpe = (
            feature_group[
                "rolling_sharpe_60d"
            ]
            .to_numpy(dtype=float)
        )

        expected_sharpe = np.asarray(
            expected_sharpe,
            dtype=float
        )

        # ----------------------------------------------------
        # Length
        # ----------------------------------------------------

        if (
            len(actual_sharpe)
            != len(expected_sharpe)
        ):

            print(
                f"FAIL: {ticker} / "
                f"rolling_sharpe_60d: "
                f"length mismatch."
            )

            sharpe_validation_failures += 1

            continue

        # ----------------------------------------------------
        # Validity pattern
        # ----------------------------------------------------

        expected_valid = np.isfinite(
            expected_sharpe
        )

        actual_valid = np.isfinite(
            actual_sharpe
        )

        validity_mismatch = (
            expected_valid != actual_valid
        ).sum()

        if validity_mismatch > 0:

            print(
                f"FAIL: {ticker} / "
                f"rolling_sharpe_60d: "
                f"validity mismatches = "
                f"{validity_mismatch}"
            )

            sharpe_validation_failures += 1

        # ----------------------------------------------------
        # Numerical values
        # ----------------------------------------------------

        valid_mask = (
            expected_valid
            &
            actual_valid
        )

        if valid_mask.any():

            values_match = np.allclose(
                actual_sharpe[valid_mask],
                expected_sharpe[valid_mask],
                rtol=1e-10,
                atol=1e-12
            )

            if not values_match:

                print(
                    f"FAIL: {ticker} / "
                    f"rolling_sharpe_60d: "
                    f"numerical values do not match."
                )

                sharpe_validation_failures += 1

        # ----------------------------------------------------
        # Valid post-warm-up Sharpe
        # ----------------------------------------------------

        if len(expected_sharpe) > 60:

            if not expected_valid[60:].any():

                sharpe_post_warmup_missing += 1

    print(
        f"Rolling Sharpe validity/value mismatches: "
        f"{sharpe_validation_failures}"
    )

    print(
        f"Funds with no mathematically valid "
        f"post-warm-up Sharpe: "
        f"{sharpe_post_warmup_missing}"
    )

    if (
        sharpe_validation_failures == 0
        and
        sharpe_post_warmup_missing == 0
    ):

        print(
            "PASS: Rolling Sharpe behavior matches "
            "its mathematical definition."
        )

    else:

        passed = False

    # --------------------------------------------------------
    # Percentile validity consistency
    # --------------------------------------------------------

    percentile_pairs = [

        (
            "return_20d",
            "return_20d_pct"
        ),

        (
            "return_60d",
            "return_60d_pct"
        ),

        (
            "return_120d",
            "return_120d_pct"
        ),

        (
            "return_252d",
            "return_252d_pct"
        ),

        (
            "volatility_60d",
            "volatility_60d_pct"
        ),

        (
            "drawdown_120d",
            "drawdown_120d_pct"
        ),

        (
            "nav_range_60d",
            "nav_range_60d_pct"
        ),
    ]

    for base_feature, pct_feature in percentile_pairs:

        base_valid = (
            features[base_feature]
            .notna()
        )

        pct_valid = (
            features[pct_feature]
            .notna()
        )

        mismatch_count = (
            base_valid != pct_valid
        ).sum()

        print(
            f"{base_feature} ↔ "
            f"{pct_feature}: "
            f"validity mismatches = "
            f"{mismatch_count}"
        )

        if mismatch_count > 0:

            passed = False

    if passed:

        print(
            "\nPASS: NaN/warm-up behavior is valid."
        )

    else:

        print(
            "\nFAIL: One or more NaN/warm-up "
            "checks failed."
        )

    return passed


# ============================================================
# 13. FEATURE SANITY AUDIT
# ============================================================

def audit_feature_sanity(
    features
):

    print("\n" + "=" * 70)
    print("13. FEATURE SANITY AUDIT")
    print("=" * 70)

    passed = True

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    drawdown_features = [
        "drawdown_60d",
        "drawdown_120d",
        "drawdown_252d",
    ]

    for feature in drawdown_features:

        positive_count = (
            (
                features[feature]
                > 0
            )
            .sum()
        )

        print(
            f"{feature}: "
            f"positive values = "
            f"{positive_count}"
        )

        if positive_count > 0:

            passed = False

    # --------------------------------------------------------
    # NAV range
    # --------------------------------------------------------

    nav_range_features = [
        "nav_range_20d",
        "nav_range_60d",
    ]

    for feature in nav_range_features:

        negative_count = (
            (
                features[feature]
                < 0
            )
            .sum()
        )

        print(
            f"{feature}: "
            f"negative values = "
            f"{negative_count}"
        )

        if negative_count > 0:

            passed = False

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    volatility_features = [
        "volatility_20d",
        "volatility_60d",
        "volatility_120d",
        "volatility_252d",
        "downside_volatility_60d",
        "upside_volatility_60d",
    ]

    for feature in volatility_features:

        negative_count = (
            (
                features[feature]
                < 0
            )
            .sum()
        )

        print(
            f"{feature}: "
            f"negative values = "
            f"{negative_count}"
        )

        if negative_count > 0:

            passed = False

    if not passed:

        print(
            "\nFAIL: Feature sanity checks failed."
        )

        return False

    print(
        "\nPASS: Feature value sanity checks passed."
    )

    return True


# ============================================================
# FINAL AUDIT
# ============================================================

def main():

    source, features = load_source()

    print(
        f"\nSource rows:   "
        f"{len(source):,}"
    )

    print(
        f"Feature rows:  "
        f"{len(features):,}"
    )

    print(
        f"Source columns: "
        f"{len(source.columns)}"
    )

    print(
        f"Feature columns: "
        f"{len(features.columns)}"
    )

    results = {}

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    results["structure"] = audit_structure(
        features
    )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    results["fund_universe"] = (
        audit_fund_universe(
            source,
            features
        )
    )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    results["keys"] = audit_keys(
        source,
        features
    )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    results["row_preservation"] = (
        audit_row_preservation(
            source,
            features
        )
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    results["key_preservation"] = (
        audit_key_preservation(
            source,
            features
        )
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    results["dates"] = audit_dates(
        source,
        features
    )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    results["temporal_order"] = (
        audit_temporal_order(
            features
        )
    )

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    results["infinite_values"] = (
        audit_infinite_values(
            features
        )
    )

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    results["percentiles"] = (
        audit_percentiles(
            features
        )
    )

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    results["cross_sectional_variation"] = (
        audit_cross_sectional_variation(
            features
        )
    )

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    results["constant_features"] = (
        audit_constant_features(
            features
        )
    )

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    results["nan_patterns"] = (
        audit_nan_patterns(
            source,
            features
        )
    )

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    results["feature_sanity"] = (
        audit_feature_sanity(
            features
        )
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL MUTUAL FUND FEATURE AUDIT")
    print("=" * 70)

    for audit_name, passed in results.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{audit_name:30s} : "
            f"{status}"
        )

    overall_pass = all(
        results.values()
    )

    print("\n" + "=" * 70)

    if overall_pass:

        print(
            "MUTUAL FUND FEATURE AUDIT: PASS"
        )

        print(
            "\nAll mutual fund feature audits passed."
        )

        print(
            "The mutual fund feature layer is "
            "ready to be locked."
        )

    else:

        print(
            "MUTUAL FUND FEATURE AUDIT: FAIL"
        )

        print(
            "\nOne or more checks failed."
        )

    print("=" * 70)

    if not overall_pass:

        raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
