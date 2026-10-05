from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# INVESTMENT COPILOT - ML DATASET BUILDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
TARGET_DIR = PROJECT_ROOT / "data" / "targets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ml"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

CONFIG = {

    "stocks": {
        "feature_file": "stock_features.parquet",

        "feature_id": "ticker",
        "target_id": "ticker",

        "expected_assets": 100,

        "output_prefix": "stock",

        "targets": {
            "6m": {
                "file": "stock_6m_1y_targets.parquet",
                "date_column": "observation_date",
                "label_column": "label_6m",
                "future_date_column": "future_date_6m",
            },

            "1y": {
                "file": "stock_6m_1y_targets.parquet",
                "date_column": "observation_date",
                "label_column": "label_1y",
                "future_date_column": "future_date_1y",
            },

            "2y": {
                "file": "stock_2y_targets.parquet",
                "date_column": "date",
                "label_column": "label_2y",
                "future_date_column": "future_date_2y",
            },
        },
    },


    "etfs": {
        "feature_file": "etf_features.parquet",

        "feature_id": "ticker",
        "target_id": "Ticker",

        "expected_assets": 41,

        "output_prefix": "etf",

        "targets": {
            "6m": {
                "file": "etf_6m_1y_targets.parquet",
                "date_column": "Date",
                "label_column": "label_6m",
                "future_date_column": "future_date_6m",
            },

            "1y": {
                "file": "etf_6m_1y_targets.parquet",
                "date_column": "Date",
                "label_column": "label_1y",
                "future_date_column": "future_date_1y",
            },

            "2y": {
                "file": "etf_2y_targets.parquet",
                "date_column": "Date",
                "label_column": "label_2y",
                "future_date_column": "future_date_2y",
            },
        },
    },


    "mutual_funds": {
        "feature_file": "mutual_fund_features.parquet",

        "feature_id": "ticker",
        "target_id": "scheme_code",

        "expected_assets": 100,

        "output_prefix": "mutual_fund",

        "targets": {
            "6m": {
                "file": "mutual_fund_6m_1y_targets.parquet",
                "date_column": "date",
                "label_column": "label_6m",
                "future_date_column": "future_date_6m",
            },

            "1y": {
                "file": "mutual_fund_6m_1y_targets.parquet",
                "date_column": "date",
                "label_column": "label_1y",
                "future_date_column": "future_date_1y",
            },

            "2y": {
                "file": "mutual_fund_2y_targets.parquet",
                "date_column": "date",
                "label_column": "label_2y",
                "future_date_column": "future_date_2y",
            },
        },
    },
}


# ============================================================
# LOCKED FEATURE SET
# ============================================================

MARKET_FEATURES = [

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


MUTUAL_FUND_FEATURES = [

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
    "nav_vs_sma_20",
    "nav_vs_sma_50",
    "nav_vs_sma_200",

    # NAV-specific
    "nav_range_20d",
    "nav_range_60d",
    "downside_volatility_60d",
    "upside_volatility_60d",
    "rolling_sharpe_60d",

    # Cross-sectional percentiles
    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",
    "volatility_60d_pct",
    "drawdown_120d_pct",
    "nav_range_60d_pct",
]


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_dates(values):

    dates = pd.to_datetime(
        values,
        errors="coerce"
    )

    if isinstance(
        dates.dtype,
        pd.DatetimeTZDtype
    ):
        dates = dates.dt.tz_localize(None)

    return dates.dt.normalize()


# ============================================================
# IDENTIFIER NORMALIZATION
# ============================================================

def normalize_identifier(
    values,
    asset_type,
):
    """
    Normalize identifiers only at the ML dataset boundary.

    Stocks / ETFs:
        ABB.NS  -> ABB_NS
        TCS.NS  -> TCS_NS

    Mutual funds:
        scheme_code remains numeric/string representation.
    """

    result = values.astype("string").str.strip()

    if asset_type in {"stocks", "etfs"}:

        # Target files use Yahoo-style .NS
        # Feature files use _NS.
        #
        # Normalize both representations to _NS.
        result = result.str.replace(
            ".NS",
            "_NS",
            regex=False,
        )

        # Also handle lowercase accidentally appearing.
        result = result.str.replace(
            ".ns",
            "_NS",
            regex=False,
        )

    elif asset_type == "mutual_funds":

        # Avoid values such as 120497.0
        result = result.str.replace(
            r"^(\d+)\.0$",
            r"\1",
            regex=True,
        )

    return result


# ============================================================
# KEY VALIDATION
# ============================================================

def validate_keys(
    df,
    name,
):
    """
    Validate asset/date keys.
    """

    if df["ticker"].isna().any():
        raise ValueError(
            f"{name}: null ticker values."
        )

    if df["date"].isna().any():
        raise ValueError(
            f"{name}: invalid date values."
        )

    duplicates = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicates:

        raise ValueError(
            f"{name}: {duplicates:,} "
            f"duplicate ticker/date rows."
        )


# ============================================================
# INFINITE VALUE VALIDATION
# ============================================================

def validate_infinite_values(
    df,
    columns,
    name,
):

    if not columns:
        return

    values = df[columns].to_numpy(
        dtype=float
    )

    count = int(
        np.isinf(values).sum()
    )

    if count:

        raise ValueError(
            f"{name}: {count:,} "
            f"infinite feature values."
        )


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    asset_type,
    config,
):

    path = (
        FEATURE_DIR
        / config["feature_file"]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    print(
        f"\nLoading features:\n{path}"
    )

    df = pd.read_parquet(path)

    if asset_type == "mutual_funds":

        feature_names = (
            MUTUAL_FUND_FEATURES
        )

    else:

        feature_names = MARKET_FEATURES

    required_columns = [
        config["feature_id"],
        "date",
        *feature_names,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{asset_type}: missing feature columns:\n"
            f"{missing}"
        )

    df = df[
        required_columns
    ].copy()

    # Standardize identifier.
    df = df.rename(
        columns={
            config["feature_id"]:
                "ticker"
        }
    )

    df["ticker"] = normalize_identifier(
        df["ticker"],
        asset_type,
    )

    df["date"] = normalize_dates(
        df["date"]
    )

    validate_keys(
        df,
        f"{asset_type} features",
    )

    validate_infinite_values(
        df,
        feature_names,
        f"{asset_type} features",
    )

    asset_count = df[
        "ticker"
    ].nunique()

    if asset_count != config[
        "expected_assets"
    ]:

        raise ValueError(
            f"{asset_type}: expected "
            f"{config['expected_assets']} "
            f"assets but found "
            f"{asset_count}."
        )

    print(
        f"Feature rows : {len(df):,}"
    )

    print(
        f"Assets       : {asset_count}"
    )

    print(
        f"Features     : {len(feature_names)}"
    )

    return df, feature_names


# ============================================================
# LOAD TARGETS
# ============================================================

def load_targets(
    asset_type,
    config,
    horizon,
):

    target_config = (
        config["targets"][horizon]
    )

    filename = target_config[
        "file"
    ]

    date_column = target_config[
        "date_column"
    ]

    label_column = target_config[
        "label_column"
    ]

    future_date_column = target_config[
        "future_date_column"
    ]

    path = (
        TARGET_DIR
        / filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Target file not found:\n{path}"
        )

    print(
        f"\nLoading {horizon} target:\n{path}"
    )

    df = pd.read_parquet(path)

    required_columns = [
        config["target_id"],
        date_column,
        label_column,
        future_date_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{asset_type}/{horizon}: "
            f"missing target columns:\n"
            f"{missing}"
        )

    # IMPORTANT:
    # Only keep the identifier, observation date,
    # label and future endpoint.
    #
    # Forward returns, future prices, scores,
    # percentiles, etc. are deliberately NOT
    # copied into the ML dataset.
    df = df[
        required_columns
    ].copy()

    df = df.rename(
        columns={
            config["target_id"]:
                "ticker",

            date_column:
                "date",

            label_column:
                "target",

            future_date_column:
                "label_end_date",
        }
    )

    df["ticker"] = normalize_identifier(
        df["ticker"],
        asset_type,
    )

    df["date"] = normalize_dates(
        df["date"]
    )

    df["label_end_date"] = normalize_dates(
        df["label_end_date"]
    )

    validate_keys(
        df,
        f"{asset_type}/{horizon} targets",
    )

    print(
        f"Target rows : {len(df):,}"
    )

    print(
        f"Target assets : "
        f"{df['ticker'].nunique()}"
    )

    return df


# ============================================================
# COMPARE FEATURE / TARGET KEYS
# ============================================================

def compare_keys(
    features,
    targets,
    asset_type,
    horizon,
):
    """
    Validate feature/target key alignment.

    Correct rule:

        target keys must be a subset of feature keys.

    Why?

    The feature layer contains the complete asset universe.

    A target layer can legitimately contain fewer observations
    because some observations/assets are not eligible for the
    particular investment horizon.

    Therefore:

        target-only keys  -> ERROR
        feature-only keys -> ALLOWED

    Example:
        Mutual-fund 2Y:
            Features: 100 funds
            Targets : 83 eligible funds

    This is expected.
    """

    feature_keys = pd.MultiIndex.from_frame(
        features[
            ["ticker", "date"]
        ]
    )

    target_keys = pd.MultiIndex.from_frame(
        targets[
            ["ticker", "date"]
        ]
    )

    feature_only = feature_keys.difference(
        target_keys
    )

    target_only = target_keys.difference(
        feature_keys
    )

    print(
        f"Feature-only keys: "
        f"{len(feature_only):,}"
    )

    print(
        f"Target-only keys : "
        f"{len(target_only):,}"
    )

    # ========================================================
    # TARGET-ONLY KEYS
    # ========================================================
    #
    # This is NEVER acceptable.
    #
    # A target observation must always have a corresponding
    # feature observation at the same asset/date.
    #

    if len(target_only) > 0:

        print(
            "\nERROR: target contains keys "
            "that are absent from features."
        )

        print(
            "First target-only key:"
        )

        print(
            target_only[0]
        )

        raise ValueError(
            f"{asset_type}/{horizon}: "
            f"{len(target_only):,} target keys "
            f"are missing from the feature dataset."
        )

    # ========================================================
    # FEATURE-ONLY KEYS
    # ========================================================
    #
    # These are allowed.
    #
    # They occur when the target-generation methodology
    # excludes observations that cannot produce a valid
    # target/cohort.
    #

    if len(feature_only) > 0:

        coverage = (
            len(target_keys)
            / len(feature_keys)
            * 100
        )

        print(
            "\nFeature-only keys are present."
        )

        print(
            "This is allowed because the target layer "
            "can contain fewer observations than the "
            "feature layer."
        )

        print(
            f"Target coverage of feature keys: "
            f"{coverage:.2f}%"
        )

        print(
            "Feature/target relationship: "
            "PASS (target is a subset of features)"
        )

    else:

        print(
            "Feature/target key relationship: "
            "EXACT MATCH"
        )

    # ========================================================
    # IMPORTANT:
    # THERE IS NO ADDITIONAL RAISE HERE.
    #
    # feature_only is allowed.
    # ========================================================

def build_dataset(
    asset_type,
    config,
    horizon,
    features,
    feature_names,
):

    print("\n")
    print("-" * 65)
    print(
        f"{asset_type.upper()} / "
        f"{horizon.upper()}"
    )
    print("-" * 65)

    targets = load_targets(
        asset_type,
        config,
        horizon,
    )

    # --------------------------------------------------------
    # KEY ALIGNMENT
    # --------------------------------------------------------

    compare_keys(
        features,
        targets,
        asset_type,
        horizon,
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = features.merge(
        targets,
        on=[
            "ticker",
            "date",
        ],
        how="inner",
        validate="one_to_one",
    )

    joined_rows = len(
        merged
    )

    print(
        f"\nJoined rows: "
        f"{joined_rows:,}"
    )

    # --------------------------------------------------------
    # TARGET VALIDATION
    # --------------------------------------------------------

    missing_target = (
        merged["target"].isna()
    ).sum()

    print(
        f"Rows without target: "
        f"{missing_target:,}"
    )

    merged = merged[
        merged["target"].notna()
    ].copy()

    if merged.empty:

        raise ValueError(
            f"{asset_type}/{horizon}: "
            f"no labeled observations."
        )

    # --------------------------------------------------------
    # FUTURE DATE VALIDATION
    # --------------------------------------------------------

    missing_end_date = (
        merged["label_end_date"]
        .isna()
    ).sum()

    if missing_end_date:

        raise ValueError(
            f"{asset_type}/{horizon}: "
            f"{missing_end_date:,} labeled rows "
            f"have no future endpoint."
        )

    invalid_end_date = (
        merged["label_end_date"]
        < merged["date"]
    ).sum()

    if invalid_end_date:

        raise ValueError(
            f"{asset_type}/{horizon}: "
            f"{invalid_end_date:,} rows have a "
            f"future endpoint earlier than the "
            f"observation date."
        )

    # --------------------------------------------------------
    # KEY VALIDATION AFTER MERGE
    # --------------------------------------------------------

    validate_keys(
        merged,
        f"{asset_type}/{horizon} merged",
    )

    # --------------------------------------------------------
    # INFINITE VALUE CHECK
    # --------------------------------------------------------

    validate_infinite_values(
        merged,
        feature_names,
        f"{asset_type}/{horizon} merged",
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    merged = merged.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    final_columns = [
        "ticker",
        "date",
        *feature_names,
        "label_end_date",
        "target",
    ]

    merged = merged[
        final_columns
    ]

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    print(
        f"Labeled rows       : "
        f"{len(merged):,}"
    )

    print(
        f"Assets             : "
        f"{merged['ticker'].nunique()}"
    )

    print(
        f"Date range         : "
        f"{merged['date'].min().date()} "
        f"-> "
        f"{merged['date'].max().date()}"
    )

    print(
        "\nTarget distribution:"
    )

    counts = (
        merged["target"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    percentages = (
        merged["target"]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .sort_index()
        * 100
    )

    for label in counts.index:

        print(
            f"  {label}: "
            f"{counts[label]:,} "
            f"({percentages[label]:.2f}%)"
        )

    # --------------------------------------------------------
    # FEATURE MISSINGNESS
    # --------------------------------------------------------

    missing = (
        merged[feature_names]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    complete_rows = int(
        merged[
            feature_names
        ]
        .notna()
        .all(axis=1)
        .sum()
    )

    print(
        "\nRows with all 27 features:"
    )

    print(
        f"  {complete_rows:,} / "
        f"{len(merged):,}"
    )

    print(
        "\nTop feature NaN counts:"
    )

    print(
        missing.head(10).to_string()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_directory = (
        OUTPUT_DIR
        / asset_type
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / (
            f"{config['output_prefix']}"
            f"_{horizon}_ml.parquet"
        )
    )

    merged.to_parquet(
        output_file,
        index=False,
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    return {
        "asset": asset_type,
        "horizon": horizon,
        "joined_rows": joined_rows,
        "labeled_rows": len(merged),
        "assets": merged["ticker"].nunique(),
        "complete_feature_rows": complete_rows,
        "min_date": merged["date"].min(),
        "max_date": merged["date"].max(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INVESTMENT COPILOT - "
        "ML DATASET BUILDER"
    )
    print("=" * 70)

    results = []

    for asset_type, config in CONFIG.items():

        print("\n")
        print("=" * 70)
        print(
            f"PROCESSING "
            f"{asset_type.upper()}"
        )
        print("=" * 70)

        features, feature_names = (
            load_features(
                asset_type,
                config,
            )
        )

        for horizon in (
            "6m",
            "1y",
            "2y",
        ):

            result = build_dataset(
                asset_type=asset_type,
                config=config,
                horizon=horizon,
                features=features,
                feature_names=feature_names,
            )

            results.append(result)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print(
        "FINAL ML DATASET BUILD SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print(
        "ML DATASET BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "These datasets are structurally built, "
        "but they are NOT locked yet."
    )

    print(
        "The next step is the independent "
        "ML dataset audit."
    )


if __name__ == "__main__":
    main()