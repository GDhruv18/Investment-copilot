from pathlib import Path
import numpy as np
import pandas as pd
import gc
import time


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
ML_DIR = PROJECT_ROOT / "data" / "ml"
OUTPUT_DIR = ML_DIR / "sequences"


# =============================================================================
# LOCKED LOOKBACK POLICY
# =============================================================================

LOOKBACKS = {
    "6m": 126,
    "1y": 252,
    "2y": 252,
}


# =============================================================================
# LOCKED FEATURE POLICY
# =============================================================================

STOCK_ETF_FEATURES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",

    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",

    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",

    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_sma_200",

    "volume_ratio_20d",
    "volume_ratio_60d",
    "volume_change_20d",

    "range_20d",
    "range_60d",

    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",

    "volatility_60d_pct",
    "drawdown_120d_pct",
    "volume_ratio_20d_pct",
]


MUTUAL_FUND_FEATURES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_252d",

    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_252d",

    "drawdown_60d",
    "drawdown_120d",
    "drawdown_252d",

    "nav_vs_sma_20",
    "nav_vs_sma_50",
    "nav_vs_sma_200",

    "nav_range_20d",
    "nav_range_60d",

    "downside_volatility_60d",
    "upside_volatility_60d",
    "rolling_sharpe_60d",

    "return_20d_pct",
    "return_60d_pct",
    "return_120d_pct",
    "return_252d_pct",

    "volatility_60d_pct",
    "drawdown_120d_pct",
    "nav_range_60d_pct",
]


# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

ASSET_CONFIG = {
    "stocks": {
        "feature_file": FEATURE_DIR / "stock_features.parquet",
        "ml_dir": ML_DIR / "stocks",
        "prefix": "stock",
        "features": STOCK_ETF_FEATURES,
    },

    "etfs": {
        "feature_file": FEATURE_DIR / "etf_features.parquet",
        "ml_dir": ML_DIR / "etfs",
        "prefix": "etf",
        "features": STOCK_ETF_FEATURES,
    },

    "mutual_funds": {
        "feature_file": FEATURE_DIR / "mutual_fund_features.parquet",
        "ml_dir": ML_DIR / "mutual_funds",
        "prefix": "mutual_fund",
        "features": MUTUAL_FUND_FEATURES,
    },
}


HORIZONS = ["2y"]

SPLITS = ["train", "validation", "test"]


# =============================================================================
# TARGET NORMALIZATION
# =============================================================================

STOCK_ETF_TARGET_MAP = {
    "Weak": 0,
    "Neutral": 1,
    "Attractive": 2,
}

MUTUAL_FUND_TARGET_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2,
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_date_column(df, column="date"):
    """
    Normalize dates to pandas datetime64[ns] without timezone conversion.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="raise")

    if hasattr(df[column].dt, "tz") and df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_localize(None)

    df[column] = df[column].dt.normalize()

    return df


def validate_feature_timeline(df, asset_class, feature_columns):
    """
    Validate the complete feature timeline before sequence construction.
    """

    print("Feature timeline validation:")

    required = ["ticker", "date"] + feature_columns

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{asset_class}: missing required feature columns: {missing}"
        )

    if df["ticker"].isna().any():
        raise ValueError(f"{asset_class}: null ticker values found.")

    if df["date"].isna().any():
        raise ValueError(f"{asset_class}: null date values found.")

    duplicate_count = df.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"{asset_class}: {duplicate_count} duplicate ticker/date rows found."
        )

    numeric = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    infinite_count = np.isinf(
        numeric.to_numpy(dtype=np.float64)
    ).sum()

    if infinite_count:
        raise ValueError(
            f"{asset_class}: {infinite_count} infinite feature values found."
        )

    print("  Required columns : PASS")
    print("  Null identifiers  : PASS")
    print("  Duplicate keys    : PASS")
    print("  Infinite values   : PASS")


def load_feature_timeline(asset_class):
    """
    Load the complete feature timeline.

    Important:
    This is NOT the labeled ML dataset.

    We intentionally use the full historical feature timeline so that
    historical observations preceding a prediction date remain available
    when constructing the LSTM lookback window.
    """

    config = ASSET_CONFIG[asset_class]

    feature_file = config["feature_file"]

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{feature_file}"
        )

    print(f"Loading feature timeline:")
    print(f"  {feature_file}")

    df = pd.read_parquet(feature_file)

    df = normalize_date_column(df, "date")

    feature_columns = config["features"]

    validate_feature_timeline(
        df,
        asset_class,
        feature_columns
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    print(
        f"  Rows       : {len(df):,}"
    )

    print(
        f"  Assets     : {df['ticker'].nunique():,}"
    )

    print(
        f"  Date range : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print("Feature timeline validation: PASS")

    return df


def load_split_dataset(asset_class, horizon, split):
    """
    Load an existing temporal split.

    No target generation happens here.
    """

    config = ASSET_CONFIG[asset_class]

    filename = (
        f"{config['prefix']}_{horizon}_{split}.parquet"
    )

    path = config["ml_dir"] / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{path}"
        )

    df = pd.read_parquet(path)

    df = normalize_date_column(df, "date")

    if "ticker" not in df.columns:
        raise ValueError(
            f"{path}: ticker column missing."
        )

    if "target" not in df.columns:
        raise ValueError(
            f"{path}: target column missing."
        )

    if "label_end_date" not in df.columns:
        raise ValueError(
            f"{path}: label_end_date column missing."
        )

    df["label_end_date"] = pd.to_datetime(
        df["label_end_date"],
        errors="raise"
    )

    if hasattr(df["label_end_date"].dt, "tz") and \
       df["label_end_date"].dt.tz is not None:
        df["label_end_date"] = (
            df["label_end_date"].dt.tz_localize(None)
        )

    df["label_end_date"] = (
        df["label_end_date"].dt.normalize()
    )

    return df


def normalize_targets(target_series, asset_class):
    """
    Convert the existing target encoding into:

        Weak       = 0
        Neutral    = 1
        Attractive = 2

    for stocks/ETFs.

    Mutual funds already use 0/1/2 semantically.
    """

    if asset_class in ["stocks", "etfs"]:
        mapping = STOCK_ETF_TARGET_MAP

        normalized = target_series.astype(str).map(mapping)

    else:
        mapping = MUTUAL_FUND_TARGET_MAP

        normalized = target_series.map(mapping)

    if normalized.isna().any():

        bad_values = (
            target_series[normalized.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{asset_class}: unknown target values: "
            f"{bad_values}"
        )

    return normalized.astype(np.int8)


# =============================================================================
# SEQUENCE CONSTRUCTION
# =============================================================================

def build_sequences_for_split(
    feature_df,
    split_df,
    feature_columns,
    lookback,
    asset_class,
    horizon,
    split_name,
):
    """
    Construct:

        X = [samples, lookback, 27]

        y = [samples]

    using the COMPLETE feature timeline.

    Each prediction date comes from the existing temporal split.
    """

    print()
    print("-" * 80)
    print(
        f"{asset_class.upper()}_{horizon.upper()} | "
        f"LOOKBACK = {lookback}"
    )
    print("-" * 80)

    print()
    print(f"  {split_name.upper()}:")

    source_rows = len(split_df)

    print(
        f"    Source prediction rows     : "
        f"{source_rows:,}"
    )

    # -------------------------------------------------------------------------
    # Prepare feature timeline by asset
    # -------------------------------------------------------------------------

    grouped_features = {}

    for ticker, group in feature_df.groupby(
        "ticker",
        sort=False
    ):

        group = group.sort_values("date").reset_index(
            drop=True
        )

        dates = group["date"].to_numpy(
            dtype="datetime64[ns]"
        )

        values = group[feature_columns].to_numpy(
            dtype=np.float32
        )

        grouped_features[ticker] = (
            dates,
            values
        )

    # -------------------------------------------------------------------------
    # Allocate lists.
    #
    # We build each sequence as float32 immediately.
    # -------------------------------------------------------------------------

    X_list = []
    y_list = []

    metadata_rows = []

    valid_sequences = 0
    missing_features = 0
    insufficient_history = 0
    missing_assets = 0

    start_time = time.time()

    total = len(split_df)

    # -------------------------------------------------------------------------
    # Build sequences
    # -------------------------------------------------------------------------

    for row_number, row in enumerate(
        split_df.itertuples(index=False),
        start=1
    ):

        ticker = row.ticker
        prediction_date = row.date
        label_end_date = row.label_end_date
        target = row.target

        # -------------------------------------------------------------
        # Asset exists in full feature timeline?
        # -------------------------------------------------------------

        if ticker not in grouped_features:

            missing_assets += 1

            continue

        dates, values = grouped_features[ticker]

        # -------------------------------------------------------------
        # Find prediction date
        # -------------------------------------------------------------

        date_position = np.searchsorted(
            dates,
            np.datetime64(prediction_date),
            side="left"
        )

        if (
            date_position >= len(dates)
            or dates[date_position]
            != np.datetime64(prediction_date)
        ):

            missing_features += 1

            continue

        # -------------------------------------------------------------
        # Need exactly LOOKBACK observations ending at
        # prediction_date.
        # -------------------------------------------------------------

        start_position = (
            date_position - lookback + 1
        )

        if start_position < 0:

            insufficient_history += 1

            continue

        end_position = date_position + 1

        sequence = values[
            start_position:end_position
        ]

        # -------------------------------------------------------------
        # Shape validation
        # -------------------------------------------------------------

        if sequence.shape != (
            lookback,
            len(feature_columns)
        ):

            raise RuntimeError(
                f"Unexpected sequence shape for "
                f"{ticker} {prediction_date}: "
                f"{sequence.shape}"
            )

        # -------------------------------------------------------------
        # No NaN / infinity inside the sequence.
        #
        # We deliberately do not impute here.
        # -------------------------------------------------------------

        if not np.isfinite(sequence).all():

            missing_features += 1

            continue

        # -------------------------------------------------------------
        # Normalize target
        # -------------------------------------------------------------

        normalized_target = normalize_targets(
            pd.Series([target]),
            asset_class
        ).iloc[0]

        # -------------------------------------------------------------
        # Store
        # -------------------------------------------------------------

        X_list.append(sequence)

        y_list.append(normalized_target)

        metadata_rows.append(
            {
                "ticker": ticker,
                "date": prediction_date,
                "label_end_date": label_end_date,
                "target": target,
                "target_numeric": int(
                    normalized_target
                ),
            }
        )

        valid_sequences += 1

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if (
            row_number % 10000 == 0
            or row_number == total
        ):

            elapsed = time.time() - start_time

            rate = (
                row_number / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"    Progress: "
                f"{row_number:,}/{total:,} "
                f"({row_number / total * 100:.1f}%) | "
                f"valid={valid_sequences:,} | "
                f"rate={rate:,.0f} rows/s"
            )

    # -------------------------------------------------------------------------
    # Convert to contiguous arrays
    # -------------------------------------------------------------------------

    if not X_list:

        raise RuntimeError(
            f"No valid sequences generated for "
            f"{asset_class} {horizon} {split_name}."
        )

    print()
    print("  Converting sequences to NumPy arrays...")

    X = np.stack(
        X_list,
        axis=0
    ).astype(
        np.float32,
        copy=False
    )

    y = np.asarray(
        y_list,
        dtype=np.int8
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    # -------------------------------------------------------------------------
    # Free list memory
    # -------------------------------------------------------------------------

    del X_list
    del y_list
    del metadata_rows

    gc.collect()

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    expected_shape = (
        valid_sequences,
        lookback,
        len(feature_columns)
    )

    if X.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected final X shape: "
            f"{X.shape}, expected {expected_shape}"
        )

    if y.shape != (valid_sequences,):

        raise RuntimeError(
            f"Unexpected y shape: "
            f"{y.shape}"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "X contains NaN or infinity."
        )

    if not np.isin(
        y,
        [0, 1, 2]
    ).all():

        raise RuntimeError(
            "y contains values outside 0/1/2."
        )

    # -------------------------------------------------------------------------
    # Memory estimate
    # -------------------------------------------------------------------------

    x_size_gb = (
        X.nbytes / (1024 ** 3)
    )

    y_size_mb = (
        y.nbytes / (1024 ** 2)
    )

    print()
    print(
        f"    Valid sequences            : "
        f"{valid_sequences:,}"
    )

    print(
        f"    Missing feature sequences  : "
        f"{missing_features:,}"
    )

    print(
        f"    Insufficient history       : "
        f"{insufficient_history:,}"
    )

    print(
        f"    Missing assets             : "
        f"{missing_assets:,}"
    )

    print(
        f"    X shape                   : "
        f"{X.shape}"
    )

    print(
        f"    X dtype                   : "
        f"{X.dtype}"
    )

    print(
        f"    X memory                  : "
        f"{x_size_gb:.2f} GB"
    )

    print(
        f"    y dtype                   : "
        f"{y.dtype}"
    )

    print(
        f"    y memory                  : "
        f"{y_size_mb:.2f} MB"
    )

    print(
        f"    Assets                    : "
        f"{metadata['ticker'].nunique():,}"
    )

    print(
        f"    Target distribution       : "
        f"{dict(metadata['target_numeric'].value_counts().sort_index())}"
    )

    return X, y, metadata


# =============================================================================
# SAVE
# =============================================================================

def save_sequence_dataset(
    X,
    y,
    metadata,
    asset_class,
    horizon,
    split_name,
):
    """
    Save X and y separately as .npy files.

    We intentionally do NOT use np.savez_compressed().
    """

    output_dir = (
        OUTPUT_DIR
        / asset_class
        / horizon
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X_path = (
        output_dir
        / f"{asset_class}_{horizon}_{split_name}_X.npy"
    )

    y_path = (
        output_dir
        / f"{asset_class}_{horizon}_{split_name}_y.npy"
    )

    metadata_path = (
        output_dir
        / f"{asset_class}_{horizon}_{split_name}_metadata.parquet"
    )

    print()
    print("  SAVING:")

    print(
        f"    X → {X_path.name}"
    )

    start = time.time()

    # -------------------------------------------------------------------------
    # np.save produces an uncompressed .npy file.
    #
    # This is intentionally fast and supports mmap_mode during training.
    # -------------------------------------------------------------------------

    np.save(
        X_path,
        X,
        allow_pickle=False
    )

    elapsed = time.time() - start

    print(
        f"    X saved in {elapsed:.1f} seconds"
    )

    print(
        f"    y → {y_path.name}"
    )

    start = time.time()

    np.save(
        y_path,
        y,
        allow_pickle=False
    )

    elapsed = time.time() - start

    print(
        f"    y saved in {elapsed:.1f} seconds"
    )

    print(
        f"    metadata → {metadata_path.name}"
    )

    start = time.time()

    metadata.to_parquet(
        metadata_path,
        index=False
    )

    elapsed = time.time() - start

    print(
        f"    metadata saved in {elapsed:.1f} seconds"
    )

    # -------------------------------------------------------------------------
    # Verify files exist and sizes are non-zero.
    # -------------------------------------------------------------------------

    for path in [
        X_path,
        y_path,
        metadata_path,
    ]:

        if not path.exists():

            raise RuntimeError(
                f"Expected output file was not created:\n"
                f"{path}"
            )

        if path.stat().st_size == 0:

            raise RuntimeError(
                f"Output file is empty:\n"
                f"{path}"
            )

    print()
    print("    Save verification: PASS")

    return (
        X_path,
        y_path,
        metadata_path
    )


# =============================================================================
# SINGLE TASK
# =============================================================================

def process_task(
    feature_df,
    asset_class,
    horizon,
    split_name,
):
    """
    Process one asset class / horizon / split.
    """

    config = ASSET_CONFIG[asset_class]

    lookback = LOOKBACKS[horizon]

    split_df = load_split_dataset(
        asset_class,
        horizon,
        split_name
    )

    X, y, metadata = build_sequences_for_split(
        feature_df=feature_df,
        split_df=split_df,
        feature_columns=config["features"],
        lookback=lookback,
        asset_class=asset_class,
        horizon=horizon,
        split_name=split_name,
    )

    paths = save_sequence_dataset(
        X=X,
        y=y,
        metadata=metadata,
        asset_class=asset_class,
        horizon=horizon,
        split_name=split_name,
    )

    # -------------------------------------------------------------------------
    # Release memory before next split.
    # -------------------------------------------------------------------------

    del X
    del y
    del metadata
    del split_df

    gc.collect()

    return paths


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("INVESTMENT COPILOT - LSTM SEQUENCE BUILDER")
    print("=" * 80)

    print()
    print("LOCKED LOOKBACK POLICY")
    print("----------------------")
    print("6M : 126 trading observations")
    print("1Y : 252 trading observations")
    print("2Y : 252 trading observations")

    print()
    print("LOCKED FEATURE POLICY")
    print("---------------------")
    print("Features per timestep : 27")
    print("Scaling               : NOT performed here")
    print("Target generation     : NOT performed here")
    print("Temporal boundaries   : Existing split files")
    print("Storage               : Separate .npy files")
    print("Data type             : float32")

    print()
    print("IMPORTANT")
    print("---------")
    print("The complete feature timeline is used for historical context.")
    print("Existing temporal split files determine prediction dates.")
    print("No target regeneration occurs.")
    print("No feature scaling occurs.")
    print("No dataset modification occurs.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================================
    # PROCESS ASSET CLASSES
    # =========================================================================

    for asset_class in [
        "stocks",
        "etfs",
        "mutual_funds",
    ]:

        print()
        print()
        print("#" * 80)
        print(
            f"ASSET CLASS: {asset_class.upper()}"
        )
        print("#" * 80)

        feature_df = load_feature_timeline(
            asset_class
        )

        for horizon in HORIZONS:

            print()
            print()
            print(
                f"Preparing horizon: "
                f"{horizon.upper()} "
                f"(lookback={LOOKBACKS[horizon]})"
            )

            for split_name in SPLITS:

                process_task(
                    feature_df=feature_df,
                    asset_class=asset_class,
                    horizon=horizon,
                    split_name=split_name,
                )

        del feature_df

        gc.collect()

    print()
    print()
    print("=" * 80)
    print("LSTM SEQUENCE BUILD COMPLETE")
    print("=" * 80)

    print()
    print("Output directory:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("All sequence datasets were constructed and saved.")
    print("NEXT STEP:")
    print("Independent LSTM sequence audit.")


if __name__ == "__main__":
    main()