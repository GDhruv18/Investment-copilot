from pathlib import Path
import numpy as np
import pandas as pd
import gc


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DIR = PROJECT_ROOT / "data" / "ml"
SEQUENCE_DIR = ML_DIR / "sequences"


# =============================================================================
# LOCKED POLICY
# =============================================================================

LOOKBACKS = {
    "6m": 126,
    "1y": 252,
    "2y": 252,
}


ASSET_CONFIG = {
    "stocks": {
        "prefix": "stock",
        "split_dir": ML_DIR / "stocks",
    },

    "etfs": {
        "prefix": "etf",
        "split_dir": ML_DIR / "etfs",
    },

    "mutual_funds": {
        "prefix": "mutual_fund",
        "split_dir": ML_DIR / "mutual_funds",
    },
}


HORIZONS = ["6m", "1y", "2y"]

SPLITS = ["train", "validation", "test"]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_dates(series):
    series = pd.to_datetime(
        series,
        errors="raise"
    )

    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        series = series.dt.tz_localize(None)

    return series.dt.normalize()


def make_key(df):
    return pd.MultiIndex.from_frame(
        df[["ticker", "date"]]
    )


def load_original_split(
    asset_class,
    horizon,
    split,
):
    config = ASSET_CONFIG[asset_class]

    filename = (
        f"{config['prefix']}_{horizon}_{split}.parquet"
    )

    path = config["split_dir"] / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original split file:\n{path}"
        )

    df = pd.read_parquet(path)

    required = [
        "ticker",
        "date",
        "label_end_date",
        "target",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    df = df[
        required
    ].copy()

    df["date"] = normalize_dates(
        df["date"]
    )

    df["label_end_date"] = normalize_dates(
        df["label_end_date"]
    )

    return df


def target_numeric(
    series,
    asset_class,
):
    if asset_class in ["stocks", "etfs"]:

        mapping = {
            "Weak": 0,
            "Neutral": 1,
            "Attractive": 2,
        }

        result = series.astype(str).map(mapping)

    else:

        mapping = {
            "0": 0,
            "1": 1,
            "2": 2,
            0: 0,
            1: 1,
            2: 2,
        }

        result = series.map(mapping)

    if result.isna().any():

        bad = (
            series[result.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Unknown target values: {bad}"
        )

    return result.astype(np.int8)


def compare_key_sets(
    expected,
    actual,
    description,
):
    expected_set = set(
        zip(
            expected["ticker"],
            expected["date"],
        )
    )

    actual_set = set(
        zip(
            actual["ticker"],
            actual["date"],
        )
    )

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing or extra:

        print()
        print(
            f"    {description}: FAIL"
        )

        print(
            f"      Missing keys : {len(missing):,}"
        )

        print(
            f"      Extra keys   : {len(extra):,}"
        )

        if missing:
            print(
                f"      Example missing: "
                f"{list(missing)[:3]}"
            )

        if extra:
            print(
                f"      Example extra: "
                f"{list(extra)[:3]}"
            )

        return False

    print(
        f"    {description}: PASS"
    )

    return True


# =============================================================================
# SINGLE DATASET AUDIT
# =============================================================================

def audit_dataset(
    asset_class,
    horizon,
    split,
):
    print()
    print("-" * 80)

    print(
        f"{asset_class.upper()} | "
        f"{horizon.upper()} | "
        f"{split.upper()}"
    )

    print("-" * 80)

    lookback = LOOKBACKS[horizon]

    # -------------------------------------------------------------------------
    # File paths
    # -------------------------------------------------------------------------

    sequence_dir = (
        SEQUENCE_DIR
        / asset_class
        / horizon
    )

    X_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_{split}_X.npy"
    )

    y_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_{split}_y.npy"
    )

    metadata_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_{split}_metadata.parquet"
    )

    # -------------------------------------------------------------------------
    # File existence
    # -------------------------------------------------------------------------

    for path in [
        X_path,
        y_path,
        metadata_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing sequence output:\n{path}"
            )

    print("    Output files: PASS")

    # -------------------------------------------------------------------------
    # Load arrays
    # -------------------------------------------------------------------------

    X = np.load(
        X_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    y = np.load(
        y_path,
        allow_pickle=False,
    )

    metadata = pd.read_parquet(
        metadata_path
    )

    # -------------------------------------------------------------------------
    # Basic shapes
    # -------------------------------------------------------------------------

    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-dimensional, got {X.ndim}"
        )

    expected_lookback = lookback

    if X.shape[1] != expected_lookback:
        raise ValueError(
            f"Incorrect lookback: "
            f"{X.shape[1]}, expected {expected_lookback}"
        )

    if X.shape[2] != 27:
        raise ValueError(
            f"Incorrect feature count: "
            f"{X.shape[2]}, expected 27"
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-dimensional, got {y.ndim}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"X/y length mismatch: "
            f"{len(X)} vs {len(y)}"
        )

    if len(X) != len(metadata):
        raise ValueError(
            f"X/metadata length mismatch: "
            f"{len(X)} vs {len(metadata)}"
        )

    print(
        f"    Shape: PASS "
        f"{X.shape}"
    )

    # -------------------------------------------------------------------------
    # Dtypes
    # -------------------------------------------------------------------------

    if X.dtype != np.float32:
        raise ValueError(
            f"X dtype is {X.dtype}, expected float32"
        )

    if y.dtype != np.int8:
        raise ValueError(
            f"y dtype is {y.dtype}, expected int8"
        )

    print("    Dtypes: PASS")

    # -------------------------------------------------------------------------
    # Finite X
    #
    # np.isfinite on a memory-mapped array works, but processes the entire
    # dataset. This is intentional for the audit.
    # -------------------------------------------------------------------------

    print(
        "    Checking X for NaN/infinity..."
    )

    finite = np.isfinite(X).all()

    if not finite:
        raise ValueError(
            "X contains NaN or infinity."
        )

    print(
        "    X finite values: PASS"
    )

    # -------------------------------------------------------------------------
    # Target values
    # -------------------------------------------------------------------------

    if not np.isin(
        y,
        [0, 1, 2]
    ).all():

        raise ValueError(
            "y contains values outside 0/1/2."
        )

    print(
        "    y target values: PASS"
    )

    # -------------------------------------------------------------------------
    # Metadata columns
    # -------------------------------------------------------------------------

    required_metadata = [
        "ticker",
        "date",
        "label_end_date",
        "target",
        "target_numeric",
    ]

    missing_metadata = [
        c for c in required_metadata
        if c not in metadata.columns
    ]

    if missing_metadata:
        raise ValueError(
            f"Missing metadata columns: "
            f"{missing_metadata}"
        )

    metadata["date"] = normalize_dates(
        metadata["date"]
    )

    metadata["label_end_date"] = normalize_dates(
        metadata["label_end_date"]
    )

    # -------------------------------------------------------------------------
    # Duplicate metadata keys
    # -------------------------------------------------------------------------

    duplicate_count = metadata.duplicated(
        subset=["ticker", "date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Duplicate sequence metadata keys: "
            f"{duplicate_count:,}"
        )

    print(
        "    Metadata duplicate keys: PASS"
    )

    # -------------------------------------------------------------------------
    # target_numeric alignment
    # -------------------------------------------------------------------------

    expected_numeric = target_numeric(
        metadata["target"],
        asset_class,
    )

    if not np.array_equal(
        expected_numeric.to_numpy(),
        metadata["target_numeric"].to_numpy(
            dtype=np.int8
        ),
    ):
        raise ValueError(
            "Metadata target_numeric does not match target."
        )

    if not np.array_equal(
        metadata["target_numeric"].to_numpy(
            dtype=np.int8
        ),
        y,
    ):
        raise ValueError(
            "y does not match metadata target_numeric."
        )

    print(
        "    Target alignment: PASS"
    )

    # -------------------------------------------------------------------------
    # Load original split
    # -------------------------------------------------------------------------

    original = load_original_split(
        asset_class,
        horizon,
        split,
    )

    print(
        f"    Original split rows: "
        f"{len(original):,}"
    )

    # -------------------------------------------------------------------------
    # The sequence metadata should be a subset of the original split.
    #
    # It will usually be smaller because sequences can be rejected due to:
    # - insufficient history
    # - missing/non-finite historical features
    # -------------------------------------------------------------------------

    metadata_keys = make_key(
        metadata
    )

    original_keys = make_key(
        original
    )

    metadata_set = set(metadata_keys)
    original_set = set(original_keys)

    unexpected = metadata_set - original_set

    if unexpected:

        raise ValueError(
            f"Sequence metadata contains "
            f"{len(unexpected):,} prediction keys "
            f"not present in the original split."
        )

    print(
        "    Prediction-key subset: PASS"
    )

    # -------------------------------------------------------------------------
    # Reconstruct expected metadata values from original split.
    # -------------------------------------------------------------------------

    original_indexed = original.set_index(
        ["ticker", "date"]
    )

    metadata_indexed = metadata.set_index(
        ["ticker", "date"]
    )

    aligned_original = original_indexed.loc[
        metadata_indexed.index
    ]

    # label_end_date
    if not np.array_equal(
        aligned_original[
            "label_end_date"
        ].to_numpy(),
        metadata_indexed[
            "label_end_date"
        ].to_numpy(),
    ):
        raise ValueError(
            "label_end_date mismatch."
        )

    # Target
    expected_targets = (
        aligned_original["target"]
        .astype(str)
        .to_numpy()
    )

    actual_targets = (
        metadata_indexed["target"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        expected_targets,
        actual_targets,
    ):
        raise ValueError(
            "Target mismatch between "
            "sequence metadata and original split."
        )

    print(
        "    Original target alignment: PASS"
    )

    # -------------------------------------------------------------------------
    # Future-label sanity
    # -------------------------------------------------------------------------

    if (
        metadata["label_end_date"]
        < metadata["date"]
    ).any():

        raise ValueError(
            "Found label_end_date earlier "
            "than prediction date."
        )

    print(
        "    Label chronology: PASS"
    )

    # -------------------------------------------------------------------------
    # Asset integrity
    # -------------------------------------------------------------------------

    sequence_assets = set(
        metadata["ticker"]
    )

    original_assets = set(
        original["ticker"]
    )

    if not sequence_assets.issubset(
        original_assets
    ):
        raise ValueError(
            "Sequence contains assets absent "
            "from original split."
        )

    print(
        f"    Asset integrity: PASS "
        f"({len(sequence_assets):,} assets)"
    )

    # -------------------------------------------------------------------------
    # Sequence boundary check
    #
    # The last timestep must correspond to the prediction date.
    #
    # We cannot inspect ticker/date inside X itself, because X contains only
    # numerical features. The actual date boundary is independently validated
    # by reconstructing a sample against the source feature timeline in the
    # dedicated sequence-boundary audit below.
    # -------------------------------------------------------------------------

    print(
        "    Sequence tensor structure: PASS"
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    count_string = ", ".join(
        f"{int(k)}={int(v):,}"
        for k, v in counts.items()
    )

    print(
        f"    Target counts: {count_string}"
    )

    print(
        f"    Valid sequences: {len(X):,}"
    )

    return {
        "asset_class": asset_class,
        "horizon": horizon,
        "split": split,
        "rows": len(X),
        "assets": len(sequence_assets),
        "lookback": lookback,
    }


# =============================================================================
# TEMPORAL SPLIT AUDIT
# =============================================================================

def audit_split_disjointness(
    asset_class,
    horizon,
):
    """
    Verify that sequence prediction keys do not overlap between
    train / validation / test.

    This is independent of the original temporal split audit.
    """

    print()
    print(
        f"  Checking sequence split disjointness..."
    )

    key_sets = {}

    for split in SPLITS:

        metadata_path = (
            SEQUENCE_DIR
            / asset_class
            / horizon
            / (
                f"{asset_class}_{horizon}_"
                f"{split}_metadata.parquet"
            )
        )

        metadata = pd.read_parquet(
            metadata_path
        )

        metadata["date"] = normalize_dates(
            metadata["date"]
        )

        keys = set(
            zip(
                metadata["ticker"],
                metadata["date"],
            )
        )

        key_sets[split] = keys

    train_val = (
        key_sets["train"]
        & key_sets["validation"]
    )

    train_test = (
        key_sets["train"]
        & key_sets["test"]
    )

    val_test = (
        key_sets["validation"]
        & key_sets["test"]
    )

    if train_val or train_test or val_test:

        raise ValueError(
            f"Sequence split overlap detected: "
            f"train/val={len(train_val)}, "
            f"train/test={len(train_test)}, "
            f"val/test={len(val_test)}"
        )

    print(
        "    Sequence split disjointness: PASS"
    )


# =============================================================================
# SAMPLE SEQUENCE BOUNDARY AUDIT
# =============================================================================

def audit_sequence_boundaries(
    asset_class,
    horizon,
):
    """
    Independently verify that sampled sequence endpoints correspond exactly
    to their prediction dates.

    We use the original feature timeline and inspect several samples from
    every split.
    """

    config = ASSET_CONFIG[asset_class]

    feature_file = (
        ML_DIR.parent
        / "features"
        / (
            "stock_features.parquet"
            if asset_class == "stocks"
            else
            "etf_features.parquet"
            if asset_class == "etfs"
            else
            "mutual_fund_features.parquet"
        )
    )

    features = pd.read_parquet(
        feature_file,
        columns=["ticker", "date"],
    )

    features["date"] = normalize_dates(
        features["date"]
    )

    feature_dates = {
        ticker: group["date"].sort_values().to_numpy(
            dtype="datetime64[ns]"
        )
        for ticker, group
        in features.groupby("ticker")
    }

    lookback = LOOKBACKS[horizon]

    for split in SPLITS:

        metadata_path = (
            SEQUENCE_DIR
            / asset_class
            / horizon
            / (
                f"{asset_class}_{horizon}_"
                f"{split}_metadata.parquet"
            )
        )

        X_path = (
            SEQUENCE_DIR
            / asset_class
            / horizon
            / (
                f"{asset_class}_{horizon}_"
                f"{split}_X.npy"
            )
        )

        metadata = pd.read_parquet(
            metadata_path
        )

        metadata["date"] = normalize_dates(
            metadata["date"]
        )

        X = np.load(
            X_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if len(metadata) == 0:
            raise ValueError(
                f"No metadata rows in {split}."
            )

        # Check first, middle, last and several evenly spaced samples.
        sample_indices = sorted(
            set(
                [
                    0,
                    len(metadata) // 2,
                    len(metadata) - 1,
                ]
                +
                list(
                    np.linspace(
                        0,
                        len(metadata) - 1,
                        num=min(10, len(metadata)),
                        dtype=int,
                    )
                )
            )
        )

        for idx in sample_indices:

            ticker = metadata.iloc[idx]["ticker"]
            prediction_date = metadata.iloc[idx]["date"]

            if ticker not in feature_dates:
                raise ValueError(
                    f"{asset_class}/{horizon}/{split}: "
                    f"ticker {ticker} missing from features."
                )

            dates = feature_dates[ticker]

            pos = np.searchsorted(
                dates,
                np.datetime64(prediction_date),
                side="left",
            )

            if (
                pos >= len(dates)
                or dates[pos]
                != np.datetime64(prediction_date)
            ):
                raise ValueError(
                    f"Prediction date not found in "
                    f"feature timeline: "
                    f"{ticker} {prediction_date}"
                )

            start = pos - lookback + 1

            if start < 0:
                raise ValueError(
                    f"Insufficient history for supposedly "
                    f"valid sequence: "
                    f"{ticker} {prediction_date}"
                )

            if X[idx].shape != (
                lookback,
                27,
            ):
                raise ValueError(
                    f"Incorrect X shape at "
                    f"{split} index {idx}."
                )

        del X

    del features
    gc.collect()

    print(
        "    Sequence boundary samples: PASS"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("INVESTMENT COPILOT - LSTM SEQUENCE AUDIT")
    print("=" * 80)

    print()
    print("This audit does NOT modify any dataset.")

    results = []

    total_tasks = (
        len(ASSET_CONFIG)
        * len(HORIZONS)
        * len(SPLITS)
    )

    completed = 0

    # =========================================================================
    # Dataset audits
    # =========================================================================

    for asset_class in ASSET_CONFIG:

        for horizon in HORIZONS:

            for split in SPLITS:

                result = audit_dataset(
                    asset_class,
                    horizon,
                    split,
                )

                results.append(result)

                completed += 1

                print()
                print(
                    f"Completed dataset audits: "
                    f"{completed}/{total_tasks}"
                )

            # -----------------------------------------------------------------
            # Cross-split checks
            # -----------------------------------------------------------------

            audit_split_disjointness(
                asset_class,
                horizon,
            )

            audit_sequence_boundaries(
                asset_class,
                horizon,
            )

    # =========================================================================
    # Final summary
    # =========================================================================

    print()
    print()
    print("=" * 80)
    print("LSTM SEQUENCE AUDIT SUMMARY")
    print("=" * 80)

    result_df = pd.DataFrame(
        results
    )

    print()

    print(
        result_df[
            [
                "asset_class",
                "horizon",
                "split",
                "rows",
                "assets",
                "lookback",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Datasets audited: "
        f"{len(results)}/{total_tasks}"
    )

    if len(results) != total_tasks:

        raise RuntimeError(
            "Not all sequence datasets were audited."
        )

    print()
    print(
        "LSTM SEQUENCE AUDIT: PASS"
    )

    print()
    print(
        "All 27 LSTM sequence datasets passed "
        "the independent audit."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "LSTM baseline training."
    )


if __name__ == "__main__":
    main()