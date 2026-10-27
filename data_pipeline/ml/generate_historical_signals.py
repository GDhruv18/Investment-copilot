
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"

LOGISTIC_DIR = (
    ML_ROOT / "selected_models" / "logistic_regression"
)
LSTM_DIR = ML_ROOT / "lstm_results"

OUTPUT_DIR = ML_ROOT / "historical_signals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR / "test_historical_signals.parquet"
)
SUMMARY_PATH = (
    OUTPUT_DIR / "test_historical_signals_summary.csv"
)

CLASS_NAMES = {
    0: "Weak",
    1: "Neutral",
    2: "Attractive",
}

# Locked model selection. Experimental LSTMs are excluded.
MODEL_SELECTION = {
    ("stocks", "6m"): "logistic",
    ("stocks", "1y"): "logistic",
    ("stocks", "2y"): "logistic",
    ("etfs", "6m"): "lstm",
    ("etfs", "1y"): "lstm",
    ("etfs", "2y"): "lstm",
    ("mutual_funds", "6m"): "logistic",
    ("mutual_funds", "1y"): "logistic",
    ("mutual_funds", "2y"): "lstm",
}

# Existing temporal test-split filenames.
DATASET_PREFIXES = {
    "stocks": "stock",
    "etfs": "etf",
    "mutual_funds": "mutual_fund",
}


# ============================================================
# HELPERS
# ============================================================

def resolve_column(df, candidates, description):
    """Find an existing column without silently guessing."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description}. "
        f"Tried {candidates}. Available columns: "
        f"{df.columns.tolist()}"
    )


def normalize_targets(series, asset_class):
    """Map targets to the locked integer class representation."""
    if asset_class in {"stocks", "etfs"}:
        mapping = {
            "Weak": 0,
            "Neutral": 1,
            "Attractive": 2,
            "0": 0,
            "1": 1,
            "2": 2,
        }

        normalized = series.astype(str).map(mapping)
    else:
        normalized = pd.to_numeric(
            series,
            errors="coerce",
        )

    if normalized.isna().any():
        invalid = (
            series[normalized.isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Invalid targets for {asset_class}: {invalid}"
        )

    normalized = normalized.astype(int)

    if not set(normalized.unique()).issubset({0, 1, 2}):
        raise ValueError(
            f"Unexpected target classes for {asset_class}"
        )

    return normalized


def normalize_identifier(series):
    """Keep identifiers stable as strings."""
    if series.isna().any():
        raise ValueError("Null asset identifiers found.")

    return series.astype(str)


def validate_probabilities(probabilities, context):
    if probabilities.ndim != 2:
        raise ValueError(
            f"{context}: probabilities must be a 2D array."
        )

    if probabilities.shape[1] != 3:
        raise ValueError(
            f"{context}: expected 3 classes, "
            f"got shape {probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{context}: probabilities contain NaN or infinity."
        )

    if (probabilities < -1e-7).any() or (
        probabilities > 1.0 + 1e-7
    ).any():
        raise ValueError(
            f"{context}: probabilities are outside [0, 1]."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            f"{context}: probabilities do not sum to 1."
        )


def get_test_dataset(asset_class, horizon):
    prefix = DATASET_PREFIXES[asset_class]

    path = (
        ML_ROOT
        / asset_class
        / f"{prefix}_{horizon}_test.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    return pd.read_parquet(path), path


# ============================================================
# LOGISTIC REGRESSION SIGNALS
# ============================================================

def generate_logistic_signals(asset_class, horizon):
    model_name = f"{asset_class}_{horizon}_logistic"
    model_path = LOGISTIC_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected Logistic Regression model missing: "
            f"{model_path}"
        )

    test_df, test_path = get_test_dataset(
        asset_class,
        horizon,
    )

    # Confirm the test dataset has identifiers, dates, and labels.
    identifier_col = resolve_column(
        test_df,
        ["ticker", "scheme_code"],
        "asset identifier",
    )
    date_col = resolve_column(
        test_df,
        ["date", "prediction_date"],
        "prediction date",
    )
    target_col = resolve_column(
        test_df,
        ["target"],
        "actual target",
    )

    metadata_col = (
        "label_end_date"
        if "label_end_date" in test_df.columns
        else None
    )

    # Load the complete saved pipeline.
    model = joblib.load(model_path)

    expected_features = list(
        model.feature_names_in_
    ) if hasattr(model, "feature_names_in_") else None

    if expected_features is None:
        raise ValueError(
            f"{model_path} does not expose feature names. "
            "Cannot safely validate feature order."
        )

    missing_features = [
        column
        for column in expected_features
        if column not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing test features: {missing_features}"
        )

    X_test = test_df[expected_features].copy()

    # No fitting occurs here. Preprocessing uses saved train parameters.
    probabilities = model.predict_proba(X_test)

    fitted_classes = model.named_steps["model"].classes_

    if list(fitted_classes) != [0, 1, 2]:
        raise ValueError(
            f"Unexpected class order in {model_path}: "
            f"{list(fitted_classes)}"
        )

    validate_probabilities(
        probabilities,
        f"{asset_class}/{horizon}/Logistic",
    )

    actual_target = normalize_targets(
        test_df[target_col],
        asset_class,
    )

    output = pd.DataFrame({
        "ticker": normalize_identifier(
            test_df[identifier_col]
        ).to_numpy(),
        "prediction_date": pd.to_datetime(
            test_df[date_col]
        ).to_numpy(),
        "asset_class": asset_class,
        "horizon": horizon,
        "p_weak": probabilities[:, 0],
        "p_neutral": probabilities[:, 1],
        "p_attractive": probabilities[:, 2],
        "predicted_class": np.argmax(
            probabilities, axis=1
        ),
        "actual_target": actual_target.to_numpy(),
        "model_used": "LogisticRegression",
        "split": "test",
    })

    if metadata_col:
        output["label_end_date"] = pd.to_datetime(
            test_df[metadata_col]
        ).to_numpy()
    else:
        output["label_end_date"] = pd.NaT

    output["predicted_label"] = (
        output["predicted_class"].map(CLASS_NAMES)
    )
    output["actual_label"] = (
        output["actual_target"].map(CLASS_NAMES)
    )

    print(
        f"  Logistic | {asset_class:12s} | {horizon:2s} | "
        f"{len(output):,} rows"
    )

    return output


# ============================================================
# LSTM SIGNALS
# ============================================================

def generate_lstm_signals(asset_class, horizon):
    model_path = (
        LSTM_DIR
        / asset_class
        / horizon
        / "best_model.keras"
    )

    sequence_dir = (
        ML_ROOT
        / "sequences"
        / asset_class
        / horizon
    )

    x_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_test_X.npy"
    )
    y_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_test_y.npy"
    )
    metadata_path = (
        sequence_dir
        / f"{asset_class}_{horizon}_test_metadata.parquet"
    )

    for path in (
        model_path,
        x_path,
        y_path,
        metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required LSTM artifact missing: {path}"
            )

    # Memory-map the sequences to reduce memory pressure.
    X_test = np.load(
        x_path,
        mmap_mode="r",
    )
    y_test = np.load(y_path)

    metadata = pd.read_parquet(metadata_path)

    if len(X_test) != len(y_test) or (
        len(X_test) != len(metadata)
    ):
        raise ValueError(
            f"{asset_class}/{horizon}: sequence, label, and "
            "metadata row counts do not match."
        )

    if X_test.ndim != 3 or X_test.shape[2] != 27:
        raise ValueError(
            f"Unexpected LSTM input shape: {X_test.shape}. "
            "Expected (samples, lookback, 27)."
        )

    identifier_col = resolve_column(
        metadata,
        ["ticker", "scheme_code"],
        "LSTM metadata asset identifier",
    )
    date_col = resolve_column(
        metadata,
        ["date", "prediction_date"],
        "LSTM metadata prediction date",
    )
    target_col = resolve_column(
        metadata,
        ["target"],
        "LSTM metadata actual target",
    )

    label_end_col = (
        "label_end_date"
        if "label_end_date" in metadata.columns
        else None
    )

    metadata_target = normalize_targets(
        metadata[target_col],
        asset_class,
    )

    # Check that the sequence labels agree with their metadata.
    sequence_target = normalize_targets(
        pd.Series(y_test),
        asset_class,
    )

    if not np.array_equal(
        sequence_target.to_numpy(),
        metadata_target.to_numpy(),
    ):
        raise ValueError(
            f"{asset_class}/{horizon}: sequence labels do not "
            "match metadata targets. Refusing to generate signals."
        )

    # Load the locked original LSTM model.
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if model.input_shape[-1] != 27:
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    # The locked LSTM pipeline uses the stored sequences directly;
    # do not introduce a new scaler here.
    probabilities = model.predict(
        X_test,
        batch_size=256,
        verbose=0,
    )

    validate_probabilities(
        probabilities,
        f"{asset_class}/{horizon}/LSTM",
    )

    output = pd.DataFrame({
        "ticker": normalize_identifier(
            metadata[identifier_col]
        ).to_numpy(),
        "prediction_date": pd.to_datetime(
            metadata[date_col]
        ).to_numpy(),
        "asset_class": asset_class,
        "horizon": horizon,
        "p_weak": probabilities[:, 0],
        "p_neutral": probabilities[:, 1],
        "p_attractive": probabilities[:, 2],
        "predicted_class": np.argmax(
            probabilities, axis=1
        ),
        "actual_target": metadata_target.to_numpy(),
        "model_used": "LSTM",
        "split": "test",
    })

    if label_end_col:
        output["label_end_date"] = pd.to_datetime(
            metadata[label_end_col]
        ).to_numpy()
    else:
        output["label_end_date"] = pd.NaT

    output["predicted_label"] = (
        output["predicted_class"].map(CLASS_NAMES)
    )
    output["actual_label"] = (
        output["actual_target"].map(CLASS_NAMES)
    )

    print(
        f"  LSTM     | {asset_class:12s} | {horizon:2s} | "
        f"{len(output):,} rows"
    )

    return output


# ============================================================
# AUDIT
# ============================================================

def audit_signals(signals):
    required_columns = [
        "ticker",
        "prediction_date",
        "asset_class",
        "horizon",
        "p_weak",
        "p_neutral",
        "p_attractive",
        "predicted_class",
        "actual_target",
        "model_used",
        "split",
        "label_end_date",
    ]

    missing = [
        column
        for column in required_columns
        if column not in signals.columns
    ]

    if missing:
        raise ValueError(
            f"Output is missing columns: {missing}"
        )

    if signals[required_columns].isna().any().any():
        raise ValueError(
            "Required output columns contain missing values."
        )

    if signals.duplicated(
        ["asset_class", "horizon", "ticker", "prediction_date"]
    ).any():
        raise ValueError(
            "Duplicate asset/date/horizon signals found."
        )

    if not (signals["split"] == "test").all():
        raise ValueError(
            "Non-test records found in the test signal output."
        )

    if not signals["predicted_class"].isin([0, 1, 2]).all():
        raise ValueError("Invalid predicted class.")

    if not signals["actual_target"].isin([0, 1, 2]).all():
        raise ValueError("Invalid actual target.")

    probability_sums = signals[
        ["p_weak", "p_neutral", "p_attractive"]
    ].sum(axis=1)

    if not np.allclose(
        probability_sums.to_numpy(),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Some output probabilities do not sum to one."
        )

    # Check that each combination uses its locked model.
    for (asset_class, horizon), expected_model in (
        MODEL_SELECTION.items()
    ):
        subset = signals[
            (signals["asset_class"] == asset_class)
            & (signals["horizon"] == horizon)
        ]

        if subset.empty:
            raise ValueError(
                f"No signals generated for "
                f"{asset_class}/{horizon}."
            )

        expected_name = (
            "LogisticRegression"
            if expected_model == "logistic"
            else "LSTM"
        )

        if set(subset["model_used"].unique()) != {
            expected_name
        }:
            raise ValueError(
                f"Incorrect model used for "
                f"{asset_class}/{horizon}."
            )

    print("\nAUDIT: PASS")
    print(f"Total signal rows : {len(signals):,}")
    print(
        "Unique assets     : "
        f"{signals['ticker'].nunique():,}"
    )
    print(
        "Prediction dates  : "
        f"{signals['prediction_date'].min()} to "
        f"{signals['prediction_date'].max()}"
    )
    print("Duplicate signals : 0")
    print("Missing required values: 0")
    print("Probability checks: PASS")
    print("Model-selection checks: PASS")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("GENERATE HISTORICAL OUT-OF-SAMPLE SIGNALS")
    print("=" * 76)

    print("Prediction split: TEST ONLY")
    print("No model fitting or model selection will occur.")
    print("No source datasets will be modified.\n")

    all_signals = []

    for (asset_class, horizon), model_type in (
        MODEL_SELECTION.items()
    ):
        if model_type == "logistic":
            signals = generate_logistic_signals(
                asset_class,
                horizon,
            )
        elif model_type == "lstm":
            signals = generate_lstm_signals(
                asset_class,
                horizon,
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}"
            )

        all_signals.append(signals)

    combined = pd.concat(
        all_signals,
        ignore_index=True,
    )

    combined["prediction_date"] = pd.to_datetime(
        combined["prediction_date"]
    )
    combined["label_end_date"] = pd.to_datetime(
        combined["label_end_date"]
    )

    combined = combined.sort_values(
        [
            "asset_class",
            "horizon",
            "ticker",
            "prediction_date",
        ]
    ).reset_index(drop=True)

    audit_signals(combined)

    combined.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary = (
        combined.groupby(
            ["asset_class", "horizon", "model_used"],
            as_index=False,
        )
        .agg(
            rows=("ticker", "size"),
            unique_assets=("ticker", "nunique"),
            first_prediction_date=("prediction_date", "min"),
            last_prediction_date=("prediction_date", "max"),
            actual_weak=("actual_target", lambda s: (s == 0).sum()),
            actual_neutral=("actual_target", lambda s: (s == 1).sum()),
            actual_attractive=("actual_target", lambda s: (s == 2).sum()),
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nSignal summary:")
    print(summary.to_string(index=False))

    print(f"\nSignals saved : {OUTPUT_PATH}")
    print(f"Summary saved : {SUMMARY_PATH}")
    print("\nNEXT STEP: independently audit these generated signals.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
