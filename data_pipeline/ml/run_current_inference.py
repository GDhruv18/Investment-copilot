import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import os
import psycopg2
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data" / "features"
ML_DIR = ROOT / "data" / "ml"
MODEL_DIR = ML_DIR / "selected_models" / "logistic_regression"
OUTPUT_DIR = ML_DIR / "current_signals"

LABELS = {0: "Weak", 1: "Neutral", 2: "Attractive"}
LOOKBACKS = {"6m": 126, "1y": 252, "2y": 252}

# Preserve the previously selected model for each asset class and horizon.
ROUTES = {
    ("stocks", "6m"): ("logistic", "stocks_6m_logistic.joblib"),
    ("stocks", "1y"): ("logistic", "stocks_1y_logistic.joblib"),
    ("stocks", "2y"): ("logistic", "stocks_2y_logistic.joblib"),
    ("etfs", "6m"): ("lstm", "etfs/6m/best_model.keras"),
    ("etfs", "1y"): ("lstm", "etfs/1y/best_model.keras"),
    ("etfs", "2y"): ("lstm", "etfs/2y/best_model.keras"),
    ("mutual_funds", "6m"): ("logistic", "mutual_funds_6m_logistic.joblib"),
    ("mutual_funds", "1y"): ("logistic", "mutual_funds_1y_logistic.joblib"),
    ("mutual_funds", "2y"): ("logistic", "mutual_funds_2y_logistic.joblib"),
}

FEATURE_FILES = {
    "stocks": FEATURE_DIR / "stock_features.parquet",
    "etfs": FEATURE_DIR / "etf_features.parquet",
    "mutual_funds": FEATURE_DIR / "mutual_fund_features.parquet",
}
def get_db_connection():
    load_dotenv(ROOT / ".env")

    required = [
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
    ]

    missing = [
        variable
        for variable in required
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            "Missing database environment variables: "
            + ", ".join(missing)
        )

    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
def load_ineligible_mutual_fund_schemes():
    query = """
        SELECT scheme_code
        FROM mutual_fund_scheme_status
        WHERE recommendation_eligible = FALSE;
    """

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        conn.close()

    return {str(row[0]) for row in rows}

def load_feature_data(asset_class):
    path = FEATURE_FILES[asset_class]
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    df = pd.read_parquet(path)
    required = {"ticker", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if df.duplicated(["ticker", "date"]).any():
        raise ValueError(
            f"{path.name} contains duplicate ticker/date rows; "
            "resolve duplicates before inference."
        )
    return df


def get_lstm_model(asset_class, horizon, relative_path):
    model_directory = "lstm_results02" if horizon == "2y" else "lstm_results"
    model_path = ML_DIR / model_directory / relative_path
    if not model_path.exists():
        raise FileNotFoundError(f"Selected LSTM artifact not found: {model_path}")

    # Import TensorFlow only when an LSTM is actually required.
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)

    expected = (LOOKBACKS[horizon], 27)
    model_shape = tuple(model.input_shape[1:])
    if model_shape != expected:
        raise ValueError(
            f"{asset_class}/{horizon}: expected model input {expected}, "
            f"but artifact expects {model_shape}"
        )
    return model


def predict_logistic(asset_class, horizon, df, model_filename):
    metadata_path = MODEL_DIR / model_filename.replace(".joblib", "_metadata.json")
    model_path = MODEL_DIR / model_filename

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = metadata["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{asset_class}/{horizon} missing features: {missing}")

    latest = df.groupby("ticker", sort=False).tail(1).copy()
    X = latest[features].apply(pd.to_numeric, errors="coerce")
    model = joblib.load(model_path)

    probabilities = model.predict_proba(X)
    classes = model.predict(X)

    return [
        {
            "asset_class": asset_class,
            "ticker": row["ticker"],
            "feature_date": row["date"].date().isoformat(),
            "horizon": horizon,
            "model_type": "LogisticRegression",
            "prediction": LABELS[int(pred)],
            "predicted_class": int(pred),
            "prob_weak": float(prob[0]),
            "prob_neutral": float(prob[1]),
            "prob_attractive": float(prob[2]),
            "status": "MODEL_SIGNAL_NOT_FINANCIAL_ADVICE",
        }
        for (_, row), pred, prob in zip(latest.iterrows(), classes, probabilities)
    ]


def predict_lstm(asset_class, horizon, df, relative_path):
    import runpy

    sequence_builder = ROOT / "data_pipeline" / "ml" / "build_lstm_sequences.py"
    config_namespace = runpy.run_path(str(sequence_builder))
    asset_config = config_namespace["ASSET_CONFIG"]
    features = asset_config[asset_class]["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{asset_class}/{horizon} missing features: {missing}")

    lookback = LOOKBACKS[horizon]
    model = get_lstm_model(asset_class, horizon, relative_path)
    output = []

    # The sequence builder's established policy requires complete, finite
    # windows; do not silently impute or apply the diagnostic-only scaler.
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date")
        if len(group) < lookback:
            continue

        values = group[features].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )
        sequence = values[-lookback:]

        if sequence.shape != (lookback, 27) or not np.isfinite(sequence).all():
            continue

        probabilities = np.asarray(
            model.predict(sequence[np.newaxis, :, :], verbose=0)
        )[0]
        if len(probabilities) != 3 or not np.isfinite(probabilities).all():
            continue

        pred = int(np.argmax(probabilities))
        output.append({
            "asset_class": asset_class,
            "ticker": ticker,
            "feature_date": group.iloc[-1]["date"].date().isoformat(),
            "horizon": horizon,
            "model_type": "LSTM",
            "prediction": LABELS[pred],
            "predicted_class": pred,
            "prob_weak": float(probabilities[0]),
            "prob_neutral": float(probabilities[1]),
            "prob_attractive": float(probabilities[2]),
            "status": "MODEL_SIGNAL_NOT_FINANCIAL_ADVICE",
        })

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_signals = []
    errors = []

    for asset_class in ("stocks", "etfs", "mutual_funds"):
        print(f"\nLoading {asset_class} features...")
        try:
            df = load_feature_data(asset_class)
            print(f"Rows: {len(df):,}; latest date: {df['date'].max().date()}")

            for horizon in ("6m", "1y", "2y"):
                model_type, artifact = ROUTES[(asset_class, horizon)]
                print(f"Predicting {asset_class}/{horizon} with {model_type}...")
                try:
                    if model_type == "logistic":
                        result = predict_logistic(
                            asset_class, horizon, df, artifact
                        )
                    else:
                        result = predict_lstm(
                            asset_class, horizon, df, artifact
                        )
                    all_signals.extend(result)
                    print(f"  Predictions: {len(result):,}")
                except Exception as exc:
                    errors.append({
                        "asset_class": asset_class,
                        "horizon": horizon,
                        "error": repr(exc),
                    })
                    print(f"  FAILED: {exc}")

        except Exception as exc:
            errors.append({
                "asset_class": asset_class,
                "horizon": "all",
                "error": repr(exc),
            })
            print(f"  FAILED to load dataset: {exc}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if all_signals:
        signals = pd.DataFrame(all_signals)

        # Exclude merged/discontinued mutual funds.
        ineligible_schemes = load_ineligible_mutual_fund_schemes()

        exclusion_mask = (
            signals["asset_class"].eq("mutual_funds")
            & signals["ticker"].astype(str).isin(ineligible_schemes)
        )

        excluded_count = int(exclusion_mask.sum())
        signals = signals.loc[~exclusion_mask].copy()

        print(f"\nIneligible mutual-fund signals excluded: {excluded_count}")

        signals["run_timestamp_utc"] = run_id
        output_path = OUTPUT_DIR / f"model_signals_{run_id}.parquet"
        signals.to_parquet(output_path, index=False)

        csv_path = OUTPUT_DIR / f"model_signals_{run_id}.csv"
        signals.to_csv(csv_path, index=False)

        print("\n=== INFERENCE SUMMARY ===")
        print(f"Signals written: {len(signals):,}")
        print(f"Parquet: {output_path}")
        print(f"CSV:     {csv_path}")
        print("\nCounts by asset class and horizon:")
        print(signals.groupby(["asset_class", "horizon"]).size().to_string())
        print("\nFeature dates:")
        print(signals.groupby("asset_class")["feature_date"].agg(["min", "max"]))
    else:
        print("\nNo predictions were generated.")

    if errors:
        error_path = OUTPUT_DIR / f"inference_errors_{run_id}.json"
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"\nErrors: {len(errors)}; details: {error_path}")
        for item in errors:
            print(item)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
