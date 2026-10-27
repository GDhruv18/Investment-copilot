
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "data" / "ml"

SIGNALS_PATH = (
    ML_ROOT
    / "historical_signals"
    / "test_historical_signals.parquet"
)

REPORT_DIR = ML_ROOT / "historical_signals"
REPORT_PATH = REPORT_DIR / "independent_audit_report.csv"

DATASET_PREFIXES = {
    "stocks": "stock",
    "etfs": "etf",
    "mutual_funds": "mutual_fund",
}

MODEL_SELECTION = {
    ("stocks", "6m"): "LogisticRegression",
    ("stocks", "1y"): "LogisticRegression",
    ("stocks", "2y"): "LogisticRegression",
    ("etfs", "6m"): "LSTM",
    ("etfs", "1y"): "LSTM",
    ("etfs", "2y"): "LSTM",
    ("mutual_funds", "6m"): "LogisticRegression",
    ("mutual_funds", "1y"): "LogisticRegression",
    ("mutual_funds", "2y"): "LSTM",
}

CLASS_MAPPING = {
    "Weak": 0,
    "Neutral": 1,
    "Attractive": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}


# ============================================================
# HELPERS
# ============================================================

def require(condition, message):
    if not condition:
        raise AssertionError(message)


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find {description}. "
        f"Tried {candidates}; columns={df.columns.tolist()}"
    )


def normalize_identifier(series):
    if series.isna().any():
        raise ValueError("Null asset identifier found.")

    return series.astype(str).str.strip()


def normalize_target(series, asset_class):
    if asset_class in {"stocks", "etfs"}:
        result = series.astype(str).map(CLASS_MAPPING)
    else:
        result = pd.to_numeric(series, errors="coerce")

    if result.isna().any():
        raise ValueError(
            f"Unrecognized targets for {asset_class}: "
            f"{series[result.isna()].astype(str).unique().tolist()}"
        )

    result = result.astype(int)

    require(
        set(result.unique()).issubset({0, 1, 2}),
        f"Unexpected target classes for {asset_class}.",
    )

    return result


def load_reference(asset_class, horizon, model_used):
    """
    Logistic signals must match the existing test parquet.
    LSTM signals must match the exact test sequence metadata,
    because the sequence builder may exclude rows without
    complete lookback histories.
    """
    if model_used == "LogisticRegression":
        prefix = DATASET_PREFIXES[asset_class]

        path = (
            ML_ROOT
            / asset_class
            / f"{prefix}_{horizon}_test.parquet"
        )

    elif model_used == "LSTM":
        path = (
            ML_ROOT
            / "sequences"
            / asset_class
            / horizon
            / f"{asset_class}_{horizon}_test_metadata.parquet"
        )

    else:
        raise ValueError(f"Unexpected model: {model_used}")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    identifier_col = find_column(
        df, ["ticker", "scheme_code"], "asset identifier"
    )
    date_col = find_column(
        df, ["date", "prediction_date"], "prediction date"
    )
    target_col = find_column(
        df, ["target"], "target"
    )
    label_end_col = find_column(
        df, ["label_end_date"], "label_end_date"
    )

    reference = pd.DataFrame({
        "ticker": normalize_identifier(df[identifier_col]),
        "prediction_date": pd.to_datetime(
            df[date_col], errors="raise"
        ).dt.normalize(),
        "actual_target_reference": normalize_target(
            df[target_col], asset_class
        ),
        "label_end_date_reference": pd.to_datetime(
            df[label_end_col], errors="raise"
        ).dt.normalize(),
    })

    require(
        not reference.duplicated(
            ["ticker", "prediction_date"]
        ).any(),
        f"Duplicate source keys in {path}",
    )

    return reference, path


# ============================================================
# AUDIT ONE MODEL/HORIZON COMBINATION
# ============================================================

def audit_combination(signals, asset_class, horizon, model_used):
    subset = signals[
        (signals["asset_class"] == asset_class)
        & (signals["horizon"] == horizon)
    ].copy()

    require(
        not subset.empty,
        f"No signals for {asset_class}/{horizon}",
    )

    require(
        set(subset["model_used"].unique()) == {model_used},
        f"Wrong model for {asset_class}/{horizon}",
    )

    reference, source_path = load_reference(
        asset_class,
        horizon,
        model_used,
    )

    subset["ticker"] = normalize_identifier(subset["ticker"])
    subset["prediction_date"] = pd.to_datetime(
        subset["prediction_date"], errors="raise"
    ).dt.normalize()
    subset["label_end_date"] = pd.to_datetime(
        subset["label_end_date"], errors="raise"
    ).dt.normalize()

    keys = ["ticker", "prediction_date"]

    require(
        not subset.duplicated(keys).any(),
        f"Duplicate signals for {asset_class}/{horizon}",
    )

    merged = subset.merge(
        reference,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    missing_predictions = int(
        (merged["_merge"] == "right_only").sum()
    )
    unexpected_predictions = int(
        (merged["_merge"] == "left_only").sum()
    )

    require(
        missing_predictions == 0,
        f"{asset_class}/{horizon}: "
        f"{missing_predictions} expected source rows missing.",
    )
    require(
        unexpected_predictions == 0,
        f"{asset_class}/{horizon}: "
        f"{unexpected_predictions} predictions have no source row.",
    )

    matched = merged[merged["_merge"] == "both"].copy()

    require(
        np.array_equal(
            matched["actual_target"].to_numpy(dtype=int),
            matched["actual_target_reference"].to_numpy(dtype=int),
        ),
        f"Actual targets disagree with source for "
        f"{asset_class}/{horizon}.",
    )

    require(
        (
            matched["label_end_date"].to_numpy()
            == matched["label_end_date_reference"].to_numpy()
        ).all(),
        f"label_end_date mismatch for {asset_class}/{horizon}.",
    )

    require(
        (
            subset["prediction_date"] < subset["label_end_date"]
        ).all(),
        f"Prediction date is not before label_end_date "
        f"for {asset_class}/{horizon}.",
    )

    probability_cols = [
        "p_weak",
        "p_neutral",
        "p_attractive",
    ]

    probabilities = subset[probability_cols].to_numpy(
        dtype=float
    )

    require(
        np.isfinite(probabilities).all(),
        f"Non-finite probabilities for {asset_class}/{horizon}.",
    )
    require(
        ((probabilities >= 0) & (probabilities <= 1)).all(),
        f"Probability outside [0,1] for {asset_class}/{horizon}.",
    )
    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-4,
        ),
        f"Probabilities do not sum to 1 for "
        f"{asset_class}/{horizon}.",
    )

    expected_predictions = np.argmax(probabilities, axis=1)

    require(
        np.array_equal(
            expected_predictions,
            subset["predicted_class"].to_numpy(dtype=int),
        ),
        f"predicted_class does not match maximum probability "
        f"for {asset_class}/{horizon}.",
    )

    require(
        (
            subset["predicted_label"].to_numpy()
            == subset["predicted_class"].map({
                0: "Weak",
                1: "Neutral",
                2: "Attractive",
            }).to_numpy()
        ).all(),
        f"Predicted label mapping incorrect for "
        f"{asset_class}/{horizon}.",
    )

    require(
        (
            subset["actual_label"].to_numpy()
            == subset["actual_target"].map({
                0: "Weak",
                1: "Neutral",
                2: "Attractive",
            }).to_numpy()
        ).all(),
        f"Actual label mapping incorrect for "
        f"{asset_class}/{horizon}.",
    )

    return {
        "asset_class": asset_class,
        "horizon": horizon,
        "model_used": model_used,
        "signal_rows": len(subset),
        "reference_rows": len(reference),
        "unique_assets": subset["ticker"].nunique(),
        "missing_predictions": missing_predictions,
        "unexpected_predictions": unexpected_predictions,
        "duplicate_keys": 0,
        "target_match": True,
        "label_end_date_match": True,
        "probability_checks": True,
        "predicted_class_check": True,
        "temporal_order_check": True,
        "source_path": str(
            source_path.relative_to(PROJECT_ROOT)
        ),
        "status": "PASS",
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("INDEPENDENT HISTORICAL SIGNAL AUDIT")
    print("=" * 76)

    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Signal file not found: {SIGNALS_PATH}"
        )

    signals = pd.read_parquet(SIGNALS_PATH)

    required_columns = [
        "ticker",
        "prediction_date",
        "asset_class",
        "horizon",
        "p_weak",
        "p_neutral",
        "p_attractive",
        "predicted_class",
        "predicted_label",
        "actual_target",
        "actual_label",
        "label_end_date",
        "model_used",
        "split",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in signals.columns
    ]

    require(
        not missing_columns,
        f"Missing signal columns: {missing_columns}",
    )

    require(
        not signals[required_columns].isna().any().any(),
        "Missing values found in required signal columns.",
    )

    require(
        set(signals["split"].unique()) == {"test"},
        "Signal file contains non-test rows.",
    )

    require(
        set(signals["asset_class"].unique())
        == {"stocks", "etfs", "mutual_funds"},
        "Unexpected asset classes.",
    )

    print(f"Loaded signal rows: {len(signals):,}\n")

    results = []

    for (asset_class, horizon), model_used in (
        MODEL_SELECTION.items()
    ):
        print(
            f"Auditing {asset_class:12s} | "
            f"{horizon:2s} | {model_used}"
        )

        result = audit_combination(
            signals,
            asset_class,
            horizon,
            model_used,
        )

        results.append(result)

        print(
            f"  PASS | {result['signal_rows']:,} signals "
            f"| {result['unique_assets']} assets"
        )

    report = pd.DataFrame(results)
    report.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 76)
    print("INDEPENDENT AUDIT SUMMARY")
    print("=" * 76)

    print(
        report[
            [
                "asset_class",
                "horizon",
                "signal_rows",
                "reference_rows",
                "unique_assets",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nOverall status: PASS")
    print(f"Total signals audited: {len(signals):,}")
    print(f"Combinations passed: {len(report)}/9")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        raise
