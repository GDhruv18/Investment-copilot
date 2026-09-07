from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FOLDER = (
    PROJECT_ROOT
    / "data_pipeline"
    / "ml"
    / "output"
    / "splits"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data_pipeline"
    / "ml"
    / "output"
    / "lstm_30d"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = SPLIT_FOLDER / "stock_train_30d.csv"
VALIDATION_FILE = SPLIT_FOLDER / "stock_validation_30d.csv"
TEST_FILE = SPLIT_FOLDER / "stock_test_30d.csv"


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK = 30

FEATURES = [
    "daily_return",
    "return_30d",
    "return_90d",
    "return_180d",
    "return_1y",
    "volatility_30d",
    "volatility_90d",
    "sma50_ratio",
    "sma200_ratio",
    "drawdown",
]

TARGET = "target_30d"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING LSTM 30D DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

for df in [train_df, validation_df, test_df]:
    df["date"] = pd.to_datetime(df["date"])

    df.sort_values(
        ["ticker", "date"],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)


print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Test rows:       {len(test_df):,}")

print(f"Training stocks:   {train_df['ticker'].nunique()}")
print(f"Validation stocks: {validation_df['ticker'].nunique()}")
print(f"Test stocks:       {test_df['ticker'].nunique()}")


# ============================================================
# FIT SCALER ON TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("FITTING FEATURE SCALER")
print("=" * 70)

scaler = StandardScaler()

scaler.fit(train_df[FEATURES])

print("Scaler fitted using training data only.")


# ============================================================
# SCALE DATA
# ============================================================

train_scaled = train_df.copy()
validation_scaled = validation_df.copy()
test_scaled = test_df.copy()

train_scaled[FEATURES] = scaler.transform(
    train_df[FEATURES]
)

validation_scaled[FEATURES] = scaler.transform(
    validation_df[FEATURES]
)

test_scaled[FEATURES] = scaler.transform(
    test_df[FEATURES]
)


SCALER_FILE = OUTPUT_FOLDER / "lstm_30d_scaler.joblib"

joblib.dump(
    scaler,
    SCALER_FILE
)

print(f"Scaler saved to: {SCALER_FILE}")


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_training_sequences(
    df,
    lookback,
    features,
    target
):
    """
    Create training sequences entirely inside the training period.

    For each stock:

        30 historical observations
                    ↓
                 target

    No validation or test observations are used.
    """

    X = []
    y = []

    for ticker, group in df.groupby(
        "ticker",
        sort=False
    ):

        group = group.sort_values(
            "date"
        ).reset_index(drop=True)

        feature_values = group[
            features
        ].to_numpy(
            dtype=np.float32
        )

        target_values = group[
            target
        ].to_numpy(
            dtype=np.float32
        )

        for i in range(
            lookback - 1,
            len(group)
        ):

            sequence = feature_values[
                i - lookback + 1 : i + 1
            ]

            target_value = target_values[i]

            if np.isnan(target_value):
                continue

            if np.isnan(sequence).any():
                continue

            X.append(sequence)
            y.append(target_value)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def create_context_validation_sequences(
    training_df,
    validation_df,
    lookback,
    features,
    target
):
    """
    Create validation sequences using historical training
    observations as context.

    For each stock:

        Last 29 training observations
                    +
              validation data
                    ↓
               30-step sequence
                    ↓
              validation target

    The validation target itself is never used as an input.
    """

    X = []
    y = []

    for ticker in validation_df["ticker"].unique():

        train_group = training_df[
            training_df["ticker"] == ticker
        ].sort_values("date")

        validation_group = validation_df[
            validation_df["ticker"] == ticker
        ].sort_values("date")

        if len(train_group) < lookback - 1:
            continue

        # Last 29 observations from training period
        context = train_group.tail(
            lookback - 1
        )

        combined = pd.concat(
            [
                context,
                validation_group
            ],
            ignore_index=True
        )

        feature_values = combined[
            features
        ].to_numpy(
            dtype=np.float32
        )

        target_values = combined[
            target
        ].to_numpy(
            dtype=np.float32
        )

        # Validation starts after the context.
        for i in range(
            lookback - 1,
            len(combined)
        ):

            sequence = feature_values[
                i - lookback + 1 : i + 1
            ]

            target_value = target_values[i]

            if np.isnan(target_value):
                continue

            if np.isnan(sequence).any():
                continue

            X.append(sequence)
            y.append(target_value)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def create_context_test_sequences(
    historical_df,
    test_df,
    lookback,
    features,
    target
):
    """
    Create test sequences using historical observations
    immediately before the test period as context.

    Test targets are never used as input features.
    """

    X = []
    y = []

    for ticker in test_df["ticker"].unique():

        historical_group = historical_df[
            historical_df["ticker"] == ticker
        ].sort_values("date")

        test_group = test_df[
            test_df["ticker"] == ticker
        ].sort_values("date")

        if len(historical_group) < lookback - 1:
            continue

        context = historical_group.tail(
            lookback - 1
        )

        combined = pd.concat(
            [
                context,
                test_group
            ],
            ignore_index=True
        )

        feature_values = combined[
            features
        ].to_numpy(
            dtype=np.float32
        )

        target_values = combined[
            target
        ].to_numpy(
            dtype=np.float32
        )

        for i in range(
            lookback - 1,
            len(combined)
        ):

            sequence = feature_values[
                i - lookback + 1 : i + 1
            ]

            target_value = target_values[i]

            if np.isnan(target_value):
                continue

            if np.isnan(sequence).any():
                continue

            X.append(sequence)
            y.append(target_value)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# ============================================================
# TRAINING SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINING SEQUENCES")
print("=" * 70)

X_train, y_train = create_training_sequences(
    train_scaled,
    LOOKBACK,
    FEATURES,
    TARGET
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# ============================================================
# VALIDATION SEQUENCES WITH TRAINING CONTEXT
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION SEQUENCES")
print("=" * 70)

X_validation, y_validation = (
    create_context_validation_sequences(
        train_scaled,
        validation_scaled,
        LOOKBACK,
        FEATURES,
        TARGET
    )
)

print(
    f"X_validation shape: "
    f"{X_validation.shape}"
)

print(
    f"y_validation shape: "
    f"{y_validation.shape}"
)


# ============================================================
# TEST SEQUENCES WITH HISTORICAL CONTEXT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEST SEQUENCES")
print("=" * 70)

# For test sequences, historical information includes both
# training and validation periods.

historical_for_test = pd.concat(
    [
        train_scaled,
        validation_scaled
    ],
    ignore_index=True
)

X_test, y_test = create_context_test_sequences(
    historical_for_test,
    test_scaled,
    LOOKBACK,
    FEATURES,
    TARGET
)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# VALIDATE SHAPES
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE VALIDATION")
print("=" * 70)

print(
    f"Expected sequence length: {LOOKBACK}"
)

print(
    f"Actual training sequence length: "
    f"{X_train.shape[1]}"
)

print(
    f"Expected feature count: {len(FEATURES)}"
)

print(
    f"Actual feature count: "
    f"{X_train.shape[2]}"
)


# ============================================================
# CHECK INVALID VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FOR INVALID VALUES")
print("=" * 70)

print(
    f"Training NaNs: "
    f"{np.isnan(X_train).sum():,}"
)

print(
    f"Validation NaNs: "
    f"{np.isnan(X_validation).sum():,}"
)

print(
    f"Test NaNs: "
    f"{np.isnan(X_test).sum():,}"
)

print(
    f"Training infinite values: "
    f"{np.isinf(X_train).sum():,}"
)

print(
    f"Validation infinite values: "
    f"{np.isinf(X_validation).sum():,}"
)

print(
    f"Test infinite values: "
    f"{np.isinf(X_test).sum():,}"
)


# ============================================================
# SAVE SAMPLE ARRAYS
# ============================================================

sample_size = min(
    1000,
    len(X_train)
)

np.save(
    OUTPUT_FOLDER / "X_train_sample.npy",
    X_train[:sample_size]
)

np.save(
    OUTPUT_FOLDER / "y_train_sample.npy",
    y_train[:sample_size]
)

np.save(
    OUTPUT_FOLDER / "X_validation_sample.npy",
    X_validation[
        :min(1000, len(X_validation))
    ]
)

np.save(
    OUTPUT_FOLDER / "y_validation_sample.npy",
    y_validation[
        :min(1000, len(y_validation))
    ]
)

np.save(
    OUTPUT_FOLDER / "X_test_sample.npy",
    X_test[
        :min(1000, len(X_test))
    ]
)

np.save(
    OUTPUT_FOLDER / "y_test_sample.npy",
    y_test[
        :min(1000, len(y_test))
    ]
)

print("\nSample sequences saved.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LSTM DATA PREPARATION COMPLETED")
print("=" * 70)

print(
    f"Training sequences:   {len(X_train):,}"
)

print(
    f"Validation sequences: {len(X_validation):,}"
)

print(
    f"Test sequences:       {len(X_test):,}"
)

print(
    "\nValidation and test sequences use historical "
    "context from the preceding period."
)

print(
    "Scaler was fitted on training data only."
)

print(
    "No validation or test targets were used during "
    "training-data preprocessing."
)