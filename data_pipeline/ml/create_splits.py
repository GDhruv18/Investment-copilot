from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data_pipeline/ml/output/stock_ml_dataset.csv"
)

OUTPUT_DIR = Path(
    "data_pipeline/ml/output/splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading stock ML dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

print(f"Total rows: {len(df):,}")
print(f"Stocks: {df['ticker'].nunique()}")

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# CALCULATE FUTURE OBSERVATION DATES
# ============================================================

print("\nCalculating future observation dates...")

for horizon in [30, 90, 252]:

    df[f"future_date_{horizon}d"] = (
        df.groupby("ticker")["date"]
        .shift(-horizon)
    )


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_split(data, filename):

    path = OUTPUT_DIR / filename

    data.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")


# ============================================================
# 30D
# ============================================================

print("\n" + "=" * 70)
print("CREATING 30D SPLIT")
print("=" * 70)

data_30 = df[
    df["target_30d"].notna()
    &
    df["future_date_30d"].notna()
].copy()


# TRAIN
train_30 = data_30[
    (data_30["date"] < "2024-01-01")
    &
    (data_30["future_date_30d"] < "2024-01-01")
].copy()


# VALIDATION
validation_30 = data_30[
    (data_30["date"] >= "2024-01-01")
    &
    (data_30["date"] < "2025-01-01")
    &
    (data_30["future_date_30d"] < "2025-01-01")
].copy()


# TEST
test_30 = data_30[
    data_30["date"] >= "2025-01-01"
].copy()


train_30.drop(
    columns=["future_date_30d"],
    inplace=True
)

validation_30.drop(
    columns=["future_date_30d"],
    inplace=True
)

test_30.drop(
    columns=["future_date_30d"],
    inplace=True
)


print(f"Training rows:   {len(train_30):,}")
print(f"Validation rows: {len(validation_30):,}")
print(f"Test rows:       {len(test_30):,}")

save_split(
    train_30,
    "stock_train_30d.csv"
)

save_split(
    validation_30,
    "stock_validation_30d.csv"
)

save_split(
    test_30,
    "stock_test_30d.csv"
)


# ============================================================
# 90D
# ============================================================

print("\n" + "=" * 70)
print("CREATING 90D SPLIT")
print("=" * 70)

data_90 = df[
    df["target_90d"].notna()
    &
    df["future_date_90d"].notna()
].copy()


# TRAIN
train_90 = data_90[
    (data_90["date"] < "2024-01-01")
    &
    (data_90["future_date_90d"] < "2024-01-01")
].copy()


# VALIDATION
validation_90 = data_90[
    (data_90["date"] >= "2024-01-01")
    &
    (data_90["date"] < "2025-01-01")
    &
    (data_90["future_date_90d"] < "2025-01-01")
].copy()


# TEST
test_90 = data_90[
    data_90["date"] >= "2025-01-01"
].copy()


train_90.drop(
    columns=["future_date_90d"],
    inplace=True
)

validation_90.drop(
    columns=["future_date_90d"],
    inplace=True
)

test_90.drop(
    columns=["future_date_90d"],
    inplace=True
)


print(f"Training rows:   {len(train_90):,}")
print(f"Validation rows: {len(validation_90):,}")
print(f"Test rows:       {len(test_90):,}")

save_split(
    train_90,
    "stock_train_90d.csv"
)

save_split(
    validation_90,
    "stock_validation_90d.csv"
)

save_split(
    test_90,
    "stock_test_90d.csv"
)


# ============================================================
# 252D WALK-FORWARD VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING 252D WALK-FORWARD SPLITS")
print("=" * 70)

data_252 = df[
    df["target_252d"].notna()
    &
    df["future_date_252d"].notna()
].copy()


# ============================================================
# FOLD 1
#
# TRAIN:
# 2019–2021
#
# VALIDATION:
# 2022
#
# The training target must finish before 2022.
#
# Validation targets are allowed to extend into 2023.
# 2023–2024 acts as an embargo before the final test.
# ============================================================

train_252_fold1 = data_252[
    (data_252["date"] < "2022-01-01")
    &
    (data_252["future_date_252d"] < "2022-01-01")
].copy()


validation_252_fold1 = data_252[
    (data_252["date"] >= "2022-01-01")
    &
    (data_252["date"] < "2023-01-01")
].copy()


train_252_fold1.drop(
    columns=["future_date_252d"],
    inplace=True
)

validation_252_fold1.drop(
    columns=["future_date_252d"],
    inplace=True
)


print("\n252D Fold 1")
print(
    f"Training rows:   {len(train_252_fold1):,}"
)

print(
    f"Validation rows: {len(validation_252_fold1):,}"
)

if len(validation_252_fold1) > 0:

    print(
        f"Validation dates: "
        f"{validation_252_fold1['date'].min().date()} → "
        f"{validation_252_fold1['date'].max().date()}"
    )


save_split(
    train_252_fold1,
    "stock_train_252d_fold1.csv"
)

save_split(
    validation_252_fold1,
    "stock_validation_252d_fold1.csv"
)


# ============================================================
# FOLD 2
#
# TRAIN:
# 2019–2022
#
# VALIDATION:
# 2023
#
# Training targets must finish before 2023.
#
# Validation targets may extend into 2024.
# 2024 acts as an embargo before the final test.
# ============================================================

train_252_fold2 = data_252[
    (data_252["date"] < "2023-01-01")
    &
    (data_252["future_date_252d"] < "2023-01-01")
].copy()


validation_252_fold2 = data_252[
    (data_252["date"] >= "2023-01-01")
    &
    (data_252["date"] < "2024-01-01")
].copy()


train_252_fold2.drop(
    columns=["future_date_252d"],
    inplace=True
)

validation_252_fold2.drop(
    columns=["future_date_252d"],
    inplace=True
)


print("\n252D Fold 2")
print(
    f"Training rows:   {len(train_252_fold2):,}"
)

print(
    f"Validation rows: {len(validation_252_fold2):,}"
)

if len(validation_252_fold2) > 0:

    print(
        f"Validation dates: "
        f"{validation_252_fold2['date'].min().date()} → "
        f"{validation_252_fold2['date'].max().date()}"
    )


save_split(
    train_252_fold2,
    "stock_train_252d_fold2.csv"
)

save_split(
    validation_252_fold2,
    "stock_validation_252d_fold2.csv"
)


# ============================================================
# FINAL 252D TEST
#
# Test begins in 2025.
#
# This keeps 2023–2024 completely outside the final test
# evaluation and provides a genuine future test period.
# ============================================================

test_252 = data_252[
    data_252["date"] >= "2025-01-01"
].copy()


test_252.drop(
    columns=["future_date_252d"],
    inplace=True
)


print("\n252D Final Test")

print(
    f"Test rows: {len(test_252):,}"
)

if len(test_252) > 0:

    print(
        f"Test dates: "
        f"{test_252['date'].min().date()} → "
        f"{test_252['date'].max().date()}"
    )


save_split(
    test_252,
    "stock_test_252d.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL STOCK SPLITS CREATED")
print("=" * 70)

print("\n30D:")
print(f"  Train:      {len(train_30):,}")
print(f"  Validation: {len(validation_30):,}")
print(f"  Test:       {len(test_30):,}")

print("\n90D:")
print(f"  Train:      {len(train_90):,}")
print(f"  Validation: {len(validation_90):,}")
print(f"  Test:       {len(test_90):,}")

print("\n252D Walk-Forward:")
print(
    f"  Fold 1 Train:      {len(train_252_fold1):,}"
)

print(
    f"  Fold 1 Validation: {len(validation_252_fold1):,}"
)

print(
    f"  Fold 2 Train:      {len(train_252_fold2):,}"
)

print(
    f"  Fold 2 Validation: {len(validation_252_fold2):,}"
)

print(
    f"  Final Test:        {len(test_252):,}"
)