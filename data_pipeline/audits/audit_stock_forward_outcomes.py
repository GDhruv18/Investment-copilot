from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/targets/stock_forward_outcomes.parquet")

EXPECTED_STOCKS = 100

REQUIRED_COLUMNS = [
    "ticker",
    "observation_date",
    "calendar_target_date",
    "future_date_6m",
    "future_date_1y",
    "observation_price",
    "future_price_6m",
    "future_price_1y",
    "forward_return_6m",
    "forward_return_1y",
]


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_required_columns(df):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def audit_duplicates(df):
    duplicate_count = df.duplicated(
        subset=["ticker", "observation_date"]
    ).sum()

    print(f"Duplicate ticker/date rows: {duplicate_count:,}")

    if duplicate_count != 0:
        print("STATUS: FAIL")
        return False

    return True


def audit_dates(df):
    print_section("DATE VALIDATION")

    observation = pd.to_datetime(
        df["observation_date"],
        errors="coerce"
    )

    target_6m = pd.to_datetime(
        df["calendar_target_date"],
        errors="coerce"
    )

    future_6m = pd.to_datetime(
        df["future_date_6m"],
        errors="coerce"
    )

    future_1y = pd.to_datetime(
        df["future_date_1y"],
        errors="coerce"
    )

    invalid_observation_dates = observation.isna().sum()
    invalid_target_dates = target_6m.isna().sum()
    invalid_future_6m = future_6m.notna().sum()
    invalid_future_1y = future_1y.notna().sum()

    print(
        f"Invalid observation dates: {invalid_observation_dates:,}"
    )
    print(
        f"Invalid 6M target dates:   {invalid_target_dates:,}"
    )

    print(
        f"Rows with 6M future date:  {invalid_future_6m:,}"
    )
    print(
        f"Rows with 1Y future date:  {invalid_future_1y:,}"
    )

    # Calendar target correctness
    expected_6m = observation + pd.DateOffset(months=6)
    expected_1y = observation + pd.DateOffset(years=1)

    if "calendar_target_date" in df.columns:
        target_6m_match = (
            target_6m.dt.normalize()
            == expected_6m.dt.normalize()
        )

        bad_6m_targets = (
            target_6m.notna()
            & observation.notna()
            & ~target_6m_match
        ).sum()

        print(
            f"Incorrect 6M calendar targets: {bad_6m_targets:,}"
        )
    else:
        bad_6m_targets = 1

    # Future dates must be >= calendar target date
    bad_future_6m = (
        future_6m.notna()
        & target_6m.notna()
        & (future_6m < target_6m)
    ).sum()

    bad_future_1y = (
        future_1y.notna()
        & (
            future_1y
            < pd.to_datetime(
                df["observation_date"],
                errors="coerce"
            ) + pd.DateOffset(years=1)
        )
    ).sum()

    print(
        f"6M future dates before target: {bad_future_6m:,}"
    )
    print(
        f"1Y future dates before target: {bad_future_1y:,}"
    )

    passed = (
        invalid_observation_dates == 0
        and invalid_target_dates == 0
        and bad_6m_targets == 0
        and bad_future_6m == 0
        and bad_future_1y == 0
    )

    return passed


def audit_prices_and_returns(df):
    print_section("PRICE / RETURN VALIDATION")

    numeric_columns = [
        "observation_price",
        "future_price_6m",
        "future_price_1y",
        "forward_return_6m",
        "forward_return_1y",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print(
        f"Invalid observation prices: "
        f"{df['observation_price'].isna().sum():,}"
    )

    non_positive_observation = (
        df["observation_price"].notna()
        & (df["observation_price"] <= 0)
    ).sum()

    print(
        f"Non-positive observation prices: "
        f"{non_positive_observation:,}"
    )

    # Recalculate returns where both prices exist
    valid_6m = (
        df["observation_price"].notna()
        & df["future_price_6m"].notna()
        & (df["observation_price"] > 0)
        & (df["future_price_6m"] > 0)
    )

    valid_1y = (
        df["observation_price"].notna()
        & df["future_price_1y"].notna()
        & (df["observation_price"] > 0)
        & (df["future_price_1y"] > 0)
    )

    calculated_6m = (
        df.loc[valid_6m, "future_price_6m"]
        / df.loc[valid_6m, "observation_price"]
        - 1.0
    )

    calculated_1y = (
        df.loc[valid_1y, "future_price_1y"]
        / df.loc[valid_1y, "observation_price"]
        - 1.0
    )

    stored_6m = df.loc[
        valid_6m,
        "forward_return_6m"
    ]

    stored_1y = df.loc[
        valid_1y,
        "forward_return_1y"
    ]

    return_error_6m = (
        calculated_6m - stored_6m
    ).abs()

    return_error_1y = (
        calculated_1y - stored_1y
    ).abs()

    bad_return_6m = (
        return_error_6m > 1e-8
    ).sum()

    bad_return_1y = (
        return_error_1y > 1e-8
    ).sum()

    print(
        f"Incorrect stored 6M returns: {bad_return_6m:,}"
    )

    print(
        f"Incorrect stored 1Y returns: {bad_return_1y:,}"
    )

    return (
        non_positive_observation == 0
        and bad_return_6m == 0
        and bad_return_1y == 0
    )


def print_return_distribution(df, column, label):
    print_section(f"{label} RETURN DISTRIBUTION")

    values = df[column].dropna()

    if values.empty:
        print("No valid observations.")
        return

    print(
        values.describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\nExtreme-return counts:")

    thresholds_positive = [
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ]

    thresholds_negative = [
        -0.25,
        -0.50,
        -0.75,
        -0.90,
    ]

    for threshold in thresholds_positive:
        count = (values > threshold).sum()

        print(
            f"> {threshold * 100:6.0f}% : {count:8,}"
        )

    for threshold in thresholds_negative:
        count = (values < threshold).sum()

        print(
            f"< {threshold * 100:6.0f}% : {count:8,}"
        )


def print_extreme_examples(df, column, ascending=False):
    print_section(
        f"EXTREME {column} OBSERVATIONS"
    )

    columns = [
        "ticker",
        "observation_date",
        "calendar_target_date",
        "future_date_6m",
        "future_date_1y",
        "observation_price",
        "forward_return_6m",
        "forward_return_1y",
    ]

    available_columns = [
        column_name
        for column_name in columns
        if column_name in df.columns
    ]

    subset = df.dropna(
        subset=[column]
    ).sort_values(
        column,
        ascending=ascending
    ).head(20)

    print(
        subset[available_columns].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STOCK FORWARD OUTCOME DATA QUALITY AUDIT")
    print("=" * 80)

    print("\nLoading:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print_section("INPUT VALIDATION")

    print(
        f"Rows:        {len(df):,}"
    )

    print(
        f"Stocks:      {df['ticker'].nunique():,}"
    )

    print(
        f"Date range:  "
        f"{pd.to_datetime(df['observation_date']).min().date()} "
        f"-> "
        f"{pd.to_datetime(df['observation_date']).max().date()}"
    )

    print(
        f"Expected stocks: {EXPECTED_STOCKS}"
    )

    validate_required_columns(df)

    duplicate_pass = audit_duplicates(df)

    stock_count_pass = (
        df["ticker"].nunique()
        == EXPECTED_STOCKS
    )

    if not stock_count_pass:
        print(
            "WARNING: stock count does not match expected universe."
        )

    print_section("MISSING FORWARD OUTCOMES")

    missing_6m = df["forward_return_6m"].isna().sum()
    missing_1y = df["forward_return_1y"].isna().sum()

    valid_6m = df["forward_return_6m"].notna().sum()
    valid_1y = df["forward_return_1y"].notna().sum()

    print(
        f"Valid 6M returns: {valid_6m:,}"
    )
    print(
        f"Missing 6M returns: {missing_6m:,}"
    )

    print(
        f"Valid 1Y returns: {valid_1y:,}"
    )
    print(
        f"Missing 1Y returns: {missing_1y:,}"
    )

    date_pass = audit_dates(df)

    price_return_pass = audit_prices_and_returns(df)

    print_return_distribution(
        df,
        "forward_return_6m",
        "6M"
    )

    print_return_distribution(
        df,
        "forward_return_1y",
        "1Y"
    )

    print_extreme_examples(
        df,
        "forward_return_6m",
        ascending=False
    )

    print_extreme_examples(
        df,
        "forward_return_6m",
        ascending=True
    )

    print_extreme_examples(
        df,
        "forward_return_1y",
        ascending=False
    )

    print_extreme_examples(
        df,
        "forward_return_1y",
        ascending=True
    )

    print_section("FINAL AUDIT STATUS")

    print(
        f"Required columns:       PASS"
    )

    print(
        f"Duplicate ticker/date:   "
        f"{'PASS' if duplicate_pass else 'FAIL'}"
    )

    print(
        f"Stock universe:          "
        f"{'PASS' if stock_count_pass else 'FAIL'}"
    )

    print(
        f"Date validation:         "
        f"{'PASS' if date_pass else 'FAIL'}"
    )

    print(
        f"Price/return validation: "
        f"{'PASS' if price_return_pass else 'FAIL'}"
    )

    overall_pass = (
        duplicate_pass
        and stock_count_pass
        and date_pass
        and price_return_pass
    )

    print("\n" + "=" * 80)

    if overall_pass:
        print("AUDIT STATUS: PASS")
        print(
            "Stock forward outcomes are ready for target generation."
        )
    else:
        print("AUDIT STATUS: FAIL")
        print(
            "Do NOT generate stock targets yet."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()