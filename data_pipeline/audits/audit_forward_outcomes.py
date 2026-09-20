import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "STOCK": "data/targets/stock_forward_outcomes.parquet",
    "ETF": "data/targets/etf_forward_outcomes.parquet",
}


# ============================================================
# DISPLAY EXTREME RETURNS
# ============================================================

def display_extremes(df, return_column, future_date_column, title, ascending=False):

    print()
    print("-" * 100)
    print(title)
    print("-" * 100)

    valid = df[
        df[return_column].notna()
    ].copy()

    if ascending:
        extreme = valid.nsmallest(
            20,
            return_column
        )
    else:
        extreme = valid.nlargest(
            20,
            return_column
        )

    columns = [
        "ticker",
        "observation_date",
        "calendar_target_date",
        future_date_column,
        "observation_price",
        return_column.replace(
            "forward_return",
            "future_price"
        ),
        return_column,
    ]

    print(
        extreme[
            columns
        ].to_string(index=False)
    )


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(asset_type, file_path):

    print()
    print("=" * 100)
    print(f"{asset_type} FORWARD RETURN AUDIT")
    print("=" * 100)

    df = pd.read_parquet(file_path)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Tickers: {df['ticker'].nunique():,}"
    )

    # --------------------------------------------------------
    # 6M EXTREMES
    # --------------------------------------------------------

    display_extremes(
        df,
        "forward_return_6m",
        "future_date_6m",
        "TOP 20 EXTREME 6M RETURNS",
        ascending=False,
    )

    display_extremes(
        df,
        "forward_return_6m",
        "future_date_6m",
        "BOTTOM 20 EXTREME 6M RETURNS",
        ascending=True,
    )

    # --------------------------------------------------------
    # 1Y EXTREMES
    # --------------------------------------------------------

    display_extremes(
        df,
        "forward_return_1y",
        "future_date_1y",
        "TOP 20 EXTREME 1Y RETURNS",
        ascending=False,
    )

    display_extremes(
        df,
        "forward_return_1y",
        "future_date_1y",
        "BOTTOM 20 EXTREME 1Y RETURNS",
        ascending=True,
    )

    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    valid_6m = df[
        df["forward_return_6m"].notna()
    ]

    valid_1y = df[
        df["forward_return_1y"].notna()
    ]

    print()
    print("-" * 100)
    print("6M RETURN DISTRIBUTION")
    print("-" * 100)

    print(
        valid_6m["forward_return_6m"].describe(
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

    print()
    print("-" * 100)
    print("1Y RETURN DISTRIBUTION")
    print("-" * 100)

    print(
        valid_1y["forward_return_1y"].describe(
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

    # --------------------------------------------------------
    # LARGE POSITIVE RETURNS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("COUNTS OF VERY LARGE POSITIVE RETURNS")
    print("-" * 100)

    thresholds = [
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
    ]

    for threshold in thresholds:

        count_6m = (
            valid_6m["forward_return_6m"]
            > threshold
        ).sum()

        count_1y = (
            valid_1y["forward_return_1y"]
            > threshold
        ).sum()

        print(
            f">{threshold * 100:>6.0f}% "
            f"| 6M: {count_6m:>6,} "
            f"| 1Y: {count_1y:>6,}"
        )

    # --------------------------------------------------------
    # LARGE NEGATIVE RETURNS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("COUNTS OF VERY LARGE NEGATIVE RETURNS")
    print("-" * 100)

    negative_thresholds = [
        -0.25,
        -0.50,
        -0.75,
        -0.90,
    ]

    for threshold in negative_thresholds:

        count_6m = (
            valid_6m["forward_return_6m"]
            < threshold
        ).sum()

        count_1y = (
            valid_1y["forward_return_1y"]
            < threshold
        ).sum()

        print(
            f"<{threshold * 100:>6.0f}% "
            f"| 6M: {count_6m:>6,} "
            f"| 1Y: {count_1y:>6,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("FORWARD OUTCOME DATA QUALITY AUDIT")
    print("=" * 100)

    for asset_type, file_path in FILES.items():

        if not os.path.exists(file_path):

            print()
            print(
                f"ERROR: File not found: {file_path}"
            )

            continue

        audit_dataset(
            asset_type,
            file_path
        )

    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()