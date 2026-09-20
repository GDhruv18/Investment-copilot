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
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(asset_type, file_path):

    print()
    print("=" * 100)
    print(f"{asset_type} FORWARD RETURN AUDIT")
    print("=" * 100)

    df = pd.read_parquet(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")

    # --------------------------------------------------------
    # 6M EXTREMES
    # --------------------------------------------------------

    valid_6m = df[
        df["forward_return_6m"].notna()
    ].copy()

    print()
    print("-" * 100)
    print("TOP 20 EXTREME 6M RETURNS")
    print("-" * 100)

    top_6m = valid_6m.nlargest(
        20,
        "forward_return_6m"
    )

    print(
        top_6m[
            [
                "ticker",
                "observation_date",
                "calendar_target_date",
                "future_date_6m",
                "observation_price",
                "future_price_6m",
                "forward_return_6m",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 100)
    print("BOTTOM 20 EXTREME 6M RETURNS")
    print("-" * 100)

    bottom_6m = valid_6m.nsmallest(
        20,
        "forward_return_6m"
    )

    print(
        bottom_6m[
            [
                "ticker",
                "observation_date",
                "calendar_target_date",
                "future_date_6m",
                "observation_price",
                "future_price_6m",
                "forward_return_6m",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 1Y EXTREMES
    # --------------------------------------------------------

    valid_1y = df[
        df["forward_return_1y"].notna()
    ].copy()

    print()
    print("-" * 100)
    print("TOP 20 EXTREME 1Y RETURNS")
    print("-" * 100)

    top_1y = valid_1y.nlargest(
        20,
        "forward_return_1y"
    )

    print(
        top_1y[
            [
                "ticker",
                "observation_date",
                "calendar_target_date_1y",
                "future_date_1y",
                "observation_price",
                "future_price_1y",
                "forward_return_1y",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 100)
    print("BOTTOM 20 EXTREME 1Y RETURNS")
    print("-" * 100)

    bottom_1y = valid_1y.nsmallest(
        20,
        "forward_return_1y"
    )

    print(
        bottom_1y[
            [
                "ticker",
                "observation_date",
                "calendar_target_date_1y",
                "future_date_1y",
                "observation_price",
                "future_price_1y",
                "forward_return_1y",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # DISTRIBUTION CHECK
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("RETURN DISTRIBUTION")
    print("-" * 100)

    print()
    print("6M:")
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
    print("1Y:")
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
    # VERY LARGE RETURNS
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("COUNTS OF VERY LARGE POSITIVE RETURNS")
    print("-" * 100)

    thresholds = [
        1.0,     # +100%
        2.0,     # +200%
        5.0,     # +500%
        10.0,    # +1000%
        20.0,    # +2000%
        50.0,    # +5000%
    ]

    for threshold in thresholds:

        count_6m = (
            valid_6m["forward_return_6m"] > threshold
        ).sum()

        count_1y = (
            valid_1y["forward_return_1y"] > threshold
        ).sum()

        print(
            f">{threshold * 100:>6.0f}% "
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
            print(f"ERROR: File not found: {file_path}")
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