import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

OUTPUT_DIR = "data/targets"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    missing = [key for key, value in DB_CONFIG.items() if not value]

    if missing:
        raise ValueError(
            "Missing database environment variables: "
            + ", ".join(missing)
        )

    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data(connection, table_name):
    query = f"""
        SELECT
            ticker,
            date,
            adj_close
        FROM {table_name}
        WHERE adj_close IS NOT NULL
        ORDER BY ticker, date;
    """

    df = pd.read_sql_query(query, connection)

    # Force one consistent datetime dtype.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    df["adj_close"] = pd.to_numeric(
        df["adj_close"],
        errors="coerce"
    )

    # Remove invalid rows.
    df = df.dropna(
        subset=["ticker", "date", "adj_close"]
    )

    return df


# ============================================================
# GENERATE FORWARD OUTCOME
# ============================================================

def add_future_observation(df, months, suffix):
    """
    For every observation date t, find the first available
    trading observation for the same ticker on or after:

        t + `months` calendar months

    Example:

        2020-01-15 + 6 months
        = 2020-07-15

    If the target calendar date is not a trading day,
    the first available trading date AFTER that date is used.
    """

    # --------------------------------------------------------
    # CURRENT / OBSERVATION DATA
    # --------------------------------------------------------

    current = df[
        [
            "ticker",
            "date",
            "adj_close",
        ]
    ].copy()

    current = current.rename(
        columns={
            "date": "observation_date",
            "adj_close": "observation_price",
        }
    )

    # Force exact same datetime resolution.
    current["observation_date"] = pd.to_datetime(
        current["observation_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # CALENDAR TARGET DATE
    # --------------------------------------------------------

    current["calendar_target_date"] = (
        current["observation_date"]
        + pd.DateOffset(months=months)
    )

    current["calendar_target_date"] = pd.to_datetime(
        current["calendar_target_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # FUTURE DATA
    # --------------------------------------------------------

    future = df[
        [
            "ticker",
            "date",
            "adj_close",
        ]
    ].copy()

    future = future.rename(
        columns={
            "date": "future_date",
            "adj_close": "future_price",
        }
    )

    future["future_date"] = pd.to_datetime(
        future["future_date"],
        errors="coerce"
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # REMOVE INVALID DATES
    # --------------------------------------------------------

    current = current.dropna(
        subset=[
            "ticker",
            "observation_date",
            "calendar_target_date",
        ]
    )

    future = future.dropna(
        subset=[
            "ticker",
            "future_date",
        ]
    )

    # --------------------------------------------------------
    # SORT FOR merge_asof
    # --------------------------------------------------------

    current = current.sort_values(
        ["ticker", "calendar_target_date"]
    ).reset_index(drop=True)

    future = future.sort_values(
        ["ticker", "future_date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # FIRST AVAILABLE TRADING DATE ON OR AFTER
    # TARGET CALENDAR DATE
    # --------------------------------------------------------

    result = pd.merge_asof(
        current,
        future,
        left_on="calendar_target_date",
        right_on="future_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
    )

    # --------------------------------------------------------
    # STORE FUTURE DATE AND PRICE
    # --------------------------------------------------------

    result[f"future_date_{suffix}"] = result[
        "future_date"
    ]

    result[f"future_price_{suffix}"] = result[
        "future_price"
    ]

    # --------------------------------------------------------
    # FORWARD RETURN
    # --------------------------------------------------------

    result[f"forward_return_{suffix}"] = (
        result[f"future_price_{suffix}"]
        / result["observation_price"]
        - 1.0
    )

    return result[
        [
            "ticker",
            "observation_date",
            "calendar_target_date",
            f"future_date_{suffix}",
            "observation_price",
            f"future_price_{suffix}",
            f"forward_return_{suffix}",
        ]
    ]


# ============================================================
# PROCESS ONE ASSET CLASS
# ============================================================

def process_asset_class(df):
    results = []

    for ticker, group in df.groupby(
        "ticker",
        sort=False
    ):
        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        # -----------------------------------------------
        # 6 MONTH
        # -----------------------------------------------

        six_month = add_future_observation(
            group,
            months=6,
            suffix="6m",
        )

        # -----------------------------------------------
        # 1 YEAR
        # -----------------------------------------------

        one_year = add_future_observation(
            group,
            months=12,
            suffix="1y",
        )

        # -----------------------------------------------
        # MERGE 6M + 1Y
        # -----------------------------------------------

        merged = six_month.merge(
            one_year,
            on=[
                "ticker",
                "observation_date",
            ],
            how="left",
            suffixes=("", "_duplicate"),
        )

        # Remove duplicate columns created by merge.
        duplicate_columns = [
            column
            for column in merged.columns
            if column.endswith("_duplicate")
        ]

        merged = merged.drop(
            columns=duplicate_columns,
            errors="ignore",
        )

        results.append(merged)

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df, filename):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df, asset_type):
    print()
    print("=" * 80)
    print(f"{asset_type} FORWARD OUTCOME SUMMARY")
    print("=" * 80)

    print(
        f"Observations: "
        f"{len(df):,}"
    )

    print(
        f"Tickers: "
        f"{df['ticker'].nunique():,}"
    )

    valid_6m = df[
        "forward_return_6m"
    ].notna().sum()

    valid_1y = df[
        "forward_return_1y"
    ].notna().sum()

    missing_6m = df[
        "forward_return_6m"
    ].isna().sum()

    missing_1y = df[
        "forward_return_1y"
    ].isna().sum()

    print(
        f"6M valid returns: "
        f"{valid_6m:,}"
    )

    print(
        f"6M missing returns: "
        f"{missing_6m:,}"
    )

    print(
        f"1Y valid returns: "
        f"{valid_1y:,}"
    )

    print(
        f"1Y missing returns: "
        f"{missing_1y:,}"
    )

    if valid_6m > 0:
        print(
            f"6M minimum return: "
            f"{df['forward_return_6m'].min():.4f}"
        )

        print(
            f"6M maximum return: "
            f"{df['forward_return_6m'].max():.4f}"
        )

        print(
            f"6M median return: "
            f"{df['forward_return_6m'].median():.4f}"
        )

    if valid_1y > 0:
        print(
            f"1Y minimum return: "
            f"{df['forward_return_1y'].min():.4f}"
        )

        print(
            f"1Y maximum return: "
            f"{df['forward_return_1y'].max():.4f}"
        )

        print(
            f"1Y median return: "
            f"{df['forward_return_1y'].median():.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FORWARD OUTCOME GENERATION — 6M / 1Y")
    print("=" * 80)

    connection = get_connection()

    try:

        # ====================================================
        # STOCKS
        # ====================================================

        print()
        print("Loading stock prices...")

        stocks = load_price_data(
            connection,
            "stock_prices",
        )

        print(
            f"Stock rows loaded: "
            f"{len(stocks):,}"
        )

        print(
            "Generating stock forward outcomes..."
        )

        stock_results = process_asset_class(
            stocks
        )

        stock_results["asset_type"] = "STOCK"

        stock_output = save_results(
            stock_results,
            "stock_forward_outcomes.parquet",
        )

        print(
            f"Stock output: "
            f"{stock_output}"
        )

        print_summary(
            stock_results,
            "STOCK",
        )

        # ====================================================
        # ETFs
        # ====================================================

        print()
        print("Loading ETF prices...")

        etfs = load_price_data(
            connection,
            "etf_prices",
        )

        print(
            f"ETF rows loaded: "
            f"{len(etfs):,}"
        )

        print(
            "Generating ETF forward outcomes..."
        )

        etf_results = process_asset_class(
            etfs
        )

        etf_results["asset_type"] = "ETF"

        etf_output = save_results(
            etf_results,
            "etf_forward_outcomes.parquet",
        )

        print(
            f"ETF output: "
            f"{etf_output}"
        )

        print_summary(
            etf_results,
            "ETF",
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        print()
        print("=" * 80)
        print("FORWARD OUTCOME GENERATION COMPLETE")
        print("=" * 80)

        print(
            "6M target: forward return generated"
        )

        print(
            "1Y target: forward return generated"
        )

        print(
            "2Y+ target: NOT GENERATED YET"
        )

        print()
        print(
            "Classification labels have NOT been created yet."
        )

    finally:
        connection.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()