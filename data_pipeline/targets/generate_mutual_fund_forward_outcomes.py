import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "targets"
OUTPUT_FILE = OUTPUT_DIR / "mutual_fund_forward_outcomes.parquet"

START_HORIZON_6M = "6M"
START_HORIZON_1Y = "1Y"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    load_dotenv(PROJECT_ROOT / ".env")

    required_vars = [
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise RuntimeError(
            f"Missing database environment variables: {', '.join(missing)}"
        )

    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )


# ============================================================
# LOAD SELECTED 100 MUTUAL FUNDS
# ============================================================

def load_mutual_fund_data():

    print("=" * 70)
    print("LOADING SELECTED MUTUAL FUND NAV DATA")
    print("=" * 70)

    query = """
        SELECT
            m.scheme_code,
            m.nav_date AS date,
            m.nav
        FROM mutual_fund_nav m
        INNER JOIN investment_fund_universe u
            ON m.scheme_code = u.scheme_code
        ORDER BY m.scheme_code, m.nav_date;
    """

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        columns = ["scheme_code", "date", "nav"]

        df = pd.DataFrame(rows, columns=columns)

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("No mutual fund NAV data returned.")

    df["scheme_code"] = df["scheme_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    print(f"Rows loaded: {len(df):,}")
    print(f"Selected schemes returned: {df['scheme_code'].nunique()}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    return df


# ============================================================
# VALIDATE SOURCE DATA
# ============================================================

def validate_source_data(df):

    print("\n" + "=" * 70)
    print("SOURCE DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Scheme count
    # --------------------------------------------------------

    scheme_count = df["scheme_code"].nunique()

    print(f"Selected scheme count: {scheme_count}")

    if scheme_count != 100:
        raise RuntimeError(
            f"Expected exactly 100 selected schemes, found {scheme_count}."
        )

    # --------------------------------------------------------
    # Duplicate scheme/date
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["scheme_code", "date"]
    ).sum()

    print(f"Duplicate (scheme_code, date): {duplicate_count}")

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate scheme/date rows detected."
        )

    # --------------------------------------------------------
    # Invalid dates
    # --------------------------------------------------------

    invalid_dates = df["date"].isna().sum()

    print(f"Invalid dates: {invalid_dates}")

    if invalid_dates != 0:
        raise RuntimeError("Invalid dates detected.")

    # --------------------------------------------------------
    # Invalid NAV
    # --------------------------------------------------------

    invalid_nav = (
        df["nav"].isna()
        | ~np.isfinite(df["nav"])
        | (df["nav"] <= 0)
    ).sum()

    print(f"Invalid NAV rows: {invalid_nav}")

    if invalid_nav != 0:
        raise RuntimeError(
            "Invalid NAV values detected in the selected 100-fund universe."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["scheme_code", "date"]
    ).reset_index(drop=True)

    print("Source validation: PASS")

    return df


# ============================================================
# CALCULATE FORWARD OUTCOMES
# ============================================================

def calculate_forward_outcomes(df):

    print("\n" + "=" * 70)
    print("CALCULATING 6M / 1Y FORWARD OUTCOMES")
    print("=" * 70)

    result = df.copy()

    # --------------------------------------------------------
    # Target dates
    # --------------------------------------------------------

    result["target_date_6m"] = (
        result["date"] + pd.DateOffset(months=6)
    )

    result["target_date_1y"] = (
        result["date"] + pd.DateOffset(years=1)
    )

    # Initialize future columns
    result["future_date_6m"] = pd.NaT
    result["future_nav_6m"] = np.nan
    result["forward_return_6m"] = np.nan

    result["future_date_1y"] = pd.NaT
    result["future_nav_1y"] = np.nan
    result["forward_return_1y"] = np.nan

    # --------------------------------------------------------
    # Process each scheme independently
    # --------------------------------------------------------

    for scheme_code, group in result.groupby(
        "scheme_code",
        sort=False
    ):

        indices = group.index.to_numpy()

        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        navs = group["nav"].to_numpy(dtype=float)

        target_6m = group["target_date_6m"].to_numpy(
            dtype="datetime64[ns]"
        )

        target_1y = group["target_date_1y"].to_numpy(
            dtype="datetime64[ns]"
        )

        # ----------------------------------------------------
        # 6 months
        # ----------------------------------------------------

        positions_6m = np.searchsorted(
            dates,
            target_6m,
            side="left"
        )

        valid_6m = positions_6m < len(dates)

        valid_indices_6m = indices[valid_6m]
        future_positions_6m = positions_6m[valid_6m]

        result.loc[
            valid_indices_6m,
            "future_date_6m"
        ] = pd.to_datetime(
            dates[future_positions_6m]
        )

        result.loc[
            valid_indices_6m,
            "future_nav_6m"
        ] = navs[future_positions_6m]

        current_nav_6m = result.loc[
            valid_indices_6m,
            "nav"
        ].to_numpy(dtype=float)

        future_nav_values_6m = navs[future_positions_6m]

        result.loc[
            valid_indices_6m,
            "forward_return_6m"
        ] = (
            future_nav_values_6m / current_nav_6m
        ) - 1.0

        # ----------------------------------------------------
        # 1 year
        # ----------------------------------------------------

        positions_1y = np.searchsorted(
            dates,
            target_1y,
            side="left"
        )

        valid_1y = positions_1y < len(dates)

        valid_indices_1y = indices[valid_1y]
        future_positions_1y = positions_1y[valid_1y]

        result.loc[
            valid_indices_1y,
            "future_date_1y"
        ] = pd.to_datetime(
            dates[future_positions_1y]
        )

        result.loc[
            valid_indices_1y,
            "future_nav_1y"
        ] = navs[future_positions_1y]

        current_nav_1y = result.loc[
            valid_indices_1y,
            "nav"
        ].to_numpy(dtype=float)

        future_nav_values_1y = navs[future_positions_1y]

        result.loc[
            valid_indices_1y,
            "forward_return_1y"
        ] = (
            future_nav_values_1y / current_nav_1y
        ) - 1.0

    return result


# ============================================================
# VALIDATE FORWARD OUTCOMES
# ============================================================

def validate_forward_outcomes(
    source_df,
    result_df
):

    print("\n" + "=" * 70)
    print("FORWARD OUTCOME VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Row preservation
    # --------------------------------------------------------

    print(
        f"Source rows: {len(source_df):,}"
    )

    print(
        f"Target rows: {len(result_df):,}"
    )

    if len(source_df) != len(result_df):
        raise RuntimeError(
            "Row count changed during target generation."
        )

    # --------------------------------------------------------
    # Scheme preservation
    # --------------------------------------------------------

    source_schemes = set(
        source_df["scheme_code"].unique()
    )

    result_schemes = set(
        result_df["scheme_code"].unique()
    )

    if source_schemes != result_schemes:
        raise RuntimeError(
            "Scheme universe changed during target generation."
        )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_count = result_df.duplicated(
        subset=["scheme_code", "date"]
    ).sum()

    print(
        f"Duplicate (scheme_code, date): {duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate scheme/date rows detected in target data."
        )

    # --------------------------------------------------------
    # 6M target dates
    # --------------------------------------------------------

    expected_6m = (
        result_df["date"]
        + pd.DateOffset(months=6)
    )

    invalid_6m_target = (
        result_df["target_date_6m"]
        != expected_6m
    ).sum()

    print(
        f"Invalid 6M target dates: {invalid_6m_target}"
    )

    if invalid_6m_target != 0:
        raise RuntimeError(
            "Incorrect 6M target dates detected."
        )

    # --------------------------------------------------------
    # 1Y target dates
    # --------------------------------------------------------

    expected_1y = (
        result_df["date"]
        + pd.DateOffset(years=1)
    )

    invalid_1y_target = (
        result_df["target_date_1y"]
        != expected_1y
    ).sum()

    print(
        f"Invalid 1Y target dates: {invalid_1y_target}"
    )

    if invalid_1y_target != 0:
        raise RuntimeError(
            "Incorrect 1Y target dates detected."
        )

    # --------------------------------------------------------
    # Future dates must be >= target dates
    # --------------------------------------------------------

    valid_future_6m = (
        result_df["future_date_6m"].notna()
    )

    invalid_future_6m = (
        result_df.loc[valid_future_6m, "future_date_6m"]
        <
        result_df.loc[valid_future_6m, "target_date_6m"]
    ).sum()

    print(
        f"6M future-date violations: {invalid_future_6m}"
    )

    if invalid_future_6m != 0:
        raise RuntimeError(
            "6M future date occurs before target date."
        )

    valid_future_1y = (
        result_df["future_date_1y"].notna()
    )

    invalid_future_1y = (
        result_df.loc[valid_future_1y, "future_date_1y"]
        <
        result_df.loc[valid_future_1y, "target_date_1y"]
    ).sum()

    print(
        f"1Y future-date violations: {invalid_future_1y}"
    )

    if invalid_future_1y != 0:
        raise RuntimeError(
            "1Y future date occurs before target date."
        )

    # --------------------------------------------------------
    # Recalculate returns
    # --------------------------------------------------------

    valid_returns_6m = (
        result_df["future_nav_6m"].notna()
        & result_df["forward_return_6m"].notna()
    )

    expected_return_6m = (
        result_df.loc[valid_returns_6m, "future_nav_6m"]
        /
        result_df.loc[valid_returns_6m, "nav"]
    ) - 1.0

    max_return_diff_6m = np.max(
        np.abs(
            expected_return_6m.to_numpy()
            -
            result_df.loc[
                valid_returns_6m,
                "forward_return_6m"
            ].to_numpy()
        )
    )

    print(
        f"Maximum 6M return difference: "
        f"{max_return_diff_6m:.12f}"
    )

    if max_return_diff_6m > 1e-10:
        raise RuntimeError(
            "6M forward-return calculation mismatch."
        )

    valid_returns_1y = (
        result_df["future_nav_1y"].notna()
        & result_df["forward_return_1y"].notna()
    )

    expected_return_1y = (
        result_df.loc[valid_returns_1y, "future_nav_1y"]
        /
        result_df.loc[valid_returns_1y, "nav"]
    ) - 1.0

    max_return_diff_1y = np.max(
        np.abs(
            expected_return_1y.to_numpy()
            -
            result_df.loc[
                valid_returns_1y,
                "forward_return_1y"
            ].to_numpy()
        )
    )

    print(
        f"Maximum 1Y return difference: "
        f"{max_return_diff_1y:.12f}"
    )

    if max_return_diff_1y > 1e-10:
        raise RuntimeError(
            "1Y forward-return calculation mismatch."
        )

    # --------------------------------------------------------
    # Invalid return values
    # --------------------------------------------------------

    invalid_return_6m = (
        result_df["forward_return_6m"].notna()
        &
        ~np.isfinite(
            result_df["forward_return_6m"]
        )
    ).sum()

    invalid_return_1y = (
        result_df["forward_return_1y"].notna()
        &
        ~np.isfinite(
            result_df["forward_return_1y"]
        )
    ).sum()

    print(
        f"Invalid 6M returns: {invalid_return_6m}"
    )

    print(
        f"Invalid 1Y returns: {invalid_return_1y}"
    )

    if invalid_return_6m != 0 or invalid_return_1y != 0:
        raise RuntimeError(
            "Invalid forward returns detected."
        )

    print("\nForward outcome validation: PASS")


# ============================================================
# PRINT COVERAGE
# ============================================================

def print_coverage(result_df):

    print("\n" + "=" * 70)
    print("FORWARD OUTCOME COVERAGE")
    print("=" * 70)

    valid_6m = result_df["forward_return_6m"].notna()
    valid_1y = result_df["forward_return_1y"].notna()

    print(
        f"6M valid outcomes: "
        f"{valid_6m.sum():,}"
    )

    print(
        f"6M missing outcomes: "
        f"{(~valid_6m).sum():,}"
    )

    print(
        f"1Y valid outcomes: "
        f"{valid_1y.sum():,}"
    )

    print(
        f"1Y missing outcomes: "
        f"{(~valid_1y).sum():,}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    if valid_6m.any():

        print("\n6M forward-return distribution:")

        print(
            result_df.loc[
                valid_6m,
                "forward_return_6m"
            ].describe(
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

    if valid_1y.any():

        print("\n1Y forward-return distribution:")

        print(
            result_df.loc[
                valid_1y,
                "forward_return_1y"
            ].describe(
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


# ============================================================
# SAVE DATA
# ============================================================

def save_output(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT SAVED")
    print("=" * 70)

    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(
        f"Schemes: {df['scheme_code'].nunique()}"
    )


# ============================================================
# RELOAD VALIDATION
# ============================================================

def reload_validation():

    print("\n" + "=" * 70)
    print("RELOAD VALIDATION")
    print("=" * 70)

    reloaded = pd.read_parquet(
        OUTPUT_FILE
    )

    print(
        f"Reloaded rows: {len(reloaded):,}"
    )

    print(
        f"Reloaded schemes: "
        f"{reloaded['scheme_code'].nunique()}"
    )

    if len(reloaded) == 0:
        raise RuntimeError(
            "Reloaded output is empty."
        )

    if reloaded["scheme_code"].nunique() != 100:
        raise RuntimeError(
            "Reloaded output does not contain exactly 100 schemes."
        )

    print("Reload validation: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MUTUAL FUND 6M / 1Y FORWARD OUTCOME GENERATION")
    print("=" * 70)
    print("\nIMPORTANT:")
    print("Only the locked 100-fund universe is used.")
    print("PostgreSQL source tables are NOT modified.")
    print("No labels are generated at this stage.")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    source_df = load_mutual_fund_data()

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    source_df = validate_source_data(
        source_df
    )

    # --------------------------------------------------------
    # Generate outcomes
    # --------------------------------------------------------

    result_df = calculate_forward_outcomes(
        source_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_forward_outcomes(
        source_df,
        result_df
    )

    # --------------------------------------------------------
    # Coverage / distributions
    # --------------------------------------------------------

    print_coverage(
        result_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        result_df
    )

    # --------------------------------------------------------
    # Reload
    # --------------------------------------------------------

    reload_validation()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()