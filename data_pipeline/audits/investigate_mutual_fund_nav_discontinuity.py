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

SCHEMES = [
    "120497",
    "120507",
]

BREAK_DATE = pd.Timestamp("2013-04-22")

WINDOW_DAYS = 10


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    load_dotenv(PROJECT_ROOT / ".env")

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


# ============================================================
# LOAD NAV HISTORY
# ============================================================

def load_scheme_data(scheme_code):

    query = """
        SELECT
            scheme_code,
            nav_date,
            nav
        FROM mutual_fund_nav
        WHERE scheme_code = %s
          AND nav_date BETWEEN %s AND %s
        ORDER BY nav_date;
    """

    start_date = (
        BREAK_DATE
        - pd.Timedelta(days=WINDOW_DAYS)
    ).date()

    end_date = (
        BREAK_DATE
        + pd.Timedelta(days=WINDOW_DAYS)
    ).date()

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    scheme_code,
                    start_date,
                    end_date,
                ),
            )

            rows = cursor.fetchall()

    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "scheme_code",
            "date",
            "nav",
        ],
    )

    if df.empty:
        raise RuntimeError(
            f"No NAV data found for scheme {scheme_code}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce",
    )

    return df


# ============================================================
# CALCULATE DAILY RETURNS
# ============================================================

def calculate_daily_returns(df):

    df = df.copy()

    df["daily_return"] = (
        df["nav"]
        .pct_change()
    )

    return df


# ============================================================
# IDENTIFY BREAK
# ============================================================

def identify_break(df):

    df = df.copy()

    df["nav_ratio"] = (
        df["nav"]
        /
        df["nav"].shift(1)
    )

    df["nav_change_pct"] = (
        df["nav_ratio"] - 1
    ) * 100

    return df


# ============================================================
# ESTIMATE SCALE FACTOR
# ============================================================

def estimate_scale_factor(df):

    # Find the largest positive jump
    idx = df["nav_ratio"].idxmax()

    if pd.isna(idx):
        raise RuntimeError(
            "Could not identify NAV jump."
        )

    jump_row = df.loc[idx]

    jump_date = jump_row["date"]

    previous_index = df.index[
        df.index.get_loc(idx) - 1
    ]

    previous_row = df.loc[
        previous_index
    ]

    previous_nav = float(
        previous_row["nav"]
    )

    post_nav = float(
        jump_row["nav"]
    )

    observed_ratio = (
        post_nav
        /
        previous_nav
    )

    # We expect approximately 100x.
    # For a scale discontinuity, the correction
    # applied to the post-break series is the
    # reciprocal of the observed ratio.
    reciprocal_factor = (
        1.0 / observed_ratio
    )

    return {
        "jump_date": jump_date,
        "previous_date": previous_row["date"],
        "previous_nav": previous_nav,
        "post_nav": post_nav,
        "observed_ratio": observed_ratio,
        "reciprocal_factor": reciprocal_factor,
    }


# ============================================================
# CONTINUITY TEST
# ============================================================

def continuity_test(
    df,
    jump_date,
    factor,
):

    test = df.copy()

    # Correct post-break observations
    test["corrected_nav"] = test["nav"]

    test.loc[
        test["date"] >= jump_date,
        "corrected_nav"
    ] *= factor

    test["corrected_daily_return"] = (
        test["corrected_nav"]
        /
        test["corrected_nav"].shift(1)
        - 1
    )

    # Find the return at the break
    break_rows = test[
        test["date"] == jump_date
    ]

    if break_rows.empty:
        # Use first date after the jump
        break_rows = test[
            test["date"] > jump_date
        ].head(1)

    if break_rows.empty:
        return np.nan

    return float(
        break_rows.iloc[0]["corrected_daily_return"]
    )


# ============================================================
# ALTERNATIVE FACTOR ESTIMATION
# ============================================================

def calculate_local_factors(df):

    df = df.copy()

    df["ratio"] = (
        df["nav"]
        /
        df["nav"].shift(1)
    )

    jump_index = df["ratio"].idxmax()

    position = df.index.get_loc(
        jump_index
    )

    before_values = []

    after_values = []

    # Several observations immediately before
    # and after the discontinuity.

    for offset in range(1, 6):

        before_position = position - offset

        after_position = position + offset - 1

        if (
            before_position >= 0
            and after_position < len(df)
        ):
            before_nav = float(
                df.iloc[
                    before_position
                ]["nav"]
            )

            after_nav = float(
                df.iloc[
                    after_position
                ]["nav"]
            )

            before_values.append(
                before_nav
            )

            after_values.append(
                after_nav
            )

    if not before_values:
        return np.nan

    ratios = (
        np.array(after_values)
        /
        np.array(before_values)
    )

    return {
        "median_local_ratio": np.median(ratios),
        "mean_local_ratio": np.mean(ratios),
        "min_local_ratio": np.min(ratios),
        "max_local_ratio": np.max(ratios),
        "local_ratios": ratios,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def investigate_scheme(scheme_code):

    print("\n" + "=" * 70)
    print(
        f"SCHEME {scheme_code} — NAV DISCONTINUITY INVESTIGATION"
    )
    print("=" * 70)

    df = load_scheme_data(
        scheme_code
    )

    df = calculate_daily_returns(
        df
    )

    df = identify_break(
        df
    )

    print("\nNAV history around suspected break:")

    print(
        df[
            [
                "date",
                "nav",
                "daily_return",
                "nav_ratio",
                "nav_change_pct",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Estimate factor
    # --------------------------------------------------------

    result = estimate_scale_factor(
        df
    )

    print("\n" + "-" * 70)
    print("BREAK DETECTION")
    print("-" * 70)

    print(
        f"Previous date: "
        f"{result['previous_date'].date()}"
    )

    print(
        f"Jump date: "
        f"{result['jump_date'].date()}"
    )

    print(
        f"Previous NAV: "
        f"{result['previous_nav']:.6f}"
    )

    print(
        f"Jump-date NAV: "
        f"{result['post_nav']:.6f}"
    )

    print(
        f"Observed NAV ratio: "
        f"{result['observed_ratio']:.12f}"
    )

    print(
        f"Reciprocal correction factor: "
        f"{result['reciprocal_factor']:.12f}"
    )

    # --------------------------------------------------------
    # Local factor
    # --------------------------------------------------------

    local = calculate_local_factors(
        df
    )

    print("\n" + "-" * 70)
    print("LOCAL SCALE-FACTOR ANALYSIS")
    print("-" * 70)

    if isinstance(local, dict):

        print(
            f"Median local ratio: "
            f"{local['median_local_ratio']:.12f}"
        )

        print(
            f"Mean local ratio: "
            f"{local['mean_local_ratio']:.12f}"
        )

        print(
            f"Minimum local ratio: "
            f"{local['min_local_ratio']:.12f}"
        )

        print(
            f"Maximum local ratio: "
            f"{local['max_local_ratio']:.12f}"
        )

        print(
            "\nLocal ratios:"
        )

        for value in local["local_ratios"]:
            print(
                f"  {value:.12f}"
            )

    # --------------------------------------------------------
    # Apply correction for continuity test
    # --------------------------------------------------------

    correction = result[
        "reciprocal_factor"
    ]

    corrected_break_return = continuity_test(
        df,
        result["jump_date"],
        correction,
    )

    print("\n" + "-" * 70)
    print("CONTINUITY TEST")
    print("-" * 70)

    print(
        f"Correction factor: "
        f"{correction:.12f}"
    )

    print(
        f"Corrected return at break: "
        f"{corrected_break_return:.8f}"
    )

    # --------------------------------------------------------
    # Compare before/after scale
    # --------------------------------------------------------

    pre_break = df[
        df["date"] < result["jump_date"]
    ]

    post_break = df[
        df["date"] >= result["jump_date"]
    ]

    if not pre_break.empty and not post_break.empty:

        pre_last = float(
            pre_break.iloc[-1]["nav"]
        )

        post_first = float(
            post_break.iloc[0]["nav"]
        )

        corrected_post_first = (
            post_first
            * correction
        )

        print(
            f"\nLast pre-break NAV: "
            f"{pre_last:.6f}"
        )

        print(
            f"First post-break NAV: "
            f"{post_first:.6f}"
        )

        print(
            f"Corrected first post-break NAV: "
            f"{corrected_post_first:.6f}"
        )

        continuity_deviation = (
            corrected_post_first
            /
            pre_last
            - 1
        ) * 100

        print(
            f"Continuity deviation: "
            f"{continuity_deviation:.6f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MUTUAL FUND NAV SCALE-DISCONTINUITY INVESTIGATION")
    print("=" * 70)

    print(
        "\nThis script ONLY investigates the NAV history."
    )

    print(
        "It does NOT modify PostgreSQL."
    )

    print(
        "It does NOT modify the existing forward-outcome file."
    )

    for scheme_code in SCHEMES:

        investigate_scheme(
            scheme_code
        )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()