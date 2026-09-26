from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "mutual_fund_nav_clean.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "targets"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mutual_fund_forward_outcomes_clean.parquet"
)

EXPECTED_ROWS = 206_639
EXPECTED_SCHEMES = 100


# ============================================================
# LOAD CLEAN NAV DATA
# ============================================================

def load_clean_nav():

    print("=" * 70)
    print("MUTUAL FUND CLEAN 6M / 1Y FORWARD OUTCOME GENERATION")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean NAV file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Schemes: {df['scheme_code'].nunique()}"
    )

    df["scheme_code"] = (
        df["scheme_code"]
        .astype(str)
    )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.normalize()
    )

    df["nav"] = pd.to_numeric(
        df["nav"],
        errors="coerce"
    )

    df = df.sort_values(
        ["scheme_code", "date"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source(df):

    print("\n" + "=" * 70)
    print("CLEAN NAV SOURCE VALIDATION")
    print("=" * 70)

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(df):,}."
        )

    scheme_count = (
        df["scheme_code"].nunique()
    )

    print(
        f"Scheme count: {scheme_count}"
    )

    if scheme_count != EXPECTED_SCHEMES:
        raise RuntimeError(
            f"Expected {EXPECTED_SCHEMES} schemes, "
            f"found {scheme_count}."
        )

    duplicate_count = df.duplicated(
        subset=[
            "scheme_code",
            "date"
        ]
    ).sum()

    print(
        f"Duplicate scheme/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate scheme/date rows detected."
        )

    invalid_nav = (
        df["nav"].isna()
        | ~np.isfinite(df["nav"])
        | (df["nav"] <= 0)
    ).sum()

    print(
        f"Invalid NAV rows: {invalid_nav}"
    )

    if invalid_nav != 0:
        raise RuntimeError(
            "Invalid NAV values detected."
        )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        "Clean NAV source validation: PASS"
    )


# ============================================================
# GENERATE FORWARD OUTCOMES
# ============================================================

def generate_forward_outcomes(df):

    print("\n" + "=" * 70)
    print("CALCULATING CLEAN 6M / 1Y FORWARD OUTCOMES")
    print("=" * 70)

    result = df.copy()

    # --------------------------------------------------------
    # Target dates
    # --------------------------------------------------------

    result["target_date_6m"] = (
        result["date"]
        + pd.DateOffset(months=6)
    )

    result["target_date_1y"] = (
        result["date"]
        + pd.DateOffset(years=1)
    )

    # --------------------------------------------------------
    # Initialize output columns
    # --------------------------------------------------------

    result["future_date_6m"] = pd.NaT
    result["future_nav_6m"] = np.nan
    result["forward_return_6m"] = np.nan

    result["future_date_1y"] = pd.NaT
    result["future_nav_1y"] = np.nan
    result["forward_return_1y"] = np.nan

    # --------------------------------------------------------
    # Process scheme by scheme
    # --------------------------------------------------------

    for scheme_code, group in result.groupby(
        "scheme_code",
        sort=False
    ):

        indices = group.index.to_numpy()

        dates = group["date"].to_numpy(
            dtype="datetime64[ns]"
        )

        navs = group["nav"].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # 6 months
        # ----------------------------------------------------

        target_6m = group[
            "target_date_6m"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        positions_6m = np.searchsorted(
            dates,
            target_6m,
            side="left"
        )

        valid_6m = (
            positions_6m < len(dates)
        )

        valid_indices_6m = (
            indices[valid_6m]
        )

        future_positions_6m = (
            positions_6m[valid_6m]
        )

        future_dates_6m = (
            dates[future_positions_6m]
        )

        future_navs_6m = (
            navs[future_positions_6m]
        )

        current_navs_6m = (
            navs[valid_6m]
        )

        result.loc[
            valid_indices_6m,
            "future_date_6m"
        ] = pd.to_datetime(
            future_dates_6m
        )

        result.loc[
            valid_indices_6m,
            "future_nav_6m"
        ] = future_navs_6m

        result.loc[
            valid_indices_6m,
            "forward_return_6m"
        ] = (
            future_navs_6m
            /
            current_navs_6m
            - 1.0
        )

        # ----------------------------------------------------
        # 1 year
        # ----------------------------------------------------

        target_1y = group[
            "target_date_1y"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        positions_1y = np.searchsorted(
            dates,
            target_1y,
            side="left"
        )

        valid_1y = (
            positions_1y < len(dates)
        )

        valid_indices_1y = (
            indices[valid_1y]
        )

        future_positions_1y = (
            positions_1y[valid_1y]
        )

        future_dates_1y = (
            dates[future_positions_1y]
        )

        future_navs_1y = (
            navs[future_positions_1y]
        )

        current_navs_1y = (
            navs[valid_1y]
        )

        result.loc[
            valid_indices_1y,
            "future_date_1y"
        ] = pd.to_datetime(
            future_dates_1y
        )

        result.loc[
            valid_indices_1y,
            "future_nav_1y"
        ] = future_navs_1y

        result.loc[
            valid_indices_1y,
            "forward_return_1y"
        ] = (
            future_navs_1y
            /
            current_navs_1y
            - 1.0
        )

    return result


# ============================================================
# VALIDATE FORWARD OUTCOMES
# ============================================================

def validate_forward_outcomes(
    source,
    result
):

    print("\n" + "=" * 70)
    print("CLEAN FORWARD OUTCOME VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Row preservation
    # --------------------------------------------------------

    print(
        f"Source rows: {len(source):,}"
    )

    print(
        f"Target rows: {len(result):,}"
    )

    if len(source) != len(result):
        raise RuntimeError(
            "Row count changed."
        )

    # --------------------------------------------------------
    # Scheme preservation
    # --------------------------------------------------------

    if (
        source["scheme_code"].nunique()
        != result["scheme_code"].nunique()
    ):
        raise RuntimeError(
            "Scheme count changed."
        )

    # --------------------------------------------------------
    # Duplicate keys
    # --------------------------------------------------------

    duplicate_count = result.duplicated(
        subset=[
            "scheme_code",
            "date"
        ]
    ).sum()

    print(
        f"Duplicate scheme/date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate keys detected."
        )

    # --------------------------------------------------------
    # Target dates
    # --------------------------------------------------------

    expected_6m = (
        result["date"]
        + pd.DateOffset(months=6)
    )

    expected_1y = (
        result["date"]
        + pd.DateOffset(years=1)
    )

    invalid_6m_target = (
        result["target_date_6m"]
        != expected_6m
    ).sum()

    invalid_1y_target = (
        result["target_date_1y"]
        != expected_1y
    ).sum()

    print(
        f"Invalid 6M target dates: "
        f"{invalid_6m_target}"
    )

    print(
        f"Invalid 1Y target dates: "
        f"{invalid_1y_target}"
    )

    if (
        invalid_6m_target != 0
        or invalid_1y_target != 0
    ):
        raise RuntimeError(
            "Invalid target dates detected."
        )

    # --------------------------------------------------------
    # Future-date checks
    # --------------------------------------------------------

    valid_6m = (
        result["future_date_6m"]
        .notna()
    )

    invalid_future_6m = (
        result.loc[
            valid_6m,
            "future_date_6m"
        ]
        <
        result.loc[
            valid_6m,
            "target_date_6m"
        ]
    ).sum()

    valid_1y = (
        result["future_date_1y"]
        .notna()
    )

    invalid_future_1y = (
        result.loc[
            valid_1y,
            "future_date_1y"
        ]
        <
        result.loc[
            valid_1y,
            "target_date_1y"
        ]
    ).sum()

    print(
        f"6M future-date violations: "
        f"{invalid_future_6m}"
    )

    print(
        f"1Y future-date violations: "
        f"{invalid_future_1y}"
    )

    if (
        invalid_future_6m != 0
        or invalid_future_1y != 0
    ):
        raise RuntimeError(
            "Future date occurs before target date."
        )

    # --------------------------------------------------------
    # Return calculations
    # --------------------------------------------------------

    valid_return_6m = (
        result["future_nav_6m"].notna()
        &
        result["forward_return_6m"].notna()
    )

    expected_return_6m = (
        result.loc[
            valid_return_6m,
            "future_nav_6m"
        ]
        /
        result.loc[
            valid_return_6m,
            "nav"
        ]
        - 1.0
    )

    max_difference_6m = np.max(
        np.abs(
            expected_return_6m.to_numpy()
            -
            result.loc[
                valid_return_6m,
                "forward_return_6m"
            ].to_numpy()
        )
    )

    valid_return_1y = (
        result["future_nav_1y"].notna()
        &
        result["forward_return_1y"].notna()
    )

    expected_return_1y = (
        result.loc[
            valid_return_1y,
            "future_nav_1y"
        ]
        /
        result.loc[
            valid_return_1y,
            "nav"
        ]
        - 1.0
    )

    max_difference_1y = np.max(
        np.abs(
            expected_return_1y.to_numpy()
            -
            result.loc[
                valid_return_1y,
                "forward_return_1y"
            ].to_numpy()
        )
    )

    print(
        f"Maximum 6M return difference: "
        f"{max_difference_6m:.12f}"
    )

    print(
        f"Maximum 1Y return difference: "
        f"{max_difference_1y:.12f}"
    )

    if max_difference_6m > 1e-10:
        raise RuntimeError(
            "6M return calculation mismatch."
        )

    if max_difference_1y > 1e-10:
        raise RuntimeError(
            "1Y return calculation mismatch."
        )

    print(
        "Forward-return calculations: PASS"
    )


# ============================================================
# EXTREME-RETURN COMPARISON
# ============================================================

def extreme_return_analysis(result):

    print("\n" + "=" * 70)
    print("POST-CORRECTION EXTREME RETURN ANALYSIS")
    print("=" * 70)

    for column, name in [
        (
            "forward_return_6m",
            "6M"
        ),
        (
            "forward_return_1y",
            "1Y"
        ),
    ]:

        valid = result[column].dropna()

        print(f"\n{name}:")

        thresholds = [
            (1.0, "> +100%"),
            (2.0, "> +200%"),
            (5.0, "> +500%"),
            (10.0, "> +1,000%"),
            (50.0, "> +5,000%"),
            (100.0, "> +10,000%"),
        ]

        for threshold, label in thresholds:

            count = (
                valid > threshold
            ).sum()

            print(
                f"{label:>12}: {count:,}"
            )

        print(
            f"\n{name} maximum: "
            f"{valid.max():.6f}"
        )

        print(
            f"{name} median: "
            f"{valid.median():.6f}"
        )

        print(
            f"{name} 99th percentile: "
            f"{valid.quantile(0.99):.6f}"
        )

        print(
            f"{name} 99.9th percentile: "
            f"{valid.quantile(0.999):.6f}"
        )


# ============================================================
# SPECIFIC SCHEME CHECK
# ============================================================

def corrected_scheme_check(result):

    print("\n" + "=" * 70)
    print("CORRECTED SCHEME CHECK")
    print("=" * 70)

    for scheme_code in [
        "120497",
        "120507",
    ]:

        scheme = result[
            result["scheme_code"]
            == scheme_code
        ]

        valid_6m = scheme[
            "forward_return_6m"
        ].dropna()

        valid_1y = scheme[
            "forward_return_1y"
        ].dropna()

        print(
            f"\nScheme {scheme_code}:"
        )

        print(
            f"6M max: "
            f"{valid_6m.max():.6f}"
        )

        print(
            f"1Y max: "
            f"{valid_1y.max():.6f}"
        )

        print(
            f"6M median: "
            f"{valid_6m.median():.6f}"
        )

        print(
            f"1Y median: "
            f"{valid_1y.median():.6f}"
        )

        extreme_6m = (
            valid_6m > 5.0
        ).sum()

        extreme_1y = (
            valid_1y > 5.0
        ).sum()

        print(
            f"6M returns > +500%: "
            f"{extreme_6m}"
        )

        print(
            f"1Y returns > +500%: "
            f"{extreme_1y}"
        )

        if extreme_6m != 0:
            raise RuntimeError(
                f"Residual 6M >500% anomaly "
                f"in scheme {scheme_code}."
            )

        if extreme_1y != 0:
            raise RuntimeError(
                f"Residual 1Y >500% anomaly "
                f"in scheme {scheme_code}."
            )


# ============================================================
# DISTRIBUTIONS
# ============================================================

def print_distributions(result):

    print("\n" + "=" * 70)
    print("CLEAN FORWARD RETURN DISTRIBUTIONS")
    print("=" * 70)

    for column, name in [
        (
            "forward_return_6m",
            "6M"
        ),
        (
            "forward_return_1y",
            "1Y"
        ),
    ]:

        print(
            f"\n{name} forward-return distribution:"
        )

        print(
            result[column]
            .dropna()
            .describe(
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
                    0.999,
                ]
            )
        )


# ============================================================
# SAVE
# ============================================================

def save_output(result):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEAN FORWARD OUTCOMES SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Schemes: "
        f"{result['scheme_code'].nunique()}"
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
        f"Reloaded rows: "
        f"{len(reloaded):,}"
    )

    print(
        f"Reloaded schemes: "
        f"{reloaded['scheme_code'].nunique()}"
    )

    if len(reloaded) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected row count after reload."
        )

    if (
        reloaded["scheme_code"].nunique()
        != EXPECTED_SCHEMES
    ):
        raise RuntimeError(
            "Unexpected scheme count after reload."
        )

    print(
        "Reload validation: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script uses the CLEAN mutual-fund NAV layer."
    )

    print(
        "The PostgreSQL source is not accessed or modified."
    )

    print(
        "No observations are deleted."
    )

    print(
        "No return clipping or labeling is performed."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    source = load_clean_nav()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_source(
        source
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = generate_forward_outcomes(
        source
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_forward_outcomes(
        source,
        result
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    extreme_return_analysis(
        result
    )

    corrected_scheme_check(
        result
    )

    print_distributions(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        result
    )

    # --------------------------------------------------------
    # Reload
    # --------------------------------------------------------

    reload_validation()

    print("\n" + "=" * 70)
    print("FINAL STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()