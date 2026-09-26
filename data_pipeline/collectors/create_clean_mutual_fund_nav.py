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

OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = OUTPUT_DIR / "mutual_fund_nav_clean.parquet"

EXPECTED_SCHEMES = 100

# ------------------------------------------------------------
# Validated NAV scale corrections
#
# The discontinuity occurs on 2013-04-22.
# Only observations on/after that date are corrected.
# ------------------------------------------------------------

CORRECTIONS = {
    "120497": {
        "start_date": pd.Timestamp("2013-04-22"),
        "factor": 0.009977962518,
    },
    "120507": {
        "start_date": pd.Timestamp("2013-04-22"),
        "factor": 0.009990251095,
    },
}


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
# LOAD LOCKED 100-FUND UNIVERSE
# ============================================================

def load_source_data():

    print("=" * 70)
    print("LOADING MUTUAL FUND NAV SOURCE DATA")
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
            "No mutual fund NAV data returned."
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
        errors="coerce",
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Schemes loaded: "
        f"{df['scheme_code'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source(df):

    print("\n" + "=" * 70)
    print("SOURCE VALIDATION")
    print("=" * 70)

    scheme_count = df["scheme_code"].nunique()

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
            "date",
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

    print("Source validation: PASS")


# ============================================================
# APPLY CORRECTIONS
# ============================================================

def apply_corrections(df):

    print("\n" + "=" * 70)
    print("APPLYING VALIDATED NAV CORRECTIONS")
    print("=" * 70)

    clean = df.copy()

    clean["nav_raw"] = clean["nav"]

    clean["correction_applied"] = False
    clean["correction_factor"] = 1.0

    total_rows_changed = 0
    total_schemes_changed = 0

    for scheme_code, rule in CORRECTIONS.items():

        mask = (
            (clean["scheme_code"] == scheme_code)
            &
            (clean["date"] >= rule["start_date"])
        )

        rows_affected = int(mask.sum())

        if rows_affected == 0:
            raise RuntimeError(
                f"No rows found for correction "
                f"rule {scheme_code}."
            )

        clean.loc[mask, "nav"] *= rule["factor"]

        clean.loc[
            mask,
            "correction_applied"
        ] = True

        clean.loc[
            mask,
            "correction_factor"
        ] = rule["factor"]

        total_rows_changed += rows_affected
        total_schemes_changed += 1

        print(
            f"\nScheme: {scheme_code}"
        )

        print(
            f"Correction start: "
            f"{rule['start_date'].date()}"
        )

        print(
            f"Correction factor: "
            f"{rule['factor']:.12f}"
        )

        print(
            f"Rows corrected: "
            f"{rows_affected:,}"
        )

    # --------------------------------------------------------
    # Verify number of changed NAV values
    # --------------------------------------------------------

    changed_mask = (
        ~np.isclose(
            clean["nav"].to_numpy(dtype=float),
            clean["nav_raw"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    )

    changed_values = int(
        changed_mask.sum()
    )

    print("\n" + "-" * 70)

    print(
        f"Correction rules applied: "
        f"{total_schemes_changed}/{len(CORRECTIONS)}"
    )

    print(
        f"Rows affected by rules: "
        f"{total_rows_changed:,}"
    )

    print(
        f"NAV values actually changed: "
        f"{changed_values:,}"
    )

    if changed_values != total_rows_changed:
        raise RuntimeError(
            "Expected every corrected row to have a changed NAV."
        )

    return clean


# ============================================================
# CLEAN-LAYER VALIDATION
# ============================================================

def validate_clean_layer(
    source,
    clean,
):

    print("\n" + "=" * 70)
    print("CLEAN NAV LAYER VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"Source rows: {len(source):,}"
    )

    print(
        f"Clean rows: {len(clean):,}"
    )

    if len(source) != len(clean):
        raise RuntimeError(
            "Clean layer changed row count."
        )

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    source_schemes = set(
        source["scheme_code"].unique()
    )

    clean_schemes = set(
        clean["scheme_code"].unique()
    )

    if source_schemes != clean_schemes:
        raise RuntimeError(
            "Clean layer changed the scheme universe."
        )

    print(
        f"Clean schemes: "
        f"{len(clean_schemes)}"
    )

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    duplicate_count = clean.duplicated(
        subset=[
            "scheme_code",
            "date",
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
    # NAV validity
    # --------------------------------------------------------

    invalid_nav = (
        clean["nav"].isna()
        | ~np.isfinite(clean["nav"])
        | (clean["nav"] <= 0)
    ).sum()

    print(
        f"Invalid clean NAV rows: "
        f"{invalid_nav}"
    )

    if invalid_nav != 0:
        raise RuntimeError(
            "Invalid NAV values detected after correction."
        )

    # --------------------------------------------------------
    # Uncorrected schemes must be identical
    # --------------------------------------------------------

    uncorrected_mask = (
        ~clean["correction_applied"]
    )

    source_uncorrected = source.loc[
        uncorrected_mask,
        [
            "scheme_code",
            "date",
            "nav",
        ],
    ].reset_index(drop=True)

    clean_uncorrected = clean.loc[
        uncorrected_mask,
        [
            "scheme_code",
            "date",
            "nav",
        ],
    ].reset_index(drop=True)

    if not source_uncorrected[
        ["scheme_code", "date"]
    ].equals(
        clean_uncorrected[
            ["scheme_code", "date"]
        ]
    ):
        raise RuntimeError(
            "Keys changed in uncorrected data."
        )

    if not np.allclose(
        source_uncorrected["nav"].to_numpy(
            dtype=float
        ),
        clean_uncorrected["nav"].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "NAV values changed outside correction rules."
        )

    print(
        "Uncorrected rows preserved exactly: PASS"
    )

    # --------------------------------------------------------
    # Corrected-row formula
    # --------------------------------------------------------

    corrected_mask = (
        clean["correction_applied"]
    )

    expected_corrected = (
        clean.loc[
            corrected_mask,
            "nav_raw"
        ].to_numpy(dtype=float)
        *
        clean.loc[
            corrected_mask,
            "correction_factor"
        ].to_numpy(dtype=float)
    )

    actual_corrected = clean.loc[
        corrected_mask,
        "nav"
    ].to_numpy(dtype=float)

    max_difference = np.max(
        np.abs(
            actual_corrected
            -
            expected_corrected
        )
    )

    print(
        f"Maximum correction calculation difference: "
        f"{max_difference:.12f}"
    )

    if max_difference > 1e-10:
        raise RuntimeError(
            "Correction calculation mismatch."
        )

    # --------------------------------------------------------
    # Correction counts
    # --------------------------------------------------------

    for scheme_code in CORRECTIONS:

        count = (
            (
                clean["scheme_code"]
                == scheme_code
            )
            &
            clean["correction_applied"]
        ).sum()

        print(
            f"{scheme_code} corrected rows: "
            f"{count:,}"
        )

    print(
        "Clean-layer structural validation: PASS"
    )


# ============================================================
# CONTINUITY VALIDATION
# ============================================================

def validate_continuity(clean):

    print("\n" + "=" * 70)
    print("POST-CORRECTION CONTINUITY VALIDATION")
    print("=" * 70)

    for scheme_code, rule in CORRECTIONS.items():

        scheme = clean[
            clean["scheme_code"]
            == scheme_code
        ].sort_values("date")

        jump_date = rule[
            "start_date"
        ]

        pre = scheme[
            scheme["date"] < jump_date
        ]

        post = scheme[
            scheme["date"] >= jump_date
        ]

        if pre.empty or post.empty:
            raise RuntimeError(
                f"Insufficient data for "
                f"continuity test: {scheme_code}"
            )

        last_pre = pre.iloc[-1]
        first_post = post.iloc[0]

        continuity_return = (
            float(first_post["nav"])
            /
            float(last_pre["nav"])
            - 1.0
        )

        print(
            f"\nScheme {scheme_code}"
        )

        print(
            f"Last pre-break date: "
            f"{last_pre['date'].date()}"
        )

        print(
            f"Last pre-break NAV: "
            f"{last_pre['nav']:.6f}"
        )

        print(
            f"First post-break date: "
            f"{first_post['date'].date()}"
        )

        print(
            f"First post-break NAV: "
            f"{first_post['nav']:.6f}"
        )

        print(
            f"Post-correction break return: "
            f"{continuity_return:.12f}"
        )

        # The original investigation established
        # exact continuity for the validated factors.
        if abs(continuity_return) > 1e-10:
            raise RuntimeError(
                f"Continuity check failed for "
                f"{scheme_code}: "
                f"{continuity_return}"
            )

    print(
        "\nContinuity validation: PASS"
    )


# ============================================================
# SAVE
# ============================================================

def save_clean_layer(clean):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Keep audit columns in the clean layer.
    clean.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEAN MUTUAL FUND NAV SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(clean):,}"
    )

    print(
        f"Schemes: "
        f"{clean['scheme_code'].nunique()}"
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

    if len(reloaded) != 206639:
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

    print("\n")
    print("=" * 70)
    print("CREATE CLEAN MUTUAL FUND NAV LAYER")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "PostgreSQL source table will NOT be modified."
    )

    print(
        "Only the two validated NAV scale corrections "
        "will be applied."
    )

    print(
        "No rows will be deleted."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    source = load_source_data()

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    validate_source(
        source
    )

    # --------------------------------------------------------
    # Apply corrections
    # --------------------------------------------------------

    clean = apply_corrections(
        source
    )

    # --------------------------------------------------------
    # Validate clean layer
    # --------------------------------------------------------

    validate_clean_layer(
        source,
        clean
    )

    # --------------------------------------------------------
    # Continuity
    # --------------------------------------------------------

    validate_continuity(
        clean
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_clean_layer(
        clean
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