from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "etfs"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean" / "etfs"

CONFIG_PATH = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "etfs_42_validated.json"
)

AUDIT_DIR = PROJECT_ROOT / "data_pipeline" / "audits"
AUDIT_OUTPUT = AUDIT_DIR / "clean_etf_price_audit.csv"


# ============================================================
# EXPECTED UNIVERSE
# ============================================================

EXPECTED_ETF_COUNT = 41
REMOVED_RAW_TICKERS = {"DYNAMIC.NS"}


# ============================================================
# HELPERS
# ============================================================

def ticker_from_filename(path: Path) -> str:
    """
    Convert:
        BANKBEES_NS.parquet
    into:
        BANKBEES.NS
    """
    return path.stem.replace("_NS", ".NS")


def load_universe():
    """
    Load the locked 41-ETF universe from the JSON config.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ETF config file not found:\n{CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    if isinstance(config, dict) and "etfs" in config:
        universe = config["etfs"]
    elif isinstance(config, list):
        universe = config
    else:
        raise ValueError(
            "Unexpected ETF config format. "
            "Expected a list or an object containing an 'etfs' key."
        )

    tickers = []

    for item in universe:
        if isinstance(item, str):
            tickers.append(item)
        elif isinstance(item, dict):
            if "ticker" in item:
                tickers.append(item["ticker"])
            elif "symbol" in item:
                tickers.append(item["symbol"])
            else:
                raise ValueError(
                    f"Could not find ticker/symbol in config item: {item}"
                )
        else:
            raise ValueError(
                f"Unexpected ETF config item: {item}"
            )

    return set(tickers)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN ETF PRICE LAYER — INDEPENDENT AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check directories
    # --------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw ETF directory not found:\n{RAW_DIR}"
        )

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Clean ETF directory not found:\n{CLEAN_DIR}"
        )

    # --------------------------------------------------------
    # 2. Load locked universe
    # --------------------------------------------------------

    expected_tickers = load_universe()

    print(f"\nLocked ETF universe: {len(expected_tickers)}")

    if len(expected_tickers) != EXPECTED_ETF_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_ETF_COUNT} ETFs in config, "
            f"but found {len(expected_tickers)}."
        )

    if "DYNAMIC.NS" in expected_tickers:
        raise ValueError(
            "DYNAMIC.NS is still present in the locked ETF universe."
        )

    print("DYNAMIC.NS in locked universe: False")

    # --------------------------------------------------------
    # 3. Find raw and clean files
    # --------------------------------------------------------

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    clean_files = sorted(CLEAN_DIR.glob("*.parquet"))

    raw_tickers = {
        ticker_from_filename(path)
        for path in raw_files
    }

    clean_tickers = {
        ticker_from_filename(path)
        for path in clean_files
    }

    print(f"\nRaw files   : {len(raw_files)}")
    print(f"Clean files : {len(clean_files)}")

    # --------------------------------------------------------
    # 4. RAW vs CLEAN universe validation
    #
    # IMPORTANT:
    #
    # Raw contains 42 files because DYNAMIC.NS is intentionally
    # retained as an audit/source artifact.
    #
    # Clean contains exactly the locked 41 ETF universe.
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("RAW / CLEAN UNIVERSE VALIDATION")
    print("-" * 70)

    # Clean must contain exactly the locked 41 ETFs.
    if clean_tickers != expected_tickers:

        missing_from_clean = expected_tickers - clean_tickers
        unexpected_in_clean = clean_tickers - expected_tickers

        print("\nMissing from clean:")
        for ticker in sorted(missing_from_clean):
            print(f"  {ticker}")

        print("\nUnexpected in clean:")
        for ticker in sorted(unexpected_in_clean):
            print(f"  {ticker}")

        raise ValueError(
            "Clean ETF universe does not exactly match the locked "
            "41-ETF universe."
        )

    # Raw is allowed to contain exactly one extra ticker:
    # DYNAMIC.NS.
    unexpected_raw = raw_tickers - expected_tickers

    if unexpected_raw != REMOVED_RAW_TICKERS:

        print("\nUnexpected raw-only tickers:")
        for ticker in sorted(unexpected_raw):
            print(f"  {ticker}")

        raise ValueError(
            "Raw ETF directory contains unexpected files. "
            "Only DYNAMIC.NS is allowed outside the locked universe."
        )

    # Make sure every expected ETF still exists in raw.
    missing_from_raw = expected_tickers - raw_tickers

    if missing_from_raw:

        print("\nMissing from raw:")
        for ticker in sorted(missing_from_raw):
            print(f"  {ticker}")

        raise ValueError(
            "One or more locked ETFs are missing from raw data."
        )

    print("Clean universe check : PASS")
    print("Raw universe check   : PASS")
    print("DYNAMIC raw retained : PASS")

    # --------------------------------------------------------
    # 5. Basic per-file quality audit
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PER-FILE DATA QUALITY AUDIT")
    print("-" * 70)

    audit_rows = []

    total_clean_rows = 0
    files_with_problems = 0

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for path in clean_files:

        ticker = ticker_from_filename(path)

        try:
            df = pd.read_parquet(path)

            problems = []

            # ------------------------------------------------
            # Empty check
            # ------------------------------------------------

            if df.empty:
                problems.append("EMPTY_FILE")

            # ------------------------------------------------
            # Required columns
            # ------------------------------------------------

            missing_columns = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                problems.append(
                    "MISSING_COLUMNS:" + ",".join(missing_columns)
                )

            if not df.empty:

                # --------------------------------------------
                # Date/index handling
                # --------------------------------------------

                dates = pd.to_datetime(df.index, errors="coerce")

                if dates.isna().any():
                    problems.append("INVALID_DATES")

                # --------------------------------------------
                # Duplicate dates
                # --------------------------------------------

                if df.index.duplicated().any():
                    problems.append("DUPLICATE_DATES")

                # --------------------------------------------
                # Required numeric columns
                # --------------------------------------------

                for col in required_columns:

                    if col not in df.columns:
                        continue

                    numeric = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )

                    if numeric.isna().any():
                        problems.append(
                            f"NULL_OR_NON_NUMERIC:{col}"
                        )

                    if (numeric < 0).any():
                        problems.append(
                            f"NEGATIVE_VALUES:{col}"
                        )

                # --------------------------------------------
                # OHLC consistency
                # --------------------------------------------

                if all(
                    col in df.columns
                    for col in ["Open", "High", "Low", "Close"]
                ):

                    high = pd.to_numeric(
                        df["High"],
                        errors="coerce"
                    )

                    low = pd.to_numeric(
                        df["Low"],
                        errors="coerce"
                    )

                    open_ = pd.to_numeric(
                        df["Open"],
                        errors="coerce"
                    )

                    close = pd.to_numeric(
                        df["Close"],
                        errors="coerce"
                    )

                    invalid_ohlc = (
                        (high < low)
                        | (high < open_)
                        | (high < close)
                        | (low > open_)
                        | (low > close)
                    )

                    if invalid_ohlc.fillna(False).any():
                        problems.append("INVALID_OHLC_RELATIONSHIP")

                # --------------------------------------------
                # Daily return anomaly audit
                # --------------------------------------------

                if "Adj Close" in df.columns:

                    adj_close = pd.to_numeric(
                        df["Adj Close"],
                        errors="coerce"
                    )

                    returns = adj_close.pct_change().dropna()

                    # These are AUDIT flags only.
                    # We do not delete, clip, or modify data here.
                    if (returns > 0.50).any():
                        problems.append("DAILY_RETURN_GT_50_PERCENT")

                    if (returns < -0.50).any():
                        problems.append("DAILY_RETURN_LT_NEG_50_PERCENT")

                    if (returns > 1.00).any():
                        problems.append("DAILY_RETURN_GT_100_PERCENT")

                    if (returns < -0.90).any():
                        problems.append("DAILY_RETURN_LT_NEG_90_PERCENT")

            # ------------------------------------------------
            # Record results
            # ------------------------------------------------

            row_count = len(df)

            total_clean_rows += row_count

            if problems:
                files_with_problems += 1

            audit_rows.append(
                {
                    "ticker": ticker,
                    "file": path.name,
                    "rows": row_count,
                    "min_date": (
                        pd.to_datetime(df.index).min()
                        if not df.empty
                        else None
                    ),
                    "max_date": (
                        pd.to_datetime(df.index).max()
                        if not df.empty
                        else None
                    ),
                    "problem_count": len(problems),
                    "problems": ";".join(problems),
                }
            )

        except Exception as e:

            files_with_problems += 1

            audit_rows.append(
                {
                    "ticker": ticker,
                    "file": path.name,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "problem_count": 1,
                    "problems": f"READ_ERROR:{e}",
                }
            )

    # --------------------------------------------------------
    # 6. Save audit report
    # --------------------------------------------------------

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(
        AUDIT_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # 7. Print per-file problems
    # --------------------------------------------------------

    if files_with_problems > 0:

        print("\nFiles with problems:")

        problem_df = audit_df[
            audit_df["problem_count"] > 0
        ]

        for _, row in problem_df.iterrows():
            print(
                f"  {row['ticker']}: {row['problems']}"
            )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Locked ETF universe       : {len(expected_tickers)}")
    print(f"Raw ETF files             : {len(raw_files)}")
    print(f"Clean ETF files           : {len(clean_files)}")
    print(f"Total clean rows          : {total_clean_rows}")
    print(f"Files with quality issues : {files_with_problems}")

    print(f"\nAudit report:")
    print(AUDIT_OUTPUT)

    # --------------------------------------------------------
    # 9. Final status
    # --------------------------------------------------------

    if files_with_problems > 0:

        print("\n" + "=" * 70)
        print("FAIL: Clean ETF dataset contains quality issues.")
        print("=" * 70)

        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("PASS: Clean ETF price layer passed independent audit.")
    print("=" * 70)

    print("\nExpected architecture:")
    print("  Raw ETF files   : 42")
    print("  Locked ETFs     : 41")
    print("  Clean ETF files : 41")
    print("  Raw-only ticker : DYNAMIC.NS")
    print("\nRaw data was not modified.")
    print("Clean data was not modified.")
    print("This script only audits the dataset.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()