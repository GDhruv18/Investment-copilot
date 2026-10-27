from pathlib import Path
import json
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "config"
    / "etfs_42_validated.json"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "etfs"

CLEAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "etfs"
)

RULES_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "audits"
    / "etf_correction_rules.csv"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "data_pipeline"
    / "audits"
    / "etf_cleaning_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def ticker_from_filename(path: Path) -> str:
    """
    Convert:
        BANKBEES_NS.parquet
    to:
        BANKBEES.NS
    """

    stem = path.stem

    if stem.endswith("_NS"):
        return stem[:-3] + ".NS"

    return stem


def load_universe():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "etfs" not in data:
            raise ValueError(
                "ETF configuration does not contain 'etfs'."
            )

        tickers = data["etfs"]

    elif isinstance(data, list):

        tickers = data

    else:

        raise ValueError(
            "Unsupported ETF configuration format."
        )

    return set(tickers)


def load_rules():

    if not RULES_FILE.exists():

        raise FileNotFoundError(
            f"Correction rules not found:\n{RULES_FILE}"
        )

    rules = pd.read_csv(RULES_FILE)

    required = {
        "ticker",
        "event_date",
        "event_end_date",
        "event_type",
        "correction_direction",
        "correction_factor",
        "affected_scope",
        "confidence",
    }

    missing = required - set(rules.columns)

    if missing:

        raise ValueError(
            "Correction rules missing columns: "
            f"{sorted(missing)}"
        )

    rules["event_date"] = pd.to_datetime(
        rules["event_date"]
    )

    rules["event_end_date"] = pd.to_datetime(
        rules["event_end_date"],
        errors="coerce"
    )

    return rules


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN ETF PRICE LAYER — 41 ETF UNIVERSE")
    print("=" * 70)

    universe = load_universe()
    rules = load_rules()

    print(f"\nLocked ETF universe: {len(universe)}")

    if len(universe) != 41:

        raise ValueError(
            f"Expected 41 ETFs, found {len(universe)}."
        )

    if "DYNAMIC.NS" in universe:

        raise ValueError(
            "DYNAMIC.NS is still present in ETF universe."
        )

    print("DYNAMIC.NS present: False")

    # --------------------------------------------------------
    # Prepare clean directory
    # --------------------------------------------------------

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_clean = list(
        CLEAN_DIR.glob("*.parquet")
    )

    for path in existing_clean:
        path.unlink()

    print(
        f"Removed existing clean ETF files: "
        f"{len(existing_clean)}"
    )

    # --------------------------------------------------------
    # Validate correction rules
    # --------------------------------------------------------

    rule_tickers = set(
        rules["ticker"].dropna()
    )

    unknown_rule_tickers = (
        rule_tickers - universe
    )

    if unknown_rule_tickers:

        print(
            "\nIgnoring correction rules for "
            "tickers no longer in the ETF universe:"
        )

        for ticker in sorted(
            unknown_rule_tickers
        ):
            print(f"  {ticker}")

    # --------------------------------------------------------
    # Process only locked universe
    # --------------------------------------------------------

    raw_files = list(
        RAW_DIR.glob("*.parquet")
    )

    processed_tickers = set()

    processed = 0
    total_rows = 0
    changed_rows = 0
    changed_values = 0

    rules_applied = set()

    audit_rows = []

    for raw_file in raw_files:

        ticker = ticker_from_filename(
            raw_file
        )

        # Ignore DYNAMIC and any other file
        # outside the locked universe.
        if ticker not in universe:
            continue

        processed_tickers.add(ticker)

        df = pd.read_parquet(
            raw_file
        )

        # ----------------------------------------------------
        # Normalize Date
        # ----------------------------------------------------

        if not isinstance(
            df.index,
            pd.DatetimeIndex
        ):

            if "Date" in df.columns:

                df["Date"] = pd.to_datetime(
                    df["Date"]
                )

                df = df.set_index("Date")

            else:

                raise ValueError(
                    f"Could not identify Date "
                    f"for {ticker}"
                )

        df.index = pd.to_datetime(
            df.index
        )

        df = df.sort_index()

        original = df.copy()

        ticker_rules = rules[
            rules["ticker"].eq(ticker)
        ]

        ticker_changed_rows = set()
        ticker_changed_values = 0
        ticker_rules_used = []

        # ----------------------------------------------------
        # Apply correction rules
        # ----------------------------------------------------

        for _, rule in ticker_rules.iterrows():

            event_date = rule["event_date"]
            event_end_date = rule[
                "event_end_date"
            ]

            event_type = rule[
                "event_type"
            ]

            direction = rule[
                "correction_direction"
            ]

            factor = float(
                rule["correction_factor"]
            )

            # =================================================
            # TEMPORARY SCALE DISCONTINUITY
            # =================================================

            if (
                event_type
                == "TEMPORARY_SCALE_DISCONTINUITY"
            ):

                if pd.isna(event_end_date):

                    raise ValueError(
                        f"Temporary correction for "
                        f"{ticker} has no event_end_date."
                    )

                mask = (
                    (df.index >= event_date)
                    & (
                        df.index
                        <= event_end_date
                    )
                )

                if (
                    direction
                    != "MULTIPLY_ANOMALOUS_ROWS"
                ):

                    raise ValueError(
                        f"Unexpected correction direction "
                        f"for {ticker}: {direction}"
                    )

                for column in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                ]:

                    if column not in df.columns:
                        raise ValueError(
                            f"{ticker} missing column "
                            f"{column}"
                        )

                    df.loc[
                        mask,
                        column
                    ] = (
                        df.loc[
                            mask,
                            column
                        ] * factor
                    )

                affected_dates = (
                    df.index[mask]
                )

                ticker_changed_rows.update(
                    affected_dates
                )

                rules_applied.add(ticker)

                ticker_rules_used.append(
                    f"{event_type}"
                    f"|{event_date.date()}"
                    f"→{event_end_date.date()}"
                    f"|factor={factor}"
                )

            # =================================================
            # PERSISTENT SCALE CHANGE
            # =================================================

            elif (
                event_type
                == "PERSISTENT_SCALE_CHANGE"
            ):

                if (
                    direction
                    != "BACK_ADJUST_PRE_EVENT"
                ):

                    raise ValueError(
                        f"Unexpected persistent "
                        f"correction direction for "
                        f"{ticker}: {direction}"
                    )

                # Scale ALL observations before
                # the persistent event.
                mask = (
                    df.index < event_date
                )

                for column in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                ]:

                    if column not in df.columns:
                        raise ValueError(
                            f"{ticker} missing column "
                            f"{column}"
                        )

                    df.loc[
                        mask,
                        column
                    ] = (
                        df.loc[
                            mask,
                            column
                        ] * factor
                    )

                affected_dates = (
                    df.index[mask]
                )

                ticker_changed_rows.update(
                    affected_dates
                )

                rules_applied.add(ticker)

                ticker_rules_used.append(
                    f"{event_type}"
                    f"|before={event_date.date()}"
                    f"|factor={factor}"
                )

            else:

                raise ValueError(
                    f"Unknown event type for "
                    f"{ticker}: {event_type}"
                )

        # ----------------------------------------------------
        # Count actual changed values
        # ----------------------------------------------------

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
        ]

        for column in numeric_columns:

            before = original[column]
            after = df[column]

            changed_mask = (
                ~before.eq(after)
                & ~(
                    before.isna()
                    & after.isna()
                )
            )

            ticker_changed_values += int(
                changed_mask.sum()
            )

        ticker_changed_rows = set(
            ticker_changed_rows
        )

        ticker_changed_row_count = len(
            ticker_changed_rows
        )

        changed_rows += (
            ticker_changed_row_count
        )

        changed_values += (
            ticker_changed_values
        )

        total_rows += len(df)
        processed += 1

        # ----------------------------------------------------
        # Save clean file
        # ----------------------------------------------------

        output_file = (
            CLEAN_DIR
            / raw_file.name
        )

        df.to_parquet(
            output_file
        )

        audit_rows.append({
            "ticker": ticker,
            "raw_file": str(raw_file),
            "clean_file": str(output_file),
            "rows": len(df),
            "rows_changed": (
                ticker_changed_row_count
            ),
            "price_values_changed": (
                ticker_changed_values
            ),
            "rules_applied": len(
                ticker_rules_used
            ),
            "rule_details": "; ".join(
                ticker_rules_used
            ),
            "status": (
                "APPLIED_AND_VERIFIED"
            ),
        })

    # --------------------------------------------------------
    # Universe verification
    # --------------------------------------------------------

    missing = (
        universe - processed_tickers
    )

    unexpected = (
        processed_tickers - universe
    )

    if missing:

        raise ValueError(
            "ETF universe files missing:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    if unexpected:

        raise ValueError(
            "Unexpected tickers processed:\n"
            + "\n".join(
                sorted(unexpected)
            )
        )

    # --------------------------------------------------------
    # Verify exact clean-file count
    # --------------------------------------------------------

    clean_files = list(
        CLEAN_DIR.glob("*.parquet")
    )

    if len(clean_files) != 41:

        raise ValueError(
            f"Expected 41 clean ETF files, "
            f"found {len(clean_files)}."
        )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEAN ETF LAYER COMPLETE")
    print("=" * 70)

    print(
        f"Universe tickers : {len(universe)}"
    )

    print(
        f"Files processed  : {processed}"
    )

    print(
        f"Total rows       : {total_rows:,}"
    )

    print(
        f"Rows changed     : {changed_rows:,}"
    )

    print(
        f"Price values changed: "
        f"{changed_values:,}"
    )

    print(
        f"Tickers with rules applied: "
        f"{len(rules_applied)}"
    )

    print(
        f"\nClean directory:\n{CLEAN_DIR}"
    )

    print(
        f"\nAudit file:\n{AUDIT_FILE}"
    )

    print("\nDYNAMIC.NS was excluded.")
    print("RAW ETF DATA WAS NOT MODIFIED.")
    print("POSTGRESQL WAS NOT MODIFIED.")
    print("OLD TARGET FILE WAS NOT MODIFIED.")


if __name__ == "__main__":
    main()