from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "stock_features.parquet"


# ============================================================
# HELPERS
# ============================================================

def normalize_calendar_dates(values):
    """
    Normalize dates while preserving the original calendar date.
    """

    if isinstance(values, pd.Series):

        dt = pd.to_datetime(values, errors="coerce")

        if isinstance(dt.dtype, pd.DatetimeTZDtype):
            return dt.dt.tz_localize(None).dt.normalize()

        return dt.dt.normalize()

    if isinstance(values, pd.DatetimeIndex):

        if values.tz is not None:
            return values.tz_localize(None).normalize()

        return values.normalize()

    dt = pd.to_datetime(values, errors="coerce")

    if isinstance(dt, pd.DatetimeIndex):

        if dt.tz is not None:
            return dt.tz_localize(None).normalize()

        return dt.normalize()

    return pd.Series(dt).dt.normalize()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOCK FEATURE KEY MISMATCH DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    print()
    print("Loading feature dataset...")

    features = pd.read_parquet(FEATURE_FILE)

    features["date"] = normalize_calendar_dates(
        features["date"]
    )

    feature_keys = features[
        ["ticker", "date"]
    ].copy()

    feature_keys = feature_keys.drop_duplicates()

    print(f"Feature rows: {len(feature_keys):,}")

    # --------------------------------------------------------
    # Build raw source keys
    # --------------------------------------------------------

    source_key_parts = []

    parquet_files = sorted(
        RAW_DIR.glob("*.parquet")
    )

    print()
    print(f"Raw files: {len(parquet_files)}")

    for file_path in parquet_files:

        ticker = file_path.stem

        df = pd.read_parquet(file_path)

        dates = normalize_calendar_dates(
            df.index
        )

        temp = pd.DataFrame({
            "ticker": ticker,
            "date": dates
        })

        source_key_parts.append(temp)

    source_keys = pd.concat(
        source_key_parts,
        ignore_index=True
    )

    source_keys = source_keys.drop_duplicates()

    print(
        f"Source rows: {len(source_keys):,}"
    )

    # --------------------------------------------------------
    # Compare keys
    # --------------------------------------------------------

    merged = source_keys.merge(
        feature_keys,
        on=["ticker", "date"],
        how="outer",
        indicator=True
    )

    source_missing = merged[
        merged["_merge"] == "left_only"
    ].copy()

    feature_missing = merged[
        merged["_merge"] == "right_only"
    ].copy()

    print()
    print("=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)

    print(
        f"Source keys missing from features: "
        f"{len(source_missing):,}"
    )

    print(
        f"Feature keys absent from source: "
        f"{len(feature_missing):,}"
    )

    # --------------------------------------------------------
    # Mismatch by ticker
    # --------------------------------------------------------

    if len(source_missing) > 0:

        print()
        print("=" * 70)
        print("SOURCE-ONLY KEYS BY TICKER")
        print("=" * 70)

        source_counts = (
            source_missing
            .groupby("ticker")
            .size()
            .sort_values(ascending=False)
        )

        print(source_counts.to_string())

    if len(feature_missing) > 0:

        print()
        print("=" * 70)
        print("FEATURE-ONLY KEYS BY TICKER")
        print("=" * 70)

        feature_counts = (
            feature_missing
            .groupby("ticker")
            .size()
            .sort_values(ascending=False)
        )

        print(feature_counts.to_string())

    # --------------------------------------------------------
    # Date ranges of mismatches
    # --------------------------------------------------------

    if len(source_missing) > 0:

        print()
        print("=" * 70)
        print("SOURCE-ONLY DATE RANGE")
        print("=" * 70)

        print(
            f"Min date: "
            f"{source_missing['date'].min()}"
        )

        print(
            f"Max date: "
            f"{source_missing['date'].max()}"
        )

    if len(feature_missing) > 0:

        print()
        print("=" * 70)
        print("FEATURE-ONLY DATE RANGE")
        print("=" * 70)

        print(
            f"Min date: "
            f"{feature_missing['date'].min()}"
        )

        print(
            f"Max date: "
            f"{feature_missing['date'].max()}"
        )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    if len(source_missing) > 0:

        print()
        print("=" * 70)
        print("FIRST 30 SOURCE-ONLY KEYS")
        print("=" * 70)

        print(
            source_missing[
                ["ticker", "date"]
            ]
            .sort_values(
                ["ticker", "date"]
            )
            .head(30)
            .to_string(index=False)
        )

    if len(feature_missing) > 0:

        print()
        print("=" * 70)
        print("FIRST 30 FEATURE-ONLY KEYS")
        print("=" * 70)

        print(
            feature_missing[
                ["ticker", "date"]
            ]
            .sort_values(
                ["ticker", "date"]
            )
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Detect possible one-day shifts
    # --------------------------------------------------------

    if len(source_missing) > 0 and len(feature_missing) > 0:

        print()
        print("=" * 70)
        print("POSSIBLE ONE-DAY DATE SHIFT CHECK")
        print("=" * 70)

        source_set = set(
            zip(
                source_missing["ticker"],
                source_missing["date"]
            )
        )

        feature_set = set(
            zip(
                feature_missing["ticker"],
                feature_missing["date"]
            )
        )

        forward_shift_matches = 0
        backward_shift_matches = 0

        for ticker, date in source_set:

            if (
                ticker,
                date + pd.Timedelta(days=1)
            ) in feature_set:

                forward_shift_matches += 1

            if (
                ticker,
                date - pd.Timedelta(days=1)
            ) in feature_set:

                backward_shift_matches += 1

        print(
            "Source date -> feature date +1 day matches: "
            f"{forward_shift_matches:,}"
        )

        print(
            "Source date -> feature date -1 day matches: "
            f"{backward_shift_matches:,}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()