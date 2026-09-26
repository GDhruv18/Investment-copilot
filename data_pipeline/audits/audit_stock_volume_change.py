from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "data"
    / "features"
    / "stock_features.parquet"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "stocks"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOCK VOLUME_CHANGE_20D AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------------

    df = pd.read_parquet(FEATURE_FILE)

    print(
        f"Feature rows loaded : {len(df):,}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required = {
        "ticker",
        "date",
        "volume_change_20d",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Find NaN rows
    # --------------------------------------------------------

    nan_rows = df[
        df["volume_change_20d"].isna()
    ].copy()

    print(
        f"\nTotal volume_change_20d NaNs : "
        f"{len(nan_rows):,}"
    )

    # --------------------------------------------------------
    # NaNs by stock
    # --------------------------------------------------------

    print("\nNaNs BY STOCK")
    print("-" * 70)

    by_stock = (
        nan_rows
        .groupby("ticker")
        .size()
        .sort_values(ascending=False)
    )

    print(by_stock.to_string())

    # --------------------------------------------------------
    # Separate expected warm-up NaNs from extra NaNs
    #
    # First 20 observations of each stock are expected to have
    # no 20-day historical comparison.
    # --------------------------------------------------------

    df["observation_number"] = (
        df.groupby("ticker").cumcount()
    )

    warmup_nan = df[
        (df["observation_number"] < 20)
        & df["volume_change_20d"].isna()
    ]

    extra_nan = df[
        (df["observation_number"] >= 20)
        & df["volume_change_20d"].isna()
    ].copy()

    print(
        f"\nExpected warm-up NaNs : "
        f"{len(warmup_nan):,}"
    )

    print(
        f"Extra/internal NaNs   : "
        f"{len(extra_nan):,}"
    )

    # --------------------------------------------------------
    # Inspect extra NaNs
    # --------------------------------------------------------

    if len(extra_nan) > 0:

        print("\nINTERNAL NaN DETAILS")
        print("-" * 70)

        print(
            extra_nan[
                [
                    "ticker",
                    "date",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Reconstruct volume history from raw data
    # --------------------------------------------------------

    print("\nRECONSTRUCTING VOLUME CONDITIONS")
    print("-" * 70)

    diagnostics = []

    for ticker in extra_nan["ticker"].unique():

        raw_file = RAW_DIR / f"{ticker}.parquet"

        if not raw_file.exists():

            print(
                f"WARNING: raw file missing for {ticker}"
            )

            continue

        raw = pd.read_parquet(raw_file)

        raw = raw.copy()

        raw["date"] = pd.to_datetime(
            raw.index,
            errors="coerce"
        )

        raw = raw.sort_values("date").reset_index(
            drop=True
        )

        raw["Volume"] = pd.to_numeric(
            raw["Volume"],
            errors="coerce"
        )

        raw["volume_20d_ago"] = (
            raw["Volume"].shift(20)
        )

        ticker_extra = extra_nan[
            extra_nan["ticker"] == ticker
        ]

        for _, row in ticker_extra.iterrows():

            match = raw[
                raw["date"].dt.normalize()
                == row["date"].normalize()
            ]

            if len(match) == 0:

                diagnostics.append({
                    "ticker": ticker,
                    "date": row["date"],
                    "current_volume": np.nan,
                    "volume_20d_ago": np.nan,
                    "reason": "DATE_NOT_FOUND",
                })

                continue

            volume_now = match.iloc[0]["Volume"]

            volume_previous = match.iloc[0][
                "volume_20d_ago"
            ]

            if pd.isna(volume_previous):

                reason = "PREVIOUS_VOLUME_NAN"

            elif volume_previous == 0:

                reason = "PREVIOUS_VOLUME_ZERO"

            elif pd.isna(volume_now):

                reason = "CURRENT_VOLUME_NAN"

            else:

                reason = "OTHER"

            diagnostics.append({
                "ticker": ticker,
                "date": row["date"],
                "current_volume": volume_now,
                "volume_20d_ago": volume_previous,
                "reason": reason,
            })

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    # --------------------------------------------------------
    # Diagnostic summary
    # --------------------------------------------------------

    print("\nREASON SUMMARY")
    print("-" * 70)

    if len(diagnostics_df) > 0:

        reason_counts = (
            diagnostics_df["reason"]
            .value_counts()
        )

        print(
            reason_counts.to_string()
        )

    else:

        print("No internal NaNs found.")

    # --------------------------------------------------------
    # Show zero-denominator examples
    # --------------------------------------------------------

    zero_cases = diagnostics_df[
        diagnostics_df["reason"]
        == "PREVIOUS_VOLUME_ZERO"
    ]

    print(
        f"\nPrevious-volume-zero cases : "
        f"{len(zero_cases):,}"
    )

    if len(zero_cases) > 0:

        print("\nZERO-VOLUME EXAMPLES")
        print("-" * 70)

        print(
            zero_cases.head(100).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Show current-volume NaNs
    # --------------------------------------------------------

    current_nan_cases = diagnostics_df[
        diagnostics_df["reason"]
        == "CURRENT_VOLUME_NAN"
    ]

    print(
        f"\nCurrent-volume-NaN cases : "
        f"{len(current_nan_cases):,}"
    )

    if len(current_nan_cases) > 0:

        print(
            current_nan_cases.head(100).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save diagnostic report
    # --------------------------------------------------------

    output_file = (
        ROOT
        / "data"
        / "features"
        / "stock_volume_change_diagnostics.parquet"
    )

    diagnostics_df.to_parquet(
        output_file,
        index=False
    )

    print(
        f"\nSaved diagnostic file:"
        f"\n{output_file}"
    )

    print("\n" + "=" * 70)
    print("VOLUME_CHANGE_20D AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()