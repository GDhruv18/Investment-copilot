"""
VALIDATE PERSISTENT ETF SCALE CHANGES
=====================================

Purpose
-------
Validate the two ETF price-scale discontinuities currently classified as
persistent:

    1. HDFCNEXT50.NS
       Event: 2023-10-20

    2. PSUBANK.NS
       Event: 2026-07-10

This script is READ-ONLY.

It does NOT:
    - modify PostgreSQL
    - modify Parquet files
    - generate corrected data
    - regenerate forward outcomes
    - change ETF targets

For each ETF, the audit checks:

    - Median price before the event
    - Median price after the event
    - Scale ratio
    - Candidate ~10x correction
    - OHLC consistency
    - Adj Close consistency
    - Volume behavior
    - Persistence of the new scale
    - Whether the candidate correction creates continuity
    - Candidate back-adjustment factor

The output is intended to determine whether the observed change should
eventually be treated as a genuine split-like scale event.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ETF_DATA_DIR = Path("data/raw/etfs")

EVENTS = {
    "HDFCNEXT50.NS": pd.Timestamp("2023-10-20"),
    "PSUBANK.NS": pd.Timestamp("2026-07-10"),
}

# Number of trading observations used for the analysis.
WINDOW = 20

# Candidate scale factors around 10x.
# We calculate the exact factor first, then test whether a simple
# approximately-10x transformation explains the discontinuity.
SCALE_TOLERANCE = 0.20


# ============================================================
# HELPERS
# ============================================================

def load_etf(ticker):
    """
    Load one ETF parquet file.

    Expected schema:
        Open
        High
        Low
        Close
        Adj Close
        Volume
    """

    filename = ticker.replace(".", "_") + ".parquet"
    path = ETF_DATA_DIR / filename

    if not path.exists():
        # Fallback: search for the ticker-derived file.
        candidates = list(ETF_DATA_DIR.glob(f"*{ticker.split('.')[0]}*.parquet"))

        if not candidates:
            raise FileNotFoundError(
                f"Could not find parquet file for {ticker} in {ETF_DATA_DIR}"
            )

        path = candidates[0]

    df = pd.read_parquet(path)

    df = df.copy()

    # Normalize index/date handling.
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Identify date column.
    possible_date_columns = ["Date", "date", "Datetime", "datetime"]

    date_column = None

    for column in possible_date_columns:
        if column in df.columns:
            date_column = column
            break

    if date_column is None:
        raise ValueError(
            f"Could not identify date column for {ticker}. "
            f"Columns found: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df[date_column]).dt.normalize()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{ticker} is missing required columns: {missing}"
        )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
    ].copy()

    df = df.sort_values("Date").drop_duplicates("Date")

    return df


def median_or_nan(series):
    """Return median as float, or NaN for an empty series."""

    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) == 0:
        return np.nan

    return float(series.median())


def safe_ratio(a, b):
    """Return a / b safely."""

    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan

    return float(a / b)


def pct_change(a, b):
    """
    Percentage change from a -> b.
    """

    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan

    return float((b / a - 1.0) * 100.0)


def describe_ohlc_consistency(row):
    """
    Check whether the OHLC fields all exhibit approximately the same
    scale transformation.

    We calculate:

        Open_after / Open_before
        High_after / High_before
        Low_after / Low_before
        Close_after / Close_before
        AdjClose_after / AdjClose_before

    using medians from the before/after windows.
    """

    ratios = {}

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]:
        before = row[f"{column}_before"]
        after = row[f"{column}_after"]

        ratios[column] = safe_ratio(after, before)

    valid_ratios = [
        value for value in ratios.values()
        if pd.notna(value)
    ]

    if not valid_ratios:
        return ratios, np.nan, "NO_DATA"

    median_ratio = float(np.median(valid_ratios))

    deviations = [
        abs(value / median_ratio - 1.0)
        for value in valid_ratios
        if median_ratio != 0
    ]

    max_deviation = max(deviations) if deviations else np.nan

    if pd.isna(max_deviation):
        status = "NO_DATA"
    elif max_deviation <= 0.03:
        status = "CONSISTENT"
    elif max_deviation <= 0.10:
        status = "MOSTLY_CONSISTENT"
    else:
        status = "INCONSISTENT"

    return ratios, median_ratio, status


def analyze_event(ticker, event_date):
    """
    Analyze one persistent scale event.
    """

    print("\n" + "=" * 90)
    print(f"PERSISTENT SCALE VALIDATION: {ticker}")
    print("=" * 90)

    print(f"Event date: {event_date.date()}")
    print(f"Window: {WINDOW} trading observations before/after")

    df = load_etf(ticker)

    print(f"Rows loaded: {len(df):,}")
    print(
        f"Full date range: "
        f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    )

    event_positions = df.index[df["Date"] == event_date].tolist()

    if not event_positions:
        raise ValueError(
            f"Exact event date {event_date.date()} not found for {ticker}"
        )

    event_position = event_positions[0]

    if event_position < WINDOW:
        raise ValueError(
            f"Not enough observations before event for {ticker}"
        )

    if event_position + WINDOW >= len(df):
        raise ValueError(
            f"Not enough observations after event for {ticker}"
        )

    before = df.iloc[event_position - WINDOW:event_position].copy()

    # Include the event date and subsequent observations.
    after = df.iloc[event_position:event_position + WINDOW].copy()

    # --------------------------------------------------------
    # Basic price statistics
    # --------------------------------------------------------

    print("\n--- PRICE LEVELS ---")

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]:
        before_median = median_or_nan(before[column])
        after_median = median_or_nan(after[column])

        ratio = safe_ratio(after_median, before_median)

        print(
            f"{column:10s} | "
            f"before median = {before_median:12.6f} | "
            f"after median = {after_median:12.6f} | "
            f"after/before = {ratio:10.6f}"
        )

    # --------------------------------------------------------
    # Scale ratio
    # --------------------------------------------------------

    close_before = median_or_nan(before["Close"])
    close_after = median_or_nan(after["Close"])

    close_ratio = safe_ratio(close_after, close_before)

    if pd.notna(close_ratio) and close_ratio != 0:
        candidate_back_adjustment = close_ratio
        candidate_forward_adjustment = 1.0 / close_ratio
    else:
        candidate_back_adjustment = np.nan
        candidate_forward_adjustment = np.nan

    print("\n--- SCALE RATIO ---")

    print(f"Median Close before event : {close_before:.6f}")
    print(f"Median Close after event  : {close_after:.6f}")
    print(f"After / Before            : {close_ratio:.6f}")

    if pd.notna(close_ratio):
        print(
            f"1 / ratio                 : "
            f"{1.0 / close_ratio:.6f}"
        )

    approximately_ten_x = (
        pd.notna(close_ratio)
        and abs(close_ratio - 0.1) <= SCALE_TOLERANCE * 0.1
    )

    approximately_tenth = (
        pd.notna(close_ratio)
        and abs(close_ratio - 10.0) <= SCALE_TOLERANCE
    )

    if approximately_ten_x:
        print("Interpretation: approximately 10x DOWNWARD scale change.")
    elif approximately_tenth:
        print("Interpretation: approximately 10x UPWARD scale change.")
    else:
        print("Interpretation: NOT a simple ~10x scale change.")

    # --------------------------------------------------------
    # OHLC consistency
    # --------------------------------------------------------

    print("\n--- OHLC / ADJ CLOSE CONSISTENCY ---")

    stats = {}

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]:
        before_value = median_or_nan(before[column])
        after_value = median_or_nan(after[column])

        stats[f"{column}_before"] = before_value
        stats[f"{column}_after"] = after_value

    ratios, median_ohlc_ratio, consistency_status = (
        describe_ohlc_consistency(stats)
    )

    for column, ratio in ratios.items():
        if pd.notna(ratio):
            print(
                f"{column:10s}: after/before = {ratio:.6f}"
            )
        else:
            print(
                f"{column:10s}: ratio unavailable"
            )

    print(
        f"Median OHLC/Adj Close ratio: "
        f"{median_ohlc_ratio:.6f}"
        if pd.notna(median_ohlc_ratio)
        else "Median ratio: unavailable"
    )

    print(
        f"Consistency status: {consistency_status}"
    )

    # --------------------------------------------------------
    # Adj Close vs Close
    # --------------------------------------------------------

    print("\n--- CLOSE vs ADJ CLOSE ---")

    before_close_adj_ratio = safe_ratio(
        median_or_nan(before["Adj Close"]),
        median_or_nan(before["Close"]),
    )

    after_close_adj_ratio = safe_ratio(
        median_or_nan(after["Adj Close"]),
        median_or_nan(after["Close"]),
    )

    print(
        f"Adj Close / Close before: "
        f"{before_close_adj_ratio:.6f}"
        if pd.notna(before_close_adj_ratio)
        else "Adj Close / Close before: unavailable"
    )

    print(
        f"Adj Close / Close after : "
        f"{after_close_adj_ratio:.6f}"
        if pd.notna(after_close_adj_ratio)
        else "Adj Close / Close after: unavailable"
    )

    if (
        pd.notna(before_close_adj_ratio)
        and pd.notna(after_close_adj_ratio)
    ):
        difference = abs(
            after_close_adj_ratio - before_close_adj_ratio
        )

        if difference <= 0.02:
            adj_status = "CONSISTENT"
        elif difference <= 0.10:
            adj_status = "SLIGHT_DIFFERENCE"
        else:
            adj_status = "MATERIAL_DIFFERENCE"

        print(f"Adj Close consistency: {adj_status}")

    # --------------------------------------------------------
    # Volume behavior
    # --------------------------------------------------------

    print("\n--- VOLUME BEHAVIOR ---")

    volume_before = median_or_nan(before["Volume"])
    volume_after = median_or_nan(after["Volume"])

    volume_ratio = safe_ratio(
        volume_after,
        volume_before,
    )

    print(
        f"Median volume before: {volume_before:,.2f}"
    )

    print(
        f"Median volume after : {volume_after:,.2f}"
    )

    print(
        f"Volume after/before : {volume_ratio:.6f}"
        if pd.notna(volume_ratio)
        else "Volume ratio unavailable"
    )

    if pd.notna(volume_ratio):
        print(
            f"Volume percentage change: "
            f"{pct_change(volume_before, volume_after):.2f}%"
        )

    print(
        "\nNOTE: Volume is NOT corrected using the price scale factor."
    )

    # --------------------------------------------------------
    # Persistence test
    # --------------------------------------------------------

    print("\n--- PERSISTENCE TEST ---")

    post_close = pd.to_numeric(
        after["Close"],
        errors="coerce"
    ).dropna()

    if len(post_close) > 0 and pd.notna(close_before):
        relative_to_old_scale = post_close / close_before

        median_relative = float(
            relative_to_old_scale.median()
        )

        print(
            f"Median post-event Close / pre-event median: "
            f"{median_relative:.6f}"
        )

        if approximately_ten_x:
            distance_from_expected = abs(
                median_relative - 0.1
            )

            if distance_from_expected <= 0.03:
                persistence_status = "PERSISTENT_~10X_LOWER_SCALE"
            else:
                persistence_status = "LOWER_SCALE_BUT_NOT_EXACTLY_10X"

        elif approximately_tenth:
            distance_from_expected = abs(
                median_relative - 10.0
            )

            if distance_from_expected <= 3.0:
                persistence_status = "PERSISTENT_~10X_HIGHER_SCALE"
            else:
                persistence_status = "HIGHER_SCALE_BUT_NOT_EXACTLY_10X"

        else:
            persistence_status = "PERSISTENT_OTHER_SCALE"

        print(
            f"Persistence classification: "
            f"{persistence_status}"
        )

    else:
        persistence_status = "INSUFFICIENT_DATA"

    # --------------------------------------------------------
    # Continuity test
    # --------------------------------------------------------

    print("\n--- CONTINUITY TEST ---")

    last_before = before.iloc[-1]
    first_after = after.iloc[0]

    raw_gap = safe_ratio(
        first_after["Close"],
        last_before["Close"],
    )

    print(
        f"Last Close before event : "
        f"{last_before['Close']:.6f}"
    )

    print(
        f"First Close at event    : "
        f"{first_after['Close']:.6f}"
    )

    print(
        f"Raw event ratio         : "
        f"{raw_gap:.6f}"
        if pd.notna(raw_gap)
        else "Raw event ratio unavailable"
    )

    # Apply the candidate correction to the event/post-event scale.
    #
    # Example:
    #   pre-event = 452
    #   post-event = 44
    #
    # If post-event prices are multiplied by approximately 10,
    # the transformed value should be close to the old scale.
    if pd.notna(close_ratio) and close_ratio > 0:
        correction_factor = 1.0 / close_ratio

        corrected_first_close = (
            first_after["Close"] * correction_factor
        )

        corrected_event_ratio = safe_ratio(
            corrected_first_close,
            last_before["Close"],
        )

        print(
            f"Candidate correction factor: "
            f"{correction_factor:.6f}"
        )

        print(
            f"Corrected first Close: "
            f"{corrected_first_close:.6f}"
        )

        print(
            f"Corrected event ratio: "
            f"{corrected_event_ratio:.6f}"
        )

        if pd.notna(corrected_event_ratio):
            continuity_gap = abs(
                corrected_event_ratio - 1.0
            )

            print(
                f"Continuity deviation: "
                f"{continuity_gap * 100:.2f}%"
            )

            if continuity_gap <= 0.03:
                continuity_status = "STRONG_CONTINUITY"
            elif continuity_gap <= 0.10:
                continuity_status = "REASONABLE_CONTINUITY"
            else:
                continuity_status = "POOR_CONTINUITY"

            print(
                f"Continuity status: {continuity_status}"
            )
        else:
            continuity_status = "UNAVAILABLE"

    else:
        correction_factor = np.nan
        corrected_first_close = np.nan
        corrected_event_ratio = np.nan
        continuity_status = "UNAVAILABLE"

    # --------------------------------------------------------
    # Event-day OHLC
    # --------------------------------------------------------

    print("\n--- EVENT-DAY OHLC ---")

    print(
        f"{'Field':10s} "
        f"{'Previous':>15s} "
        f"{'Event':>15s} "
        f"{'Raw Ratio':>15s}"
    )

    print("-" * 60)

    previous_row = before.iloc[-1]
    event_row = after.iloc[0]

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]:
        previous_value = previous_row[column]
        event_value = event_row[column]

        ratio = safe_ratio(
            event_value,
            previous_value,
        )

        print(
            f"{column:10s} "
            f"{previous_value:15.6f} "
            f"{event_value:15.6f} "
            f"{ratio:15.6f}"
        )

    # --------------------------------------------------------
    # Short post-event table
    # --------------------------------------------------------

    print("\n--- FIRST 10 OBSERVATIONS AFTER EVENT ---")

    display_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    print(
        after[display_columns]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Candidate conclusion
    # --------------------------------------------------------

    print("\n--- AUDIT CONCLUSION ---")

    if (
        approximately_ten_x
        and consistency_status == "CONSISTENT"
        and persistence_status.startswith("PERSISTENT")
        and continuity_status in [
            "STRONG_CONTINUITY",
            "REASONABLE_CONTINUITY",
        ]
    ):
        conclusion = (
            "STRONG EVIDENCE OF A PERSISTENT ~10X PRICE-SCALE CHANGE. "
            "A back-adjustment should be investigated."
        )

    elif (
        approximately_ten_x
        and persistence_status.startswith("PERSISTENT")
    ):
        conclusion = (
            "PERSISTENT ~10X SCALE CHANGE DETECTED, "
            "but additional validation is required before correction."
        )

    else:
        conclusion = (
            "The event does not cleanly satisfy the expected "
            "persistent ~10X scale-change pattern."
        )

    print(conclusion)

    print(
        "\nCandidate back-adjustment factor "
        f"(multiply post-event scale by): "
        f"{correction_factor:.6f}"
        if pd.notna(correction_factor)
        else "\nCandidate back-adjustment factor unavailable."
    )

    return {
        "ticker": ticker,
        "event_date": event_date,
        "close_before_median": close_before,
        "close_after_median": close_after,
        "close_ratio": close_ratio,
        "candidate_correction_factor": correction_factor,
        "ohlc_consistency": consistency_status,
        "persistence": persistence_status,
        "continuity": continuity_status,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("ETF PERSISTENT SCALE CHANGE VALIDATION")
    print("=" * 90)

    print(
        "\nREAD-ONLY AUDIT"
        "\nNo PostgreSQL rows will be changed."
        "\nNo Parquet files will be changed."
        "\nNo target files will be changed."
    )

    results = []

    for ticker, event_date in EVENTS.items():

        try:
            result = analyze_event(
                ticker,
                event_date,
            )

            results.append(result)

        except Exception as exc:

            print("\n" + "!" * 90)
            print(f"ERROR analyzing {ticker}")
            print(f"{type(exc).__name__}: {exc}")
            print("!" * 90)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    if not results:
        print("No ETF events were successfully analyzed.")
        return

    summary = pd.DataFrame(results)

    summary["event_date"] = summary["event_date"].dt.strftime(
        "%Y-%m-%d"
    )

    display_columns = [
        "ticker",
        "event_date",
        "close_before_median",
        "close_after_median",
        "close_ratio",
        "candidate_correction_factor",
        "ohlc_consistency",
        "persistence",
        "continuity",
    ]

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("IMPORTANT")
    print("=" * 90)

    print(
        """
This audit only determines whether the persistent scale changes are
consistent with a split-like price-scale event.

No correction has been applied.

If the evidence is strong, the next step will be to design a separate
clean ETF price layer:

    etf_prices (raw / immutable)
            |
            v
    ETF scale correction rules
            |
            v
    etf_prices_clean

The original raw data will remain untouched.
"""
    )


if __name__ == "__main__":
    main()