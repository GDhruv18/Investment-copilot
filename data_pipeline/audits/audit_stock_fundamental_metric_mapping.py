
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    ROOT / "data_pipeline" / "config"
    / "stock_fundamental_metric_map.json"
)

OBSERVATIONS_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "integration_candidate"
    / "fundamental_observations_candidate.parquet"
)

PROFILE_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "integration_candidate"
    / "company_profile_snapshot.parquet"
)

EXPECTED_SOURCES = {
    "Screener.in",
    "Yahoo Finance via yfinance",
}


def stop(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_json(path: Path) -> dict:
    if not path.exists():
        stop(f"Mapping file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        stop(f"Invalid JSON at line {exc.lineno}: {exc.msg}")

    if not isinstance(data, dict):
        stop("Mapping JSON must contain a top-level object.")

    return data


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        stop(
            f"{name} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def fmt_date(series: pd.Series) -> str:
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return (
        f"{dates.min().date()} to {dates.max().date()}"
    )


def main() -> None:
    print("=" * 76)
    print("STOCK FUNDAMENTAL METRIC MAPPING AUDIT")
    print("READ-ONLY | NO OUTPUT FILES | SCORING DISABLED")
    print("=" * 76)

    mapping = load_json(MAPPING_PATH)

    metrics = mapping.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        stop("Mapping JSON must contain a non-empty 'metrics' object.")

    print(f"Mapping file: {MAPPING_PATH.relative_to(ROOT)}")
    print(f"Mapping status: {mapping.get('status', 'not specified')}")
    print(f"Canonical metrics: {len(metrics)}")

    if not OBSERVATIONS_PATH.exists():
        stop(f"Observations dataset not found: {OBSERVATIONS_PATH}")

    if not PROFILE_PATH.exists():
        stop(f"Profile dataset not found: {PROFILE_PATH}")

    observations = pd.read_parquet(OBSERVATIONS_PATH)
    profiles = pd.read_parquet(PROFILE_PATH)

    require_columns(
        observations,
        {
            "ticker",
            "source",
            "metric_raw",
            "statement_type",
            "period_end",
            "value",
        },
        "Fundamental observations",
    )
    require_columns(profiles, {"ticker"}, "Company profile snapshot")

    # Normalize only whitespace for matching; do not rewrite source data.
    for column in ("source", "metric_raw", "statement_type"):
        observations[column] = (
            observations[column].astype("string").str.strip()
        )

    observations["period_end"] = pd.to_datetime(
        observations["period_end"], errors="coerce"
    )

    print("\nDATASET OVERVIEW")
    print("-" * 76)
    print(
        f"Statement observations: {len(observations):,} rows | "
        f"{observations['ticker'].nunique():,} tickers"
    )
    print(
        f"Profile snapshot: {len(profiles):,} rows | "
        f"{profiles['ticker'].nunique():,} tickers"
    )
    print(
        "Observation sources: "
        + ", ".join(
            str(x) for x in sorted(
                observations["source"].dropna().unique()
            )
        )
    )

    unknown_sources = (
        set(observations["source"].dropna().unique())
        - EXPECTED_SOURCES
    )
    if unknown_sources:
        print(
            "NOTE: Additional source labels found: "
            + ", ".join(sorted(unknown_sources))
        )

    profile_fields = {
        "return_on_equity": ["returnOnEquity"],
        "return_on_assets": ["returnOnAssets"],
        "price_to_earnings": ["trailingPE", "forwardPE"],
        "price_to_book": ["priceToBook"],
        "operating_margin": ["operatingMargins"],
    }

    print("\nEXACT CANDIDATE MATCHES")
    print("-" * 76)

    matched_candidates = 0
    missing_candidates = 0
    matched_profile_fields = 0
    missing_profile_fields = 0

    for canonical_name, definition in metrics.items():
        if not isinstance(definition, dict):
            print(f"\n{canonical_name}: INVALID mapping entry")
            continue

        status = definition.get("mapping_status", "not specified")
        category = definition.get("category", "unspecified")
        candidates_by_source = definition.get("candidates", {})

        print(f"\n[{canonical_name}] category={category} | status={status}")

        # Check profile fields only when explicitly defined in candidates.
        for source, candidate_names in candidates_by_source.items():
            if not isinstance(candidate_names, list):
                print(f"  INVALID candidate list for source: {source}")
                continue

            if source == "Yahoo Finance via yfinance":
                for candidate in candidate_names:
                    if candidate in profiles.columns:
                        non_null = profiles[candidate].notna()
                        ticker_count = profiles.loc[
                            non_null, "ticker"
                        ].nunique()
                        matched_profile_fields += 1
                        print(
                            f"  PROFILE FOUND: {candidate} | "
                            f"{int(non_null.sum()):,}/{len(profiles):,} "
                            f"non-null rows | "
                            f"{ticker_count:,} tickers"
                        )
                    elif (
                        candidate in profile_fields.get(
                            canonical_name, []
                        )
                    ):
                        missing_profile_fields += 1
                        print(
                            f"  PROFILE MISSING: {candidate} "
                            "(expected profile field)"
                        )

            if not candidate_names:
                print(f"  {source}: no candidate names defined")
                continue

            for candidate in candidate_names:
                mask = (
                    observations["source"].eq(source)
                    & observations["metric_raw"].eq(candidate)
                )
                matched = observations.loc[mask]

                if matched.empty:
                    missing_candidates += 1
                    print(
                        f"  NOT FOUND: {source} -> {candidate}"
                    )
                    continue

                matched_candidates += 1
                ticker_count = matched["ticker"].nunique()
                row_count = len(matched)
                statement_types = sorted(
                    str(x)
                    for x in matched["statement_type"].dropna().unique()
                )

                print(
                    f"  FOUND: {source} -> {candidate} | "
                    f"{row_count:,} rows | "
                    f"{ticker_count:,} tickers"
                )
                print(
                    f"      statement types: "
                    f"{', '.join(statement_types) or 'n/a'}"
                )
                print(
                    f"      period range: "
                    f"{fmt_date(matched['period_end'])}"
                )

    print("\nAUDIT SUMMARY")
    print("-" * 76)
    print(f"Statement candidates found: {matched_candidates}")
    print(f"Statement candidates not found: {missing_candidates}")
    print(f"Profile fields found: {matched_profile_fields}")
    print(f"Expected profile fields missing: {missing_profile_fields}")

    print("\nIMPORTANT LIMITATIONS")
    print("- Exact name matches confirm presence, not semantic equivalence.")
    print("- Row counts do not establish complete ticker or period coverage.")
    print("- Currency, units, consolidation basis and accounting definitions")
    print("  still require validation before comparing providers.")
    print("- Current profile metrics are not historical point-in-time series.")
    print("- No metric is approved for scoring by this audit.")

    print("\nRESULT: MAPPING PRESENCE AUDIT COMPLETE")
    print("No datasets or mapping files were modified.")
    print("No CSV, Parquet, or JSON report was generated.")
    print("=" * 76)


if __name__ == "__main__":
    main()
