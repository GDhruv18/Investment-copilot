
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

YF_PROFILE = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_company_snapshot.parquet"
)
YF_STATEMENTS = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_statement_observations.parquet"
)
SCREENER = (
    ROOT / "data/raw/fundamentals/screener_consolidated_100/"
    "consolidated_financials_long.csv"
)

OUT_DIR = ROOT / "data/clean/fundamentals/integration_candidate"
OBS_OUT = OUT_DIR / "fundamental_observations_candidate.parquet"
PROFILE_OUT = OUT_DIR / "company_profile_snapshot.parquet"
SUMMARY_OUT = OUT_DIR / "integration_candidate_summary.json"

EXPECTED_TICKERS = 100

# Normalize known source-specific ticker spelling differences.
# Keep the original spelling in ticker_source_raw for traceability.
TICKER_ALIASES = {
    "M_M.NS": "M&M.NS",
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def parse_dates(series):
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").astype("string")


def canonicalize_tickers(series):
    return (
        series.astype("string")
        .replace(TICKER_ALIASES)
    )


def build_yfinance_observations():
    df = pd.read_parquet(YF_STATEMENTS)

    required = {
        "ticker",
        "statement_type",
        "financial_item",
        "period_end",
        "value",
        "source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Yahoo statement dataset missing columns: "
            f"{sorted(missing)}"
        )

    source_tickers = df["ticker"].astype("string")

    result = pd.DataFrame({
        "ticker": canonicalize_tickers(source_tickers),
        "ticker_source_raw": source_tickers,
        "source": df["source"].astype("string"),
        "reporting_basis": "provider_reported",
        "statement_type": df["statement_type"].astype("string"),
        "metric_raw": df["financial_item"].astype("string"),
        "period_end": parse_dates(df["period_end"]),
        "period_type": df["statement_type"].map(
            lambda value: (
                "quarterly"
                if "quarter" in str(value).lower()
                else "annual"
                if "annual" in str(value).lower()
                else "provider_period_unspecified"
            )
        ),
        "value": pd.to_numeric(df["value"], errors="coerce"),
        "raw_value": df["value"].astype("string"),
        "unit": "provider_unit_unverified",
        "unit_verified": False,
        "financial_value_independently_verified": False,
    })

    result["scoring_eligible"] = False
    result["eligibility_reason"] = (
        "Pending unit, metric-definition and period validation"
    )

    return result


def build_screener_observations():
    df = pd.read_csv(SCREENER, low_memory=False)

    required = {
        "ticker",
        "reporting_basis",
        "section",
        "section_id",
        "table_index",
        "metric",
        "period_end",
        "raw_value",
        "numeric_value",
        "unit",
        "source",
        "unit_verified",
        "financial_value_independently_verified",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Screener dataset missing columns: "
            f"{sorted(missing)}"
        )

    parsed_dates = pd.to_datetime(df["period_end"], errors="coerce")

    # Keep dated rows from the actual statement table.
    # Undated summary tables are excluded from this observations dataset.
    keep = df["table_index"].eq(0) & parsed_dates.notna()
    dated = df.loc[keep].copy()
    parsed_dates = parsed_dates.loc[keep]

    source_tickers = dated["ticker"].astype("string")

    statement_map = {
        "Annual Profit and Loss": "annual_income_statement",
        "Quarterly Results": "quarterly_income_statement",
        "Balance Sheet": "balance_sheet",
        "Cash Flow": "cash_flow",
    }

    period_type_map = {
        "Annual Profit and Loss": "annual",
        "Quarterly Results": "quarterly",
        "Balance Sheet": "annual_or_point_in_time",
        "Cash Flow": "annual",
    }

    result = pd.DataFrame({
        "ticker": canonicalize_tickers(source_tickers),
        "ticker_source_raw": source_tickers,
        "source": dated["source"].astype("string"),
        "reporting_basis": dated["reporting_basis"].astype("string"),
        "statement_type": dated["section"].map(statement_map).fillna(
            "statement_type_unmapped"
        ),
        "metric_raw": dated["metric"].astype("string"),
        "period_end": parsed_dates.dt.strftime("%Y-%m-%d").astype("string"),
        "period_type": dated["section"].map(
            period_type_map
        ).fillna("period_type_unmapped"),
        "value": pd.to_numeric(
            dated["numeric_value"], errors="coerce"
        ),
        "raw_value": dated["raw_value"].astype("string"),
        "unit": dated["unit"].astype("string"),
        "unit_verified": (
            dated["unit_verified"].fillna(False).astype(bool)
        ),
        "financial_value_independently_verified": (
            dated["financial_value_independently_verified"]
            .fillna(False)
            .astype(bool)
        ),
    })

    result["scoring_eligible"] = False
    result["eligibility_reason"] = (
        "Pending unit verification and metric mapping"
    )

    return result


def main():
    for path in [YF_PROFILE, YF_STATEMENTS, SCREENER]:
        require_file(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_parquet(YF_PROFILE)

    if "ticker" not in profiles.columns:
        raise ValueError(
            "Yahoo profile dataset has no ticker column."
        )

    if profiles["ticker"].isna().any():
        raise ValueError(
            "Yahoo profile dataset contains null tickers."
        )

    if profiles["ticker"].duplicated().any():
        raise ValueError(
            "Yahoo profile dataset has duplicate ticker snapshots."
        )

    # Use the profile universe as the canonical stock universe.
    profiles["ticker"] = canonicalize_tickers(profiles["ticker"])
    profile_tickers = set(profiles["ticker"].astype(str))

    if len(profile_tickers) != EXPECTED_TICKERS:
        raise ValueError(
            f"Expected {EXPECTED_TICKERS} profile tickers, "
            f"found {len(profile_tickers)}."
        )

    observations = pd.concat(
        [
            build_yfinance_observations(),
            build_screener_observations(),
        ],
        ignore_index=True,
    )

    observations["ticker"] = observations["ticker"].astype("string")
    observations["ticker_source_raw"] = (
        observations["ticker_source_raw"].astype("string")
    )
    observations["period_end"] = observations["period_end"].astype("string")
    observations["metric_raw"] = observations["metric_raw"].astype("string")

    if observations["ticker"].isna().any():
        raise ValueError(
            "Integrated observations contain null tickers."
        )

    if observations["period_end"].isna().any():
        raise ValueError(
            "Integrated observations contain invalid dates."
        )

    observation_tickers = set(
        observations["ticker"].dropna().astype(str)
    )

    extra_tickers = observation_tickers - profile_tickers
    missing_tickers = profile_tickers - observation_tickers

    # Stop rather than silently accepting an unexpected universe.
    if extra_tickers:
        raise ValueError(
            "Observations contain tickers outside the profile universe: "
            f"{sorted(extra_tickers)}"
        )

    if missing_tickers:
        raise ValueError(
            "Profile tickers missing from observations: "
            f"{sorted(missing_tickers)}"
        )

    # All rows remain ineligible until the separate validation stage.
    observations["scoring_eligible"] = False

    # Save only candidate datasets; source inputs remain unchanged.
    profiles.to_parquet(PROFILE_OUT, index=False)
    observations.to_parquet(OBS_OUT, index=False)

    all_tickers = sorted(observation_tickers)

    source_counts = (
        observations.groupby("source", dropna=False)
        .size()
        .to_dict()
    )

    statement_counts = (
        observations.groupby(
            ["source", "statement_type"], dropna=False
        )
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
    )

    summary = {
        "status": "CANDIDATE_BUILT_REQUIRES_VALIDATION",
        "scoring_approved": False,
        "historical_ml_changed": False,
        "database_changed": False,
        "ticker_normalization": {
            "aliases_applied": TICKER_ALIASES,
            "profile_universe_tickers": len(profile_tickers),
            "observation_tickers": len(all_tickers),
            "extra_tickers": sorted(
                observation_tickers - profile_tickers
            ),
            "missing_tickers": sorted(
                profile_tickers - observation_tickers
            ),
        },
        "profile_snapshot": {
            "rows": int(len(profiles)),
            "unique_tickers": int(profiles["ticker"].nunique()),
            "path": str(PROFILE_OUT.relative_to(ROOT)),
        },
        "observations": {
            "rows": int(len(observations)),
            "unique_tickers": len(all_tickers),
            "expected_stock_universe": EXPECTED_TICKERS,
            "missing_numeric_values": int(
                observations["value"].isna().sum()
            ),
            "missing_dates": int(
                observations["period_end"].isna().sum()
            ),
            "unit_verified_rows": int(
                observations["unit_verified"].sum()
            ),
            "scoring_eligible_rows": int(
                observations["scoring_eligible"].sum()
            ),
            "earliest_period": observations["period_end"].min(),
            "latest_period": observations["period_end"].max(),
            "rows_by_source": {
                str(key): int(value)
                for key, value in source_counts.items()
            },
            "rows_by_source_and_statement": statement_counts,
            "path": str(OBS_OUT.relative_to(ROOT)),
        },
        "limitations": [
            "Source-specific metric names have not been fully mapped.",
            "Yahoo Finance statement units have not been independently verified.",
            "Screener.in units remain unverified.",
            "Duplicate metrics across sources are retained, not reconciled.",
            "No observation is approved for recommendation scoring.",
        ],
    }

    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nFundamental integration candidate built.")
    print(f"Profile snapshots: {len(profiles):,}")
    print(f"Statement observations: {len(observations):,}")
    print(f"Unique observation tickers: {len(all_tickers):,}")
    print(f"Missing numeric values: {observations['value'].isna().sum():,}")
    print(f"Missing dates: {observations['period_end'].isna().sum():,}")
    print(f"Scoring-approved rows: {observations['scoring_eligible'].sum():,}")
    print(f"Extra tickers: {len(extra_tickers)}")
    print(f"Missing tickers: {len(missing_tickers)}")
    print(f"Observations: {OBS_OUT.relative_to(ROOT)}")
    print(f"Profiles:     {PROFILE_OUT.relative_to(ROOT)}")
    print(f"Summary:      {SUMMARY_OUT.relative_to(ROOT)}")
    print("\nStatus: CANDIDATE BUILT — NOT APPROVED FOR SCORING")


if __name__ == "__main__":
    main()
