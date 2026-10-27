
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "clean" / "fundamentals"
NORMALIZATION_DIR = BASE / "normalization_candidate"
INTEGRATION_DIR = BASE / "integration_candidate"

SNAPSHOT_PATH = (
    INTEGRATION_DIR / "company_fundamental_snapshot_candidate.parquet"
)
OBSERVATIONS_PATH = (
    INTEGRATION_DIR / "latest_fundamental_observations_candidate.parquet"
)
REGISTER_PATH = (
    NORMALIZATION_DIR / "fundamental_mapping_approval_register.csv"
)

OUTPUT_DIR = INTEGRATION_DIR / "readiness"
REPORT_PATH = OUTPUT_DIR / "fundamental_metric_readiness.csv"
SUMMARY_PATH = OUTPUT_DIR / "fundamental_metric_readiness_summary.json"


def require_columns(df, required, label):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def parse_approval(series):
    """Interpret CSV booleans safely; bool('False') is True in Python."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def clean_field(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value


def main():
    for path in [SNAPSHOT_PATH, OBSERVATIONS_PATH, REGISTER_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    snapshot = pd.read_parquet(SNAPSHOT_PATH)
    observations = pd.read_parquet(OBSERVATIONS_PATH)
    register = pd.read_csv(REGISTER_PATH)

    require_columns(
        snapshot,
        ["ticker", "currency_quarantine", "fundamental_snapshot_status"],
        "Fundamental snapshot",
    )
    require_columns(
        observations,
        [
            "ticker", "source", "metric_raw", "period_end",
            "currency_quarantine", "scoring_eligible",
            "metric_mapping_approved",
        ],
        "Latest fundamental observations",
    )
    require_columns(
        register,
        [
            "metric", "screener_field", "yahoo_field",
            "decision", "approved_for_scoring",
        ],
        "Mapping approval register",
    )

    # Normalize join keys without changing the source artifacts.
    aliases = {"M_M.NS": "M&M.NS"}

    def canonical_ticker(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        return aliases.get(value, value)

    for df in [snapshot, observations]:
        df["ticker"] = df["ticker"].map(canonical_ticker)

    snapshot["currency_quarantine"] = (
        snapshot["currency_quarantine"].fillna(True).astype(bool)
    )
    observations["currency_quarantine"] = (
        observations["currency_quarantine"].fillna(True).astype(bool)
    )

    observations["period_end"] = pd.to_datetime(
        observations["period_end"], errors="coerce"
    )

    if snapshot["ticker"].duplicated().any():
        raise ValueError("Snapshot contains duplicate canonical tickers.")

    if register.empty:
        raise ValueError("Mapping approval register contains no mappings.")

    register = register.copy()
    register["approved_for_scoring"] = parse_approval(
        register["approved_for_scoring"]
    )

    # Preserve each register mapping as a separate candidate.
    # Do not infer approval from the presence of source data.
    tickers = snapshot[
        ["ticker", "currency_quarantine", "fundamental_snapshot_status"]
    ].sort_values("ticker")

    obs_by_ticker = {
        ticker: group
        for ticker, group in observations.groupby("ticker", dropna=False)
    }

    rows = []

    for mapping_index, mapping in register.reset_index(drop=True).iterrows():
        metric = str(mapping["metric"])
        screener_field = clean_field(mapping["screener_field"])
        yahoo_field = clean_field(mapping["yahoo_field"])
        mapping_approved = bool(mapping["approved_for_scoring"])

        for stock in tickers.itertuples(index=False):
            ticker = stock.ticker
            ticker_obs = obs_by_ticker.get(ticker)

            if ticker_obs is None:
                ticker_obs = observations.iloc[0:0]

            if screener_field:
                screener_rows = ticker_obs[
                    ticker_obs["metric_raw"].astype("string").str.strip()
                    == screener_field
                ]
            else:
                screener_rows = ticker_obs.iloc[0:0]

            if yahoo_field:
                yahoo_rows = ticker_obs[
                    ticker_obs["metric_raw"].astype("string").str.strip()
                    == yahoo_field
                ]
            else:
                yahoo_rows = ticker_obs.iloc[0:0]

            screener_dates = screener_rows["period_end"].dropna()
            yahoo_dates = yahoo_rows["period_end"].dropna()

            screener_date = (
                screener_dates.max().date().isoformat()
                if not screener_dates.empty else None
            )
            yahoo_date = (
                yahoo_dates.max().date().isoformat()
                if not yahoo_dates.empty else None
            )

            has_screener = not screener_rows.empty
            has_yahoo = not yahoo_rows.empty
            both_sources = has_screener and has_yahoo

            if bool(stock.currency_quarantine):
                readiness_status = "CURRENCY_REVIEW_REQUIRED"
            elif both_sources:
                readiness_status = (
                    "CROSS_SOURCE_EVIDENCE_AVAILABLE_REQUIRES_REVIEW"
                )
            elif has_screener or has_yahoo:
                readiness_status = "SINGLE_SOURCE_EVIDENCE_REQUIRES_REVIEW"
            else:
                readiness_status = "NO_LATEST_FIELD_EVIDENCE"

            rows.append({
                "ticker": ticker,
                "metric": metric,
                "screener_field": screener_field,
                "yahoo_field": yahoo_field,
                "screener_observation_present": has_screener,
                "yahoo_observation_present": has_yahoo,
                "both_fields_present": both_sources,
                "screener_latest_period_end": screener_date,
                "yahoo_latest_period_end": yahoo_date,
                "screener_latest_observation_count": int(
                    len(screener_rows)
                ),
                "yahoo_latest_observation_count": int(len(yahoo_rows)),
                "currency_quarantine": bool(stock.currency_quarantine),
                "mapping_register_decision": str(mapping["decision"]),
                "mapping_approved_for_scoring": mapping_approved,
                "readiness_status": readiness_status,
                "scoring_eligible": False,
            })

    report = pd.DataFrame(rows)

    # Keep the report diagnostic: no metric is approved by this script.
    if report["scoring_eligible"].any():
        raise AssertionError("Unexpected scoring eligibility detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    metric_summary = []
    for metric, group in report.groupby("metric", dropna=False):
        metric_summary.append({
            "metric": str(metric),
            "stock_mapping_rows": int(len(group)),
            "stocks_with_screener_field": int(
                group["screener_observation_present"].sum()
            ),
            "stocks_with_yahoo_field": int(
                group["yahoo_observation_present"].sum()
            ),
            "stocks_with_both_fields": int(
                group["both_fields_present"].sum()
            ),
            "stocks_in_currency_review": int(
                group["currency_quarantine"].sum()
            ),
            "approved_mapping_rows": int(
                group["mapping_approved_for_scoring"].sum()
            ),
            "scoring_eligible_rows": 0,
        })

    summary = {
        "stage": "fundamental_metric_readiness_diagnostic",
        "snapshot_tickers": int(snapshot["ticker"].nunique()),
        "latest_observation_rows": int(len(observations)),
        "mapping_register_rows": int(len(register)),
        "readiness_report_rows": int(len(report)),
        "stocks_with_any_currency_quarantine": int(
            snapshot["currency_quarantine"].sum()
        ),
        "readiness_status_counts": {
            str(key): int(value)
            for key, value in report["readiness_status"]
            .value_counts(dropna=False).items()
        },
        "metrics": metric_summary,
        "approved_mapping_rows": int(
            report["mapping_approved_for_scoring"].sum()
        ),
        "scoring_eligible_rows": 0,
        "outputs": [
            str(REPORT_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
        ],
        "limitations": [
            "Field presence does not establish metric equivalence.",
            "Field presence does not verify units or financial values.",
            "The report does not compare source values or approve mappings.",
            "Latest source observations can have different reporting dates.",
            "All rows remain ineligible for scoring.",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FUNDAMENTAL METRIC READINESS REPORT")
    print("=" * 72)
    print("Snapshot tickers:", summary["snapshot_tickers"])
    print("Latest observation rows:", summary["latest_observation_rows"])
    print("Mapping register rows:", summary["mapping_register_rows"])
    print("Readiness report rows:", summary["readiness_report_rows"])
    print("\nReadiness statuses:")

    for status, count in summary["readiness_status_counts"].items():
        print(f"  {status}: {count}")

    print("\nApproved mappings:", summary["approved_mapping_rows"])
    print("Scoring-eligible rows:", summary["scoring_eligible_rows"])
    print("\nReport:", REPORT_PATH.relative_to(ROOT))
    print("Summary:", SUMMARY_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
