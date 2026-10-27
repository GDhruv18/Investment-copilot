
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "clean" / "fundamentals" / "normalization_candidate"
OUTPUT_DIR = BASE
DATA_PATH = BASE / "fundamental_observations_normalization_candidate.parquet"
CURRENCY_PATH = BASE / "ticker_currency_reference_audit.parquet"

PAIR_PATH = BASE / "monetary_scale_final_pairs.csv"
SUMMARY_PATH = BASE / "monetary_scale_final_summary.csv"
REPORT_PATH = BASE / "monetary_scale_final_report.json"

EXPECTED_SCALE = 1e7
LOG_TOLERANCE = 0.25

# Only exact source labels are accepted. No fuzzy substitutions.
CONFIG = {
    "revenue": {
        "screener_fields": ["Sales +", "Revenue +"],
        "yahoo_fields": ["Total Revenue", "Operating Revenue"],
        "screener_statement_types": [
            "annual_income_statement", "quarterly_income_statement"
        ],
        "yahoo_statement_types": [
            "annual_income_statement", "quarterly_income_statement"
        ],
        "period_group": "income",
    },
    "total_assets": {
        "screener_fields": ["Total Assets"],
        "yahoo_fields": ["Total Assets"],
        "screener_statement_types": ["balance_sheet"],
        "yahoo_statement_types": ["annual_balance_sheet"],
        "period_group": "balance_sheet",
    },
    "total_debt": {
        "screener_fields": ["Borrowings +", "Borrowing"],
        "yahoo_fields": ["Total Debt"],
        "screener_statement_types": ["balance_sheet"],
        "yahoo_statement_types": ["annual_balance_sheet"],
        "period_group": "balance_sheet",
    },
    "operating_cash_flow": {
        "screener_fields": ["Cash from Operating Activity +"],
        "yahoo_fields": [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        ],
        "screener_statement_types": ["cash_flow"],
        "yahoo_statement_types": ["annual_cash_flow"],
        "period_group": "cash_flow",
    },
}


def normalize_text(series):
    return series.astype("string").str.strip()


def main():
    for path in [DATA_PATH, CURRENCY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    df = pd.read_parquet(DATA_PATH)
    currency = pd.read_parquet(CURRENCY_PATH)

    required = [
        "ticker", "source", "statement_type", "metric_raw",
        "period_end", "original_numeric_value",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Candidate missing columns: {sorted(missing)}")

    df["ticker"] = normalize_text(df["ticker"])
    df["source"] = normalize_text(df["source"])
    df["statement_type"] = normalize_text(df["statement_type"])
    df["metric_raw"] = normalize_text(df["metric_raw"])
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["original_numeric_value"] = pd.to_numeric(
        df["original_numeric_value"], errors="coerce"
    )

    currency["ticker"] = normalize_text(currency["ticker"])
    currency_columns = [
        "ticker", "financial_currency", "currency_status"
    ]
    missing_currency = set(currency_columns) - set(currency.columns)
    if missing_currency:
        raise ValueError(
            f"Currency reference missing columns: {sorted(missing_currency)}"
        )

    currency = currency[currency_columns].drop_duplicates("ticker")
    if currency["ticker"].duplicated().any():
        raise ValueError("Currency reference has duplicate tickers.")

    # Report actual source coverage for each configured field first.
    coverage = []
    pair_frames = []

    for metric, cfg in CONFIG.items():
        for source_name, fields, statement_types in [
            (
                "Screener.in",
                cfg["screener_fields"],
                cfg["screener_statement_types"],
            ),
            (
                "Yahoo Finance via yfinance",
                cfg["yahoo_fields"],
                cfg["yahoo_statement_types"],
            ),
        ]:
            part = df[
                df["source"].eq(source_name)
                & df["metric_raw"].isin(fields)
                & df["statement_type"].isin(statement_types)
            ]

            for field in fields:
                field_part = part[part["metric_raw"].eq(field)]
                coverage.append({
                    "metric": metric,
                    "source": source_name,
                    "field": field,
                    "statement_types_configured": " | ".join(statement_types),
                    "observations": int(len(field_part)),
                    "tickers": int(field_part["ticker"].nunique()),
                    "dated_observations": int(
                        field_part["period_end"].notna().sum()
                    ),
                })

        screener = df[
            df["source"].eq("Screener.in")
            & df["metric_raw"].isin(cfg["screener_fields"])
            & df["statement_type"].isin(cfg["screener_statement_types"])
        ].copy()

        yahoo = df[
            df["source"].eq("Yahoo Finance via yfinance")
            & df["metric_raw"].isin(cfg["yahoo_fields"])
            & df["statement_type"].isin(cfg["yahoo_statement_types"])
        ].copy()

        # Compare exact ticker, exact date, exact canonical metric.
        # Source field names remain visible so alternate definitions
        # are never silently merged into one metric value.
        keys = ["ticker", "period_end"]

        def remove_ambiguous_rows(frame, label):
            group_keys = keys + ["metric_raw"]
            counts = frame.groupby(group_keys, dropna=False).size()
            ambiguous_keys = counts[counts > 1].index
            if len(ambiguous_keys) == 0:
                return frame

            ambiguous_index = pd.MultiIndex.from_frame(
                frame[group_keys]
            )
            clean = frame[
                ~ambiguous_index.isin(ambiguous_keys)
            ].copy()

            print(
                f"{metric}: excluded {len(frame) - len(clean)} "
                f"ambiguous {label} rows."
            )
            return clean

        screener = remove_ambiguous_rows(screener, "Screener")
        yahoo = remove_ambiguous_rows(yahoo, "Yahoo")

        if screener.empty or yahoo.empty:
            continue

        screener = screener.rename(columns={
            "metric_raw": "screener_field",
            "original_numeric_value": "screener_value",
        })
        yahoo = yahoo.rename(columns={
            "metric_raw": "yahoo_field",
            "original_numeric_value": "yahoo_value",
        })

        left = screener[
            keys + ["screener_field", "screener_value"]
        ]
        right = yahoo[
            keys + ["yahoo_field", "yahoo_value"]
        ]

        joined = left.merge(
            right,
            on=keys,
            how="inner",
            validate="many_to_many",
        )

        if joined.empty:
            continue

        joined["metric"] = metric
        joined["period_group"] = cfg["period_group"]

        joined = joined.merge(
            currency, on="ticker", how="left", validate="many_to_one"
        )

        joined["currency_mismatch_review"] = joined[
            "currency_status"
        ].eq("FINANCIAL_CURRENCY_MISMATCH_REVIEW")

        joined["currency_metadata_missing"] = (
            joined["currency_status"].isna()
            | joined["currency_status"].eq("MISSING_METADATA")
        )

        joined["ratio_yahoo_to_screener"] = np.nan
        valid = (
            joined["screener_value"].notna()
            & joined["yahoo_value"].notna()
            & joined["screener_value"].ne(0)
            & joined["yahoo_value"].ne(0)
        )
        joined.loc[valid, "ratio_yahoo_to_screener"] = (
            joined.loc[valid, "yahoo_value"].abs()
            / joined.loc[valid, "screener_value"].abs()
        )

        joined["near_1e7"] = (
            np.log10(joined["ratio_yahoo_to_screener"]).sub(7).abs()
            <= LOG_TOLERANCE
        )

        joined["sign_disagreement"] = (
            np.sign(joined["screener_value"])
            != np.sign(joined["yahoo_value"])
        )

        joined["review_status"] = "REVIEW"
        joined.loc[
            joined["currency_mismatch_review"], "review_status"
        ] = "CURRENCY_MISMATCH_REVIEW"
        joined.loc[
            ~joined["currency_mismatch_review"]
            & joined["currency_metadata_missing"],
            "review_status",
        ] = "CURRENCY_METADATA_MISSING"
        joined.loc[
            ~joined["currency_mismatch_review"]
            & ~joined["currency_metadata_missing"]
            & joined["near_1e7"]
            & ~joined["sign_disagreement"],
            "review_status",
        ] = "SCALE_PATTERN_ONLY_NOT_APPROVED"

        joined["unit_verified"] = False
        joined["scoring_eligible"] = False
        pair_frames.append(joined)

    coverage_df = pd.DataFrame(coverage)

    if pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
    else:
        pairs = pd.DataFrame()

    summary_rows = []
    if not pairs.empty:
        for (metric, period_group, screener_field, yahoo_field), group in (
            pairs.groupby(
                ["metric", "period_group", "screener_field", "yahoo_field"],
                dropna=False,
            )
        ):
            usable = group[
                ~group["currency_mismatch_review"]
                & ~group["currency_metadata_missing"]
                & group["ratio_yahoo_to_screener"].notna()
            ]

            summary_rows.append({
                "metric": metric,
                "period_group": period_group,
                "screener_field": screener_field,
                "yahoo_field": yahoo_field,
                "matched_pairs": int(len(group)),
                "distinct_tickers": int(group["ticker"].nunique()),
                "currency_mismatch_pairs": int(
                    group["currency_mismatch_review"].sum()
                ),
                "usable_for_scale_diagnostic": int(len(usable)),
                "median_ratio_nonmismatch": (
                    float(usable["ratio_yahoo_to_screener"].median())
                    if not usable.empty else None
                ),
                "pairs_near_1e7_nonmismatch": int(
                    usable["near_1e7"].sum()
                ) if not usable.empty else 0,
                "sign_disagreements": int(group["sign_disagreement"].sum()),
                "unit_verified_pairs": 0,
                "scoring_eligible_pairs": 0,
            })

    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_path = BASE / "monetary_scale_final_field_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    pairs.to_csv(PAIR_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    report = {
        "audit": "consolidated_monetary_scale_validation",
        "candidate_tickers": int(df["ticker"].nunique()),
        "configured_metrics": list(CONFIG),
        "coverage_rows": int(len(coverage_df)),
        "matched_pairs": int(len(pairs)),
        "matched_metrics": (
            sorted(pairs["metric"].dropna().unique().tolist())
            if not pairs.empty else []
        ),
        "metrics_without_matched_pairs": [
            metric for metric in CONFIG
            if pairs.empty or metric not in set(pairs["metric"])
        ],
        "unit_verified_pairs": 0,
        "scoring_eligible_pairs": 0,
        "policy": (
            "Read-only diagnostic. Exact date matches only. No currency "
            "conversion, normalization approval, or scoring approval."
        ),
        "outputs": [
            str(coverage_path.relative_to(ROOT)),
            str(PAIR_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("CONSOLIDATED MONETARY SCALE VALIDATION")
    print("=" * 72)
    print("Canonical tickers in candidate:", df["ticker"].nunique())
    print("Metrics configured:", ", ".join(CONFIG))
    print("Exact-date matched pairs:", len(pairs))
    print(
        "Metrics with pairs:",
        ", ".join(report["matched_metrics"]) or "None",
    )
    print(
        "Metrics without pairs:",
        ", ".join(report["metrics_without_matched_pairs"]) or "None",
    )

    print("\nField coverage:")
    print(coverage_df.to_string(index=False))

    print("\nScale summary:")
    if summary.empty:
        print("No exact-date pairs were found.")
    else:
        print(summary.to_string(index=False))

    print("\nUnit verified: 0")
    print("Scoring eligible: 0")
    print("\nReport:", REPORT_PATH.relative_to(ROOT))
    print("No source data modified; no conversion approved.")


if __name__ == "__main__":
    main()
