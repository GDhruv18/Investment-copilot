
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
RAW_ROOT = ROOT / "data/raw/fundamentals/yfinance"
PROFILE_PATH = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_company_snapshot.parquet"
)
STATEMENT_PATH = (
    ROOT / "data/clean/fundamentals/yfinance/"
    "yfinance_statement_observations.parquet"
)
OUTPUT_DIR = ROOT / "data/processed/fundamentals"

# Match each profile field to its corresponding statement item.
FIELD_MAP = {
    "totalRevenue": ["Total Revenue", "Operating Revenue"],
    "netIncomeToCommon": [
        "Net Income Common Stockholders",
        "Net Income",
    ],
    "totalDebt": ["Total Debt"],
}

# Differences above these thresholds are investigation flags, not proof of error.
FLAG_THRESHOLDS = {
    "totalRevenue": 10.0,
    "netIncomeToCommon": 20.0,
    "totalDebt": 20.0,
}


def load_raw_profile(ticker):
    path = RAW_ROOT / ticker / "info.json"
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("data", {})
    except (OSError, json.JSONDecodeError):
        return {}


def relative_difference_pct(profile_value, statement_value):
    """Absolute percentage difference, using statement value as denominator."""
    try:
        p = float(profile_value)
        s = float(statement_value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(p) or not np.isfinite(s) or s == 0:
        return np.nan

    return abs(p - s) / abs(s) * 100.0


def main():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing profile dataset: {PROFILE_PATH}")
    if not STATEMENT_PATH.exists():
        raise FileNotFoundError(f"Missing statement dataset: {STATEMENT_PATH}")

    profiles = pd.read_parquet(PROFILE_PATH)
    statements = pd.read_parquet(STATEMENT_PATH)

    required_profile_columns = {"ticker", "currency"}
    required_statement_columns = {
        "ticker", "statement_type", "financial_item",
        "period_end", "value",
    }

    if not required_profile_columns.issubset(profiles.columns):
        raise ValueError(
            "Profile dataset missing columns: "
            f"{required_profile_columns - set(profiles.columns)}"
        )
    if not required_statement_columns.issubset(statements.columns):
        raise ValueError(
            "Statement dataset missing columns: "
            f"{required_statement_columns - set(statements.columns)}"
        )

    profiles = profiles.drop_duplicates("ticker", keep="last").copy()
    statements = statements.copy()
    statements["period_end"] = pd.to_datetime(
        statements["period_end"], errors="coerce"
    )
    statements["value"] = pd.to_numeric(
        statements["value"], errors="coerce"
    )

    # This audit compares annual statements only. It does not assume
    # the profile snapshot is the same reporting period.
    annual = statements[
        statements["statement_type"].isin([
            "annual_income_statement",
            "annual_balance_sheet",
        ])
    ].copy()

    rows = []

    for _, profile in profiles.iterrows():
        ticker = profile["ticker"]
        raw = load_raw_profile(ticker)
        quote_currency = raw.get("currency")
        financial_currency = raw.get("financialCurrency")

        for profile_field, statement_items in FIELD_MAP.items():
            if profile_field not in profiles.columns:
                continue

            profile_value = profile.get(profile_field)

            # For revenue and net income, prioritize the specific item
            # corresponding to the field; use the alternative only if
            # that item is unavailable at the latest statement period.
            if profile_field == "totalRevenue":
                statement_type = "annual_income_statement"
            elif profile_field == "netIncomeToCommon":
                statement_type = "annual_income_statement"
            else:
                statement_type = "annual_balance_sheet"

            candidates = annual[
                (annual["ticker"] == ticker)
                & (annual["statement_type"] == statement_type)
                & annual["financial_item"].isin(statement_items)
                & annual["value"].notna()
                & annual["period_end"].notna()
            ].copy()

            if candidates.empty:
                statement_value = np.nan
                statement_item = None
                statement_period = None
            else:
                # Choose the latest reporting date available for this ticker
                # and metric, then prefer the first item in FIELD_MAP.
                latest_date = candidates["period_end"].max()
                latest = candidates[candidates["period_end"] == latest_date]
                chosen = None

                for item in statement_items:
                    match = latest[latest["financial_item"] == item]
                    if not match.empty:
                        chosen = match.iloc[0]
                        break

                if chosen is None:
                    chosen = latest.iloc[0]

                statement_value = chosen["value"]
                statement_item = chosen["financial_item"]
                statement_period = chosen["period_end"].date().isoformat()

            discrepancy = relative_difference_pct(
                profile_value, statement_value
            )

            currency_status = (
                "CURRENCY_MISMATCH"
                if quote_currency and financial_currency
                and quote_currency != financial_currency
                else "CURRENCY_MATCH_OR_UNKNOWN"
            )

            if pd.isna(profile_value):
                status = "PROFILE_VALUE_MISSING"
            elif pd.isna(statement_value):
                status = "STATEMENT_VALUE_MISSING"
            elif currency_status == "CURRENCY_MISMATCH":
                status = "REVIEW_CURRENCY_BEFORE_COMPARISON"
            elif discrepancy > FLAG_THRESHOLDS[profile_field]:
                status = "REVIEW_LARGE_DIFFERENCE"
            else:
                status = "WITHIN_SCREENING_THRESHOLD"

            rows.append({
                "ticker": ticker,
                "profile_field": profile_field,
                "profile_value": profile_value,
                "profile_currency": financial_currency,
                "quote_currency": quote_currency,
                "statement_item": statement_item,
                "statement_value": statement_value,
                "statement_period": statement_period,
                "statement_currency": "NOT_INFERRED",
                "absolute_difference_pct": discrepancy,
                "screening_threshold_pct": FLAG_THRESHOLDS[profile_field],
                "currency_status": currency_status,
                "audit_status": status,
            })

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "profile_statement_discrepancy_audit.csv"
    result.to_csv(output_path, index=False)

    print("=" * 75)
    print("PROFILE VS STATEMENT DISCREPANCY AUDIT")
    print("=" * 75)
    print("Profile tickers:", profiles["ticker"].nunique())
    print("Comparison rows:", len(result))
    print("\nAudit status counts:")
    print(result["audit_status"].value_counts(dropna=False).to_string())

    print("\nLargest differences (excluding currency-mismatch cases):")
    comparable = result[
        result["currency_status"] == "CURRENCY_MATCH_OR_UNKNOWN"
    ].copy()
    comparable = comparable.sort_values(
        "absolute_difference_pct", ascending=False, na_position="last"
    )
    print(comparable[[
        "ticker", "profile_field", "profile_value",
        "statement_value", "statement_period",
        "absolute_difference_pct", "audit_status",
    ]].head(25).to_string(index=False))

    print("\nReport:", output_path)
    print("Read-only audit: no database, model, target, or source data changed.")


if __name__ == "__main__":
    main()
