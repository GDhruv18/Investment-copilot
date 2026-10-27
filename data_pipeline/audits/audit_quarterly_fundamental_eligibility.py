
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".")
RAW_ROOT = ROOT / "data/raw/fundamentals/yfinance"
CLEAN_FILE = ROOT / "data/clean/fundamentals/yfinance/yfinance_statement_observations.parquet"
PROFILE_FILE = ROOT / "data/clean/fundamentals/yfinance/yfinance_company_snapshot.parquet"
OUTPUT_DIR = ROOT / "data/processed/fundamentals"

OUTPUT_FILE = OUTPUT_DIR / "yfinance_quarterly_eligibility_audit.csv"
SUMMARY_FILE = OUTPUT_DIR / "yfinance_quarterly_eligibility_summary.json"

METRICS = {
    "revenue": [
        "Operating Revenue",
        "Total Revenue",
    ],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
}

ANNUAL_TYPE = "annual_income_statement"
RECONCILIATION_TOLERANCE = 0.25


def read_profile():
    df = pd.read_parquet(PROFILE_FILE)
    if "ticker" not in df.columns:
        raise ValueError("Profile is missing ticker column.")
    return df.drop_duplicates("ticker", keep="last").set_index("ticker")


def read_observations():
    df = pd.read_parquet(CLEAN_FILE)

    required = {
        "ticker", "statement_type", "financial_item",
        "period_end", "value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["ticker", "period_end"])


def read_raw_info(ticker):
    path = RAW_ROOT / ticker / "info.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data", {})
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def currency_status(ticker, profile):
    if ticker not in profile.index:
        return "CURRENCY_REVIEW", None, None

    quote_currency = profile.loc[ticker].get("currency")
    info = read_raw_info(ticker)
    financial_currency = info.get("financialCurrency")

    quote_currency = (
        str(quote_currency).strip().upper()
        if pd.notna(quote_currency) else None
    )
    financial_currency = (
        str(financial_currency).strip().upper()
        if financial_currency is not None else None
    )

    if not quote_currency or not financial_currency:
        return "CURRENCY_REVIEW", quote_currency, financial_currency

    if quote_currency != financial_currency:
        return "CURRENCY_REVIEW", quote_currency, financial_currency

    return "CURRENCY_MATCH", quote_currency, financial_currency


def choose_quarterly_metric(df, ticker, metric):
    rows = df[
        (df["ticker"] == ticker)
        & (df["statement_type"] == "quarterly_income_statement")
        & (df["financial_item"].isin(METRICS[metric]))
    ].copy()

    if rows.empty:
        return rows

    priority = {name: i for i, name in enumerate(METRICS[metric])}
    rows["priority"] = rows["financial_item"].map(priority)
    rows = rows.sort_values(["period_end", "priority"])
    return rows.drop_duplicates("period_end", keep="first").sort_values("period_end")


def choose_annual_metric(df, ticker, metric):
    rows = df[
        (df["ticker"] == ticker)
        & (df["statement_type"] == ANNUAL_TYPE)
        & (df["financial_item"].isin(METRICS[metric]))
    ].copy()

    if rows.empty:
        return np.nan, None

    priority = {name: i for i, name in enumerate(METRICS[metric])}
    rows["priority"] = rows["financial_item"].map(priority)
    rows = rows.sort_values(["period_end", "priority"])
    rows = rows.drop_duplicates("period_end", keep="first")
    rows = rows[np.isfinite(rows["value"])]

    if rows.empty:
        return np.nan, None

    latest = rows.iloc[-1]
    return float(latest["value"]), latest["period_end"]


def get_profile_value(profile, ticker, metric):
    if ticker not in profile.index:
        return np.nan

    field = {
        "revenue": "totalRevenue",
        "net_income": "netIncomeToCommon",
    }[metric]

    value = profile.loc[ticker].get(field, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    return value if np.isfinite(value) else np.nan


def latest_four_coverage(rows):
    if rows.empty:
        return [], [], "INSUFFICIENT_QUARTERS", False

    latest_dates = sorted(rows["period_end"].dropna().unique())[-4:]
    latest_rows = rows[rows["period_end"].isin(latest_dates)].copy()
    valid_rows = latest_rows[np.isfinite(latest_rows["value"])]

    if len(latest_dates) < 4:
        status = "INSUFFICIENT_QUARTERS"
    elif len(valid_rows) < 4:
        status = "MISSING_VALUES"
    else:
        # Do not infer eligibility from date spacing alone.
        # A six-month gap is not consecutive quarterly coverage.
        gaps = np.diff(
            pd.DatetimeIndex(latest_dates).values
        ).astype("timedelta64[D]").astype(int)
        if np.all((gaps >= 70) & (gaps <= 110)):
            status = "ELIGIBLE"
        else:
            status = "INSUFFICIENT_QUARTERS"

    return latest_dates, valid_rows, status, (
        len(latest_dates) == 4 and len(valid_rows) == 4
    )


def relative_difference(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = read_profile()
    observations = read_observations()

    tickers = sorted(
        set(profile.index.astype(str))
        | set(observations["ticker"].astype(str))
    )

    results = []

    for ticker in tickers:
        curr_status, quote_currency, financial_currency = currency_status(
            ticker, profile
        )

        for metric in METRICS:
            rows = choose_quarterly_metric(observations, ticker, metric)
            dates, valid_rows, coverage, four_valid = latest_four_coverage(rows)

            annual_value, annual_date = choose_annual_metric(
                observations, ticker, metric
            )
            profile_value = get_profile_value(profile, ticker, metric)

            quarterly_sum = (
                float(valid_rows["value"].sum())
                if coverage == "ELIGIBLE" and four_valid
                else np.nan
            )

            annual_diff = relative_difference(quarterly_sum, annual_value)
            profile_diff = relative_difference(quarterly_sum, profile_value)

            # Annual/profile comparisons are flagged only when the
            # quarter coverage and currency checks pass.
            if curr_status != "CURRENCY_MATCH":
                final_status = "CURRENCY_REVIEW"
            elif coverage != "ELIGIBLE":
                final_status = coverage
            elif (
                (np.isfinite(annual_diff) and annual_diff > RECONCILIATION_TOLERANCE)
                or
                (np.isfinite(profile_diff) and profile_diff > RECONCILIATION_TOLERANCE)
            ):
                final_status = "RECONCILIATION_REVIEW"
            else:
                final_status = "ELIGIBLE"

            results.append({
                "ticker": ticker,
                "metric": metric,
                "chosen_alias": (
                    rows["financial_item"].iloc[-1] if not rows.empty else None
                ),
                "latest_period": (
                    pd.Timestamp(max(dates)).date().isoformat() if dates else None
                ),
                "latest_four_dates": " | ".join(
                    pd.Timestamp(d).date().isoformat() for d in dates
                ),
                "latest_four_date_count": len(dates),
                "valid_latest_four_values": len(valid_rows),
                "coverage_status": coverage,
                "quote_currency": quote_currency,
                "financial_currency": financial_currency,
                "currency_status": curr_status,
                "latest_annual_value": annual_value,
                "latest_annual_period": (
                    pd.Timestamp(annual_date).date().isoformat()
                    if annual_date is not None else None
                ),
                "profile_value": profile_value,
                "quarter_sum_for_screening": quarterly_sum,
                "annual_relative_difference": annual_diff,
                "profile_relative_difference": profile_diff,
                "final_status": final_status,
            })

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_FILE, index=False)

    summary = {
        "audit": "quarterly_fundamental_eligibility",
        "read_only": True,
        "tickers_audited": int(result["ticker"].nunique()),
        "ticker_metric_rows": int(len(result)),
        "status_counts": {
            str(k): int(v)
            for k, v in result["final_status"].value_counts().items()
        },
        "coverage_status_counts": {
            str(k): int(v)
            for k, v in result["coverage_status"].value_counts().items()
        },
        "currency_status_counts": {
            str(k): int(v)
            for k, v in result["currency_status"].value_counts().items()
        },
        "currency_mismatch_tickers": sorted(
            result.loc[
                result["currency_status"] == "CURRENCY_REVIEW", "ticker"
            ].unique().tolist()
        ),
        "output_csv": str(OUTPUT_FILE),
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAudit complete.")
    print(json.dumps(summary, indent=2))
    print("\nCurrency examples:")
    print(
        result[result["ticker"].isin(
            ["HCLTECH.NS", "INFY.NS", "RELIANCE.NS"]
        )][[
            "ticker", "metric", "quote_currency",
            "financial_currency", "coverage_status",
            "currency_status", "final_status",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
