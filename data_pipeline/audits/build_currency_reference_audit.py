
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = ROOT / "data" / "raw" / "fundamentals" / "yfinance"
PROFILE_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "yfinance" / "yfinance_company_snapshot.parquet"
)
CANDIDATE_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
    / "fundamental_observations_normalization_candidate.parquet"
)
OUTPUT_DIR = ROOT / "data" / "clean" / "fundamentals" / "normalization_candidate"

OUTPUT_PATH = OUTPUT_DIR / "ticker_currency_reference_audit.parquet"
SUMMARY_PATH = OUTPUT_DIR / "ticker_currency_reference_audit_summary.json"

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}
TICKER_ALIASES = {
    "M_M.NS": "M&M.NS",
}


def canonical_ticker(value):
    if value is None:
        return None
    value = str(value).strip()
    return TICKER_ALIASES.get(value, value)

def read_info_json(path):
    """Read a raw Yahoo info.json and return its metadata dictionary."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    # The raw files may store fields inside a top-level "data" object.
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
        return obj["data"]
    return obj if isinstance(obj, dict) else {}


def nonempty(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def main():
    for path in [PROFILE_PATH, CANDIDATE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    profile = pd.read_parquet(PROFILE_PATH)
    candidate = pd.read_parquet(CANDIDATE_PATH)
    profile["ticker"] = profile["ticker"].map(canonical_ticker)
    for column in ["ticker", "currency"]:
        if column not in profile.columns:
            raise ValueError(
                f"Profile snapshot missing required column: {column}"
            )

    for column in [
        "ticker", "source", "reporting_basis", "unit",
        "unit_verified", "financial_value_independently_verified",
        "scoring_eligible",
    ]:
        if column not in candidate.columns:
            raise ValueError(
                f"Fundamental candidate missing required column: {column}"
            )

    if profile["ticker"].duplicated().any():
        raise ValueError("Duplicate ticker rows in profile snapshot.")

    # Recover financialCurrency from raw info.json files.
    raw_records = []
    for info_path in sorted(RAW_ROOT.glob("*/info.json")):
        ticker_folder = info_path.parent.name
        info = read_info_json(info_path)

        raw_records.append({
            "ticker": canonical_ticker(
                nonempty(info.get("symbol")) or ticker_folder
            ),
            "raw_ticker_folder": ticker_folder,
            "raw_quote_currency": nonempty(info.get("currency")),
            "financial_currency": nonempty(
                info.get("financialCurrency")
            ),
            "raw_exchange": nonempty(info.get("exchange")),
            "raw_full_exchange_name": nonempty(
                info.get("fullExchangeName")
            ),
            "raw_metadata_file": str(info_path.relative_to(ROOT)),
        })

    raw = pd.DataFrame(raw_records)

    if raw.empty:
        raise ValueError(f"No info.json files found under {RAW_ROOT}")

    if raw["ticker"].duplicated().any():
        duplicates = raw.loc[
            raw["ticker"].duplicated(keep=False), "ticker"
        ].tolist()
        raise ValueError(f"Duplicate raw ticker metadata: {duplicates}")

    # Count candidate observations and capture source/unit inventory.
    candidate["ticker"] = candidate["ticker"].map(canonical_ticker)
    candidate_summary = (
        candidate.groupby("ticker", dropna=False)
        .agg(
            candidate_observation_count=("ticker", "size"),
            source_count=("source", "nunique"),
            sources=(
                "source",
                lambda x: " | ".join(sorted(set(x.dropna().astype(str)))),
            ),
            reporting_bases=(
                "reporting_basis",
                lambda x: " | ".join(sorted(set(x.dropna().astype(str)))),
            ),
            source_units=(
                "unit",
                lambda x: " | ".join(sorted(set(x.dropna().astype(str)))),
            ),
            unit_verified_any=(
                "unit_verified",
                lambda x: bool(x.fillna(False).astype(bool).any()),
            ),
            independently_verified_any=(
                "financial_value_independently_verified",
                lambda x: bool(
                    x.fillna(False).astype(bool).any()
                ),
            ),
            scoring_eligible_any=(
                "scoring_eligible",
                lambda x: bool(x.fillna(False).astype(bool).any()),
            ),
        )
        .reset_index()
    )

    result = (
        profile[["ticker", "currency"]]
        .rename(columns={"currency": "profile_quote_currency"})
        .merge(raw, on="ticker", how="outer", validate="one_to_one")
        .merge(
            candidate_summary,
            on="ticker",
            how="outer",
            validate="one_to_one",
        )
    )

    # Compare quote currencies only when both metadata fields are available.
    result["currency_status"] = "MISSING_METADATA"

    both_quote = (
        result["profile_quote_currency"].notna()
        & result["raw_quote_currency"].notna()
    )
    quote_match = (
        result["profile_quote_currency"].astype("string")
        == result["raw_quote_currency"].astype("string")
    )
    result.loc[both_quote & quote_match, "currency_status"] = (
        "QUOTE_CURRENCY_MATCH"
    )
    result.loc[both_quote & ~quote_match, "currency_status"] = (
        "QUOTE_CURRENCY_CONFLICT_REVIEW"
    )

    has_financial = result["financial_currency"].notna()
    has_quote = result["raw_quote_currency"].notna()
    result.loc[
        has_financial & has_quote
        & result["financial_currency"].astype("string").eq(
            result["raw_quote_currency"].astype("string")
        ),
        "currency_status",
    ] = "QUOTE_AND_FINANCIAL_CURRENCY_MATCH"

    result.loc[
        has_financial & has_quote
        & result["financial_currency"].astype("string").ne(
            result["raw_quote_currency"].astype("string")
        ),
        "currency_status",
    ] = "FINANCIAL_CURRENCY_MISMATCH_REVIEW"

    result["currency_review_required"] = (
        result["currency_status"].isin([
            "MISSING_METADATA",
            "QUOTE_CURRENCY_CONFLICT_REVIEW",
            "FINANCIAL_CURRENCY_MISMATCH_REVIEW",
        ])
        | result["ticker"].isin(QUARANTINED_TICKERS)
    )

    result["unit_verified"] = False
    result["financial_value_independently_verified"] = False
    result["scoring_eligible"] = False

    result["quarantine_reason"] = None
    result.loc[
        result["ticker"].isin(QUARANTINED_TICKERS),
        "quarantine_reason",
    ] = "Quote currency INR but reported financial currency USD"

    result = result.sort_values("ticker").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    summary = {
        "audit": "ticker_currency_reference_audit",
        "profile_ticker_count": int(profile["ticker"].nunique()),
        "raw_info_file_count": int(len(raw)),
        "candidate_ticker_count": int(candidate["ticker"].nunique()),
        "reference_ticker_count": int(result["ticker"].nunique()),
        "currency_status_counts": {
            str(k): int(v)
            for k, v in result["currency_status"]
            .value_counts(dropna=False).items()
        },
        "financial_currency_counts": {
            str(k): int(v)
            for k, v in result["financial_currency"]
            .fillna("<MISSING>").value_counts().items()
        },
        "review_tickers": result.loc[
            result["currency_review_required"], "ticker"
        ].tolist(),
        "unit_verified_rows": int(result["unit_verified"].sum()),
        "independently_verified_rows": int(
            result["financial_value_independently_verified"].sum()
        ),
        "scoring_eligible_rows": int(result["scoring_eligible"].sum()),
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
        "policy": (
            "Audit reference only. No currency conversion, unit approval, "
            "financial-value approval, or scoring eligibility granted."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("TICKER CURRENCY REFERENCE AUDIT")
    print("=" * 72)
    print("Reference tickers:", len(result))
    print("\nCurrency status:")
    print(result["currency_status"].value_counts(dropna=False).to_string())
    print("\nFinancial currency:")
    print(
        result["financial_currency"]
        .fillna("<MISSING>")
        .value_counts()
        .to_string()
    )
    print("\nTickers requiring review:")
    print(
        result.loc[
            result["currency_review_required"],
            ["ticker", "profile_quote_currency", "raw_quote_currency",
             "financial_currency", "currency_status"],
        ].to_string(index=False)
    )
    print("\nUnit verified:", int(result["unit_verified"].sum()))
    print(
        "Independently verified:",
        int(result["financial_value_independently_verified"].sum()),
    )
    print("Scoring eligible:", int(result["scoring_eligible"].sum()))
    print("\nOutput:", OUTPUT_PATH.relative_to(ROOT))
    print("Summary:", SUMMARY_PATH.relative_to(ROOT))
    print("\nNo input data modified; no scoring approval granted.")


if __name__ == "__main__":
    main()
