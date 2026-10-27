
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "normalization_candidate"
    / "fundamental_observations_normalization_candidate.parquet"
)

PROFILE_PATH = (
    ROOT / "data" / "clean" / "fundamentals"
    / "yfinance"
    / "yfinance_company_snapshot.parquet"
)

OUTPUT_DIR = CANDIDATE_PATH.parent

QUARANTINED_TICKERS = {"HCLTECH.NS", "INFY.NS"}


def find_column(columns, candidates):
    """Return the first matching column, case-insensitively."""
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def value_counts(df, column):
    if column not in df.columns:
        return {"column_missing": column}
    values = df[column].astype("string").fillna("<NULL>")
    return {str(k): int(v) for k, v in values.value_counts().items()}


def main():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Candidate file not found: {CANDIDATE_PATH}")

    candidate = pd.read_parquet(CANDIDATE_PATH)

    required = {
        "ticker", "source", "reporting_basis", "statement_type",
        "metric_raw", "period_end", "period_type", "unit",
        "unit_verified", "financial_value_independently_verified",
        "scoring_eligible", "normalization_status",
        "normalization_candidate_unit", "fundamental_review_flag",
    }
    missing = required - set(candidate.columns)
    if missing:
        raise ValueError(f"Candidate schema missing columns: {sorted(missing)}")

    candidate["period_end"] = pd.to_datetime(
        candidate["period_end"], errors="coerce"
    )
    candidate["is_quarantined_ticker"] = candidate["ticker"].isin(
        QUARANTINED_TICKERS
    )

    print("=" * 72)
    print("FUNDAMENTAL METADATA AND UNIT PROVENANCE AUDIT")
    print("=" * 72)
    print("Candidate rows:", f"{len(candidate):,}")
    print("Distinct tickers:", candidate["ticker"].nunique())
    print("\nReporting basis:")
    print(candidate["reporting_basis"].value_counts(dropna=False).to_string())
    print("\nSource:")
    print(candidate["source"].value_counts(dropna=False).to_string())
    print("\nOriginal unit labels:")
    print(candidate["unit"].value_counts(dropna=False).head(30).to_string())
    print("\nNormalization candidate units:")
    print(
        candidate["normalization_candidate_unit"]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )

    print("\nVerification status:")
    for col in [
        "unit_verified",
        "financial_value_independently_verified",
        "scoring_eligible",
    ]:
        print(f"{col}:")
        print(candidate[col].value_counts(dropna=False).to_string())

    # Coverage by source, reporting basis and source unit.
    grouped = (
        candidate.groupby(
            ["source", "reporting_basis", "unit",
             "normalization_candidate_unit"],
            dropna=False,
        )
        .agg(
            observations=("ticker", "size"),
            distinct_tickers=("ticker", "nunique"),
            verified_unit_rows=(
                "unit_verified",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
            independently_verified_rows=(
                "financial_value_independently_verified",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
            scoring_eligible_rows=(
                "scoring_eligible",
                lambda x: int(x.fillna(False).astype(bool).sum()),
            ),
        )
        .reset_index()
        .sort_values(["source", "reporting_basis"], kind="stable")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped_path = OUTPUT_DIR / "metadata_unit_provenance_by_source.csv"
    grouped.to_csv(grouped_path, index=False)

    # Ticker-level currency provenance from the profile snapshot, if available.
    currency_audit = None
    profile_schema = []

    if PROFILE_PATH.exists():
        profile = pd.read_parquet(PROFILE_PATH)
        profile_schema = list(profile.columns)

        profile_ticker = find_column(profile.columns, ["ticker", "symbol"])
        quote_currency = find_column(
            profile.columns,
            ["currency", "quote_currency", "quoteCurrency"],
        )
        financial_currency = find_column(
            profile.columns,
            ["financial_currency", "financialCurrency",
             "financial_currency_code"],
        )
        profile_name = find_column(profile.columns, ["longName", "long_name"])

        if profile_ticker is None:
            print("\nProfile snapshot found, but no ticker column was recognized.")
        else:
            selected = [profile_ticker]
            rename = {profile_ticker: "ticker"}

            if quote_currency:
                selected.append(quote_currency)
                rename[quote_currency] = "profile_quote_currency"

            if financial_currency:
                selected.append(financial_currency)
                rename[financial_currency] = "profile_financial_currency"

            if profile_name:
                selected.append(profile_name)
                rename[profile_name] = "company_name"

            profile_small = profile[selected].rename(columns=rename)

            if profile_small["ticker"].duplicated().any():
                print(
                    "\nWARNING: duplicate tickers in profile snapshot; "
                    "currency join is not treated as one-to-one."
                )
                profile_small = profile_small.drop_duplicates(
                    "ticker", keep=False
                )

            ticker_summary = (
                candidate.groupby("ticker", dropna=False)
                .agg(
                    candidate_rows=("ticker", "size"),
                    sources=("source", lambda x: " | ".join(
                        sorted(set(x.dropna().astype(str)))
                    )),
                    reporting_bases=("reporting_basis", lambda x: " | ".join(
                        sorted(set(x.dropna().astype(str)))
                    )),
                    source_units=("unit", lambda x: " | ".join(
                        sorted(set(x.dropna().astype(str)))
                    )),
                    all_units_verified=(
                        "unit_verified",
                        lambda x: bool(
                            len(x) > 0 and x.fillna(False).astype(bool).all()
                        ),
                    ),
                    any_review_flag=(
                        "fundamental_review_flag",
                        lambda x: bool(x.fillna(False).astype(bool).any()),
                    ),
                )
                .reset_index()
            )

            currency_audit = ticker_summary.merge(
                profile_small, on="ticker", how="left", validate="one_to_one"
            )
            currency_audit["is_quarantined"] = currency_audit["ticker"].isin(
                QUARANTINED_TICKERS
            )

            currency_audit["currency_metadata_status"] = "NOT_COMPARABLE"
            if (
                "profile_quote_currency" in currency_audit.columns
                and "profile_financial_currency" in currency_audit.columns
            ):
                quote = currency_audit["profile_quote_currency"].astype(
                    "string"
                )
                financial = currency_audit[
                    "profile_financial_currency"
                ].astype("string")

                valid = quote.notna() & financial.notna()
                currency_audit.loc[valid, "currency_metadata_status"] = (
                    "CURRENCY_MATCH"
                )
                currency_audit.loc[
                    valid & quote.ne(financial), "currency_metadata_status"
                ] = "CURRENCY_MISMATCH_REVIEW"
            elif "profile_quote_currency" in currency_audit.columns:
                currency_audit["currency_metadata_status"] = (
                    "QUOTE_CURRENCY_ONLY_FINANCIAL_CURRENCY_UNAVAILABLE"
                )

            currency_path = OUTPUT_DIR / "metadata_currency_provenance_by_ticker.csv"
            currency_audit.to_csv(currency_path, index=False)

            print("\nProfile currency columns detected:")
            print("Quote currency:", quote_currency)
            print("Financial currency:", financial_currency)
            print("\nCurrency metadata status:")
            print(
                currency_audit["currency_metadata_status"]
                .value_counts(dropna=False)
                .to_string()
            )
            print("\nQuarantined tickers:")
            print(
                currency_audit[
                    currency_audit["is_quarantined"]
                ].to_string(index=False)
            )
            print("Currency audit output:", currency_path.relative_to(ROOT))
    else:
        print("\nProfile snapshot not found:", PROFILE_PATH)

    summary = {
        "audit": "fundamental_metadata_provenance",
        "candidate_rows": int(len(candidate)),
        "unique_tickers": int(candidate["ticker"].nunique()),
        "unit_verified_rows": int(
            candidate["unit_verified"].fillna(False).astype(bool).sum()
        ),
        "independently_verified_rows": int(
            candidate["financial_value_independently_verified"]
            .fillna(False).astype(bool).sum()
        ),
        "scoring_eligible_rows": int(
            candidate["scoring_eligible"].fillna(False).astype(bool).sum()
        ),
        "profile_snapshot_found": PROFILE_PATH.exists(),
        "profile_columns": [str(c) for c in profile_schema],
        "quarantined_tickers": sorted(QUARANTINED_TICKERS),
        "policy": (
            "Metadata inventory only. No units, reporting bases, or metrics "
            "are approved by this audit."
        ),
        "outputs": [grouped_path.name],
    }

    if currency_audit is not None:
        summary["outputs"].append("metadata_currency_provenance_by_ticker.csv")

    summary_path = OUTPUT_DIR / "metadata_provenance_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nNo input data modified. No metric approved for scoring.")


if __name__ == "__main__":
    main()
