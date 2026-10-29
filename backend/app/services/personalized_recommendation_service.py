
import logging
import math

from sqlalchemy import text

from backend.app.db.session import SessionLocal
import pandas as pd

from backend.app.core.config import PROJECT_ROOT
from backend.app.services.recommendation_service import load_recommendations


logger = logging.getLogger(__name__)


# Asset-class weights are percentages of the TOTAL investment amount.
# Unused amounts are not redistributed.
ASSET_CLASS_WEIGHTS = {
    "conservative": {
        "stocks": 0.00,
        "etfs": 0.25,
        "mutual_funds": 0.75,
    },
    "moderate": {
        "stocks": 0.30,
        "etfs": 0.30,
        "mutual_funds": 0.40,
    },
    "aggressive": {
        "stocks": 0.55,
        "etfs": 0.30,
        "mutual_funds": 0.15,
    },
}


# Maximum number of holdings selected in each asset class.
TOP_N_BY_ASSET_CLASS = {
    "stocks": 6,
    "etfs": 2,
    "mutual_funds": 3,
}


# Maximum allocation per instrument as a fraction of total investment.
MAX_INSTRUMENT_ALLOCATION = 0.10


VALID_ASSET_CLASSES = {
    "stocks",
    "etfs",
    "mutual_funds",
}

VALID_RISK_PROFILES = set(ASSET_CLASS_WEIGHTS)

VALID_HORIZONS = {
    "6m",
    "1y",
    "2y",
}


# Diversification proxy limits.
MAX_HOLDINGS_PER_DIVERSIFICATION_GROUP = {
    "stocks": 2,
    "etfs": 1,
    "mutual_funds": 2,
}


# Stock metadata source.
STOCK_PROFILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "fundamentals"
    / "yfinance"
    / "yfinance_company_snapshot.parquet"
)


# Optional canonical asset-name mapping.
# Expected columns:
# asset_class, ticker, asset_name
ASSET_NAME_MAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "asset_name_map.parquet"
)


def _load_stock_sector_metadata() -> pd.DataFrame:
    """
    Load stock sector classifications.

    If metadata is unavailable, recommendations continue without
    sector-based diversification for those stocks.
    """
    if not STOCK_PROFILE_FILE.exists():
        logger.warning(
            "Stock sector metadata not found at %s.",
            STOCK_PROFILE_FILE,
        )

        return pd.DataFrame(
            columns=["ticker", "sector"]
        )

    try:
        metadata = pd.read_parquet(STOCK_PROFILE_FILE)

    except Exception:
        logger.exception(
            "Unable to read stock sector metadata."
        )

        return pd.DataFrame(
            columns=["ticker", "sector"]
        )

    required = {
        "ticker",
        "sector",
    }

    if not required.issubset(metadata.columns):
        logger.warning(
            "Stock sector metadata is missing columns: %s",
            sorted(required - set(metadata.columns)),
        )

        return pd.DataFrame(
            columns=["ticker", "sector"]
        )

    metadata = metadata[
        ["ticker", "sector"]
    ].copy()

    metadata["ticker"] = (
        metadata["ticker"]
        .astype("string")
        .str.strip()
    )

    metadata["sector"] = (
        metadata["sector"]
        .astype("string")
        .str.strip()
    )

    metadata = metadata.dropna(
        subset=["ticker"]
    )

    metadata = metadata.drop_duplicates(
        subset=["ticker"],
        keep="last",
    )

    return metadata


def _attach_diversification_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach stock sector metadata.

    ETF and mutual-fund classification fields are retained
    if they already exist in the recommendation dataset.
    """
    df = df.copy()

    if "sector" not in df.columns:
        df["sector"] = pd.NA

    stock_mask = df["asset_class"].eq("stocks")

    if stock_mask.any():

        stock_metadata = _load_stock_sector_metadata()

        if not stock_metadata.empty:

            stock_rows = df.loc[
                stock_mask,
                ["ticker"],
            ].copy()

            stock_rows["_original_index"] = (
                stock_rows.index
            )

            stock_rows = stock_rows.merge(
                stock_metadata,
                on="ticker",
                how="left",
            )

            sector_by_index = (
                stock_rows
                .set_index("_original_index")["sector"]
            )

            mapped_sectors = pd.Series(
                df.loc[stock_mask].index.map(
                    sector_by_index
                ),
                index=df.loc[stock_mask].index,
                dtype="string",
            )

            df.loc[stock_mask, "sector"] = (
                df.loc[stock_mask, "sector"]
                .astype("string")
                .fillna(mapped_sectors)
            )

    return df


def _normalise_group_value(
    value,
) -> str | None:

    if value is None or pd.isna(value):
        return None

    text = " ".join(
        str(value).strip().casefold().split()
    )

    if not text:
        return None

    if text in {
        "nan",
        "none",
        "null",
        "unknown",
        "n/a",
        "na",
    }:
        return None

    return text


def _get_diversification_group(
    candidate: pd.Series,
    asset_class: str,
) -> str | None:
    """
    Identify the diversification group.

    Stocks:
        Sector or industry.

    ETFs:
        Underlying index or category.

    Mutual funds:
        Scheme category.

    These are classification proxies, not actual
    constituent-level holdings-overlap calculations.
    """

    if asset_class == "stocks":

        fields = (
            "sector",
            "industry",
        )

    elif asset_class == "etfs":

        fields = (
            "underlying_index",
            "index_name",
            "category",
            "fund_category",
        )

    else:

        fields = (
            "scheme_category",
            "fund_category",
            "category",
        )

    for field in fields:

        if field not in candidate.index:
            continue

        value = _normalise_group_value(
            candidate[field]
        )

        if value is not None:
            return value

    return None


def _select_top_n_diversified(
    candidates: pd.DataFrame,
    asset_class: str,
    top_n: int,
) -> pd.DataFrame:
    """
    Select the highest-scoring candidates while preferring
    different sectors, indices, or scheme categories.

    First pass:
        Enforce diversification group limits.

    Second pass:
        Fill remaining positions using score ranking.
    """

    if candidates.empty or top_n <= 0:
        return candidates.head(0).copy()

    ranked = (
        candidates
        .sort_values(
            "final_score",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    group_limit = (
        MAX_HOLDINGS_PER_DIVERSIFICATION_GROUP[
            asset_class
        ]
    )

    selected_indices = []
    group_counts = {}

    # Pass 1: Prefer different diversification groups.
    for index, candidate in ranked.iterrows():

        if len(selected_indices) >= top_n:
            break

        group = _get_diversification_group(
            candidate,
            asset_class,
        )

        group_key = (
            group
            if group is not None
            else "__unknown__"
        )

        if group_counts.get(group_key, 0) >= group_limit:
            continue

        selected_indices.append(index)

        group_counts[group_key] = (
            group_counts.get(group_key, 0) + 1
        )

    # Pass 2: Fill remaining positions by score.
    if len(selected_indices) < top_n:

        selected_set = set(selected_indices)

        for index in ranked.index:

            if index in selected_set:
                continue

            selected_indices.append(index)

            if len(selected_indices) >= top_n:
                break

    return (
        ranked
        .loc[selected_indices]
        .reset_index(drop=True)
    )


def _ticker_key(value) -> str:
    """
    Normalize ticker formats for metadata matching.

    Examples:
        KOTAKBANK_NS -> KOTAKBANK
        KOTAKBANK.NS -> KOTAKBANK
    """
    if value is None or pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace(".NS", "")
        .replace("_NS", "")
        .replace(".BO", "")
        .replace("_BO", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _first_non_empty(
    row: pd.Series,
    fields: tuple[str, ...],
) -> str | None:

    for field in fields:

        if field not in row.index:
            continue

        value = row[field]

        if value is None or pd.isna(value):
            continue

        value = " ".join(
            str(value).strip().split()
        )

        if value and value.casefold() not in {
            "nan",
            "none",
            "null",
            "n/a",
        }:
            return value

    return None


def _load_asset_name_map() -> pd.DataFrame:
    """
    Load optional asset-name mapping.

    Expected columns:
        asset_class
        ticker
        asset_name
    """

    if not ASSET_NAME_MAP_FILE.exists():

        return pd.DataFrame(
            columns=[
                "asset_class",
                "ticker",
                "asset_name",
            ]
        )

    try:

        names = pd.read_parquet(
            ASSET_NAME_MAP_FILE
        )

    except Exception:

        logger.exception(
            "Unable to read asset name map."
        )

        return pd.DataFrame(
            columns=[
                "asset_class",
                "ticker",
                "asset_name",
            ]
        )

    required = {
        "asset_class",
        "ticker",
        "asset_name",
    }

    if not required.issubset(names.columns):

        logger.warning(
            "Asset name map is missing columns: %s",
            sorted(required - set(names.columns)),
        )

        return pd.DataFrame(
            columns=[
                "asset_class",
                "ticker",
                "asset_name",
            ]
        )

    names = names[
        [
            "asset_class",
            "ticker",
            "asset_name",
        ]
    ].copy()

    names["_ticker_key"] = (
        names["ticker"].map(_ticker_key)
    )

    names["asset_name"] = (
        names["asset_name"]
        .astype("string")
        .str.strip()
    )

    names = names.dropna(
        subset=["asset_name"]
    )

    names = names.drop_duplicates(
        subset=[
            "asset_class",
            "_ticker_key",
        ],
        keep="last",
    )

    return names

def _load_mutual_fund_names() -> dict[str, str]:
    """
    Load official mutual fund scheme names from PostgreSQL.

    Returns:
        Mapping of scheme_code to scheme_name.
    """
    lookup = {}

    try:
        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT scheme_code, scheme_name
                    FROM mutual_funds
                    WHERE scheme_code IS NOT NULL
                      AND scheme_name IS NOT NULL
                """)
            ).mappings().all()

        for row in rows:
            scheme_code = str(row["scheme_code"]).strip()
            scheme_name = str(row["scheme_name"]).strip()

            if scheme_code and scheme_name:
                lookup[scheme_code] = scheme_name

    except Exception:
        logger.exception(
            "Unable to load mutual fund names from PostgreSQL."
        )

    return lookup

def _load_etf_names() -> dict[str, str]:
    """
    Load ETF names from PostgreSQL.
    """
    lookup = {}

    try:
        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT ticker, etf_name
                    FROM etfs
                    WHERE ticker IS NOT NULL
                      AND etf_name IS NOT NULL
                """)
            ).mappings().all()

        for row in rows:
            ticker = str(row["ticker"]).strip()
            etf_name = str(row["etf_name"]).strip()

            if ticker and etf_name:
                lookup[_ticker_key(ticker)] = etf_name

    except Exception:
        logger.exception(
            "Unable to load ETF names from PostgreSQL."
        )

    return lookup

def _build_asset_name_lookup(
    df: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    """
    Build asset-name lookup.

    Priority:
        1. Names in recommendation dataset.
        2. Stock company snapshot.
        3. Optional canonical asset-name mapping.
    """

    lookup = {}

    name_fields = (
        "asset_name",
        "common_name",
        "scheme_name",
        "fund_name",
        "company_name",
        "longName",
        "long_name",
        "instrument_name",
        "name",
    )

    # First: recommendation dataset.
    for _, row in df.iterrows():

        asset_class = str(
            row.get("asset_class", "")
        )

        ticker = str(
            row.get("ticker", "")
        )

        name = _first_non_empty(
            row,
            name_fields,
        )

        if name:

            lookup[
                (
                    asset_class,
                    _ticker_key(ticker),
                )
            ] = name

    # Second: stock company names.
    if STOCK_PROFILE_FILE.exists():

        try:

            profiles = pd.read_parquet(
                STOCK_PROFILE_FILE
            )

            if {
                "ticker",
                "longName",
            }.issubset(profiles.columns):

                stock_names = profiles[
                    ["ticker", "longName"]
                ].dropna()

                for _, row in stock_names.iterrows():

                    key = (
                        "stocks",
                        _ticker_key(row["ticker"]),
                    )

                    lookup.setdefault(
                        key,
                        str(row["longName"]).strip(),
                    )

        except Exception:

            logger.exception(
                "Unable to load stock names."
            )

    # Third: canonical mapping.
    name_map = _load_asset_name_map()

    if not name_map.empty:
        for _, row in name_map.iterrows():
            key = (
                str(row["asset_class"]),
                str(row["_ticker_key"]),
            )

            lookup.setdefault(
                key,
                str(row["asset_name"]).strip(),
            )

    # Fourth: official mutual fund names from PostgreSQL.
    mutual_fund_names = _load_mutual_fund_names()

    for scheme_code, scheme_name in mutual_fund_names.items():
        lookup[
            ("mutual_funds", _ticker_key(scheme_code))
        ] = scheme_name

    # Fifth: official ETF names from PostgreSQL.
    etf_names = _load_etf_names()

    for ticker, etf_name in etf_names.items():
        lookup[
            ("etfs", ticker)
        ] = etf_name

    return lookup


def _display_asset_name(
    asset_class: str,
    ticker: str,
    name_lookup: dict[tuple[str, str], str],
) -> str:

    name = name_lookup.get(
        (
            asset_class,
            _ticker_key(ticker),
        )
    )

    if name:
        return name

    # Do not invent a mutual-fund scheme name from its numeric code.
    if (
        asset_class == "mutual_funds"
        and str(ticker).strip().isdigit()
    ):
        return f"Mutual Fund Scheme {ticker}"

    cleaned = (
        str(ticker)
        .replace("_NS", "")
        .replace(".NS", "")
    )

    return cleaned.replace(
        "_",
        " ",
    ).strip()


def get_personalized_recommendations(
    investment_amount: float,
    risk_profile: str,
    horizon: str,
    asset_classes: list[str] | None = None,
) -> dict:
    """
    Generate a personalized Top-N portfolio.

    Policies:
        1. Preserve existing asset-class weights.
        2. Select at most 6 stocks, 2 ETFs, 3 mutual funds.
        3. Prefer different sectors / indices / scheme categories.
        4. Allocate proportionally using positive finite final_score.
        5. Maximum 10% of total investment per instrument.
        6. Do not redistribute unavailable or capped amounts.
        7. Excluded asset classes retain their original budgets.
    """

    try:

        investment_amount = float(
            investment_amount
        )

    except (TypeError, ValueError):

        raise ValueError(
            "Investment amount must be a valid number."
        )

    if not math.isfinite(investment_amount):

        raise ValueError(
            "Investment amount must be finite."
        )

    if investment_amount <= 0:

        raise ValueError(
            "Investment amount must be positive."
        )

    if risk_profile not in VALID_RISK_PROFILES:

        raise ValueError(
            "Invalid risk profile."
        )

    if horizon not in VALID_HORIZONS:

        raise ValueError(
            "Invalid investment horizon."
        )

    if asset_classes is not None:

        if not asset_classes:

            raise ValueError(
                "Select at least one asset class."
            )

        if not set(asset_classes).issubset(
            VALID_ASSET_CLASSES
        ):

            raise ValueError(
                "Invalid asset class selection."
            )

    selected_asset_classes = (
        set(asset_classes)
        if asset_classes is not None
        else VALID_ASSET_CLASSES
    )

    # Preserve original asset-class budgets.
    asset_class_budgets = {
        asset_class: investment_amount * weight
        for asset_class, weight
        in ASSET_CLASS_WEIGHTS[risk_profile].items()
    }

    df = load_recommendations()

    required_columns = {
        "candidate_status",
        "risk_profile",
        "horizon",
        "asset_class",
        "ticker",
        "final_score",
    }

    missing = (
        required_columns - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Recommendation data is missing required "
            f"columns: {sorted(missing)}"
        )

    # Filter eligible candidates.
    df = df[
        (df["candidate_status"] == "RECOMMENDATION_CANDIDATE")
        & (df["risk_profile"] == risk_profile)
        & (df["horizon"] == horizon)
        & (df["asset_class"].isin(selected_asset_classes))
    ].copy()

    # Convert scores to numeric values.
    df["final_score"] = pd.to_numeric(
        df["final_score"],
        errors="coerce",
    )

    # Only positive finite scores are eligible.
    valid_score_mask = df["final_score"].map(
        lambda score: (
            pd.notna(score)
            and math.isfinite(float(score))
            and float(score) > 0
        )
    )

    df = df.loc[
        valid_score_mask
    ].copy()

    # Attach available diversification metadata.
    df = _attach_diversification_metadata(df)

    # Build asset-name lookup.
    asset_name_lookup = (
        _build_asset_name_lookup(df)
    )

    recommendations = []

    # Process each asset class independently.
    for asset_class, budget in asset_class_budgets.items():

        if (
            asset_class not in selected_asset_classes
            or budget <= 0
        ):
            continue

        class_candidates = df[
            df["asset_class"] == asset_class
        ].copy()

        if class_candidates.empty:
            continue

        # Select Top-N diversified candidates.
        selected = _select_top_n_diversified(
            candidates=class_candidates,
            asset_class=asset_class,
            top_n=TOP_N_BY_ASSET_CLASS[asset_class],
        )

        if selected.empty:
            continue

        total_score = float(
            selected["final_score"].sum()
        )

        if (
            not math.isfinite(total_score)
            or total_score <= 0
        ):
            continue

        remaining_budget = round(
            budget,
            2,
        )

        maximum_amount = (
            investment_amount
            * MAX_INSTRUMENT_ALLOCATION
        )

        # Allocate proportionally to selected candidates.
        for rank, (_, candidate) in enumerate(
            selected.iterrows(),
            start=1,
        ):

            score = float(
                candidate["final_score"]
            )

            proportional_amount = (
                budget * score / total_score
            )

            allocation_amount = min(
                proportional_amount,
                maximum_amount,
                remaining_budget,
            )

            allocation_amount = round(
                allocation_amount,
                2,
            )

            if allocation_amount <= 0:
                continue

            remaining_budget = round(
                remaining_budget - allocation_amount,
                2,
            )

            allocation_percentage = (
                allocation_amount
                / investment_amount
                * 100
            )

            reason = candidate.get("reason")

            if (
                reason is not None
                and pd.isna(reason)
            ):
                reason = None

            ticker = str(
                candidate["ticker"]
            )

            recommendations.append(
                {
                    "asset_class": asset_class,

                    "asset_name": _display_asset_name(
                        asset_class=asset_class,
                        ticker=ticker,
                        name_lookup=asset_name_lookup,
                    ),

                    "ticker": ticker,

                    "allocation_percentage": round(
                        allocation_percentage,
                        4,
                    ),

                    "allocation_amount": allocation_amount,

                    "final_score": score,

                    "rank": rank,

                    "reason": reason,
                }
            )

    # Calculate total allocated and unallocated amounts.
    allocated_amount = round(
        sum(
            item["allocation_amount"]
            for item in recommendations
        ),
        2,
    )

    unallocated_amount = round(
        investment_amount - allocated_amount,
        2,
    )

    # Sort the final response by recommendation score.
    recommendations.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return {
        "investment_amount": round(
            investment_amount,
            2,
        ),

        "allocated_amount": allocated_amount,

        "unallocated_amount": unallocated_amount,

        "risk_profile": risk_profile,

        "horizon": horizon,

        "recommendations": recommendations,
    }
