from pathlib import Path

import pandas as pd

from backend.app.core.config import PROJECT_ROOT


RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "recommendation_engine"
    / "final_recommendations.parquet"
)

REQUIRED_COLUMNS = {
    "risk_profile",
    "asset_class",
    "ticker",
    "horizon",
    "prediction",
    "prob_attractive",
    "prob_weak",
    "final_score",
    "rank_within_group",
    "candidate_status",
    "reason",
}


def load_recommendations() -> pd.DataFrame:
    """Load and validate the latest recommendation output."""

    if not RECOMMENDATION_FILE.exists():
        raise FileNotFoundError(
            f"Recommendation file not found: {RECOMMENDATION_FILE}"
        )

    df = pd.read_parquet(RECOMMENDATION_FILE)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Recommendation file is missing columns: {sorted(missing)}"
        )

    return df


def get_recommendations(
    risk_profile: str | None = None,
    asset_class: str | None = None,
    horizon: str | None = None,
    skip: int = 0,
    limit: int = 20,
    sort_by: str = "rank",
    include_ineligible: bool = False,
) -> dict:

    df = load_recommendations()

    # By default, return only eligible recommendation candidates.
    if not include_ineligible:
        df = df[
            df["candidate_status"] == "RECOMMENDATION_CANDIDATE"
        ]

    # Apply filters.
    if risk_profile:
        df = df[df["risk_profile"] == risk_profile]

    if asset_class:
        df = df[df["asset_class"] == asset_class]

    if horizon:
        df = df[df["horizon"] == horizon]

    total = len(df)

    # Sort results.
    if sort_by == "rank":
        df = df.sort_values(
            "rank_within_group",
            ascending=True,
            na_position="last",
        )

    elif sort_by == "score":
        df = df.sort_values(
            "final_score",
            ascending=False,
            na_position="last",
        )

    # Pagination.
    df = df.iloc[skip:skip + limit]

    # Convert NaN values to None for valid JSON responses.
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "recommendations": df.to_dict(orient="records"),
    }