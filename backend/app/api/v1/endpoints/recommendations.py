
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.recommendation import (
    RecommendationResponse,
    PersonalizedRecommendationRequest,
    PersonalizedRecommendationResponse,
)

from backend.app.services.recommendation_service import (
    get_recommendations,
)

from backend.app.services.personalized_recommendation_service import (
    get_personalized_recommendations,
)


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


# ============================================================
# EXISTING RECOMMENDATION ENDPOINT
# ============================================================

@router.get("", response_model=RecommendationResponse)
def read_recommendations(
    risk_profile: Literal[
        "conservative", "moderate", "aggressive"
    ] | None = None,

    asset_class: Literal[
        "stocks", "etfs", "mutual_funds"
    ] | None = None,

    horizon: Literal["6m", "1y", "2y"] | None = None,

    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),

    sort_by: Literal["rank", "score"] = "rank",

    include_ineligible: bool = False,
):
    try:
        return get_recommendations(
            risk_profile=risk_profile,
            asset_class=asset_class,
            horizon=horizon,
            skip=skip,
            limit=limit,
            sort_by=sort_by,
            include_ineligible=include_ineligible,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Recommendation data is currently unavailable.",
        )

    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Recommendation data failed validation.",
        )


# ============================================================
# NEW PERSONALIZED RECOMMENDATION ENDPOINT
# ============================================================

@router.post(
    "/personalized",
    response_model=PersonalizedRecommendationResponse,
)
def create_personalized_recommendations(
    request: PersonalizedRecommendationRequest,
):

    try:
        return get_personalized_recommendations(
            investment_amount=request.investment_amount,
            risk_profile=request.risk_profile,
            horizon=request.horizon,
            asset_classes=request.asset_classes,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Recommendation data is currently unavailable.",
        )

    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Recommendation data failed validation.",
        )
