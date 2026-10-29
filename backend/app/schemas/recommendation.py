
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    risk_profile: str
    asset_class: str
    ticker: str
    horizon: str

    prediction: str
    prob_attractive: float
    prob_weak: float

    ml_score: float
    risk_score: float
    trend_score: float
    final_score: float

    rank_within_group: int | None = None

    candidate_signal_type: str | None = None
    coverage_status: str | None = None
    candidate_status: str
    reason: str | None = None

    feature_date: datetime | None = None
    model_type: str | None = None
    signal_age_days: int | None = None
    freshness_status: str | None = None
    status: str | None = None


class RecommendationResponse(BaseModel):
    total: int
    skip: int
    limit: int
    recommendations: list[RecommendationItem]


class PersonalizedRecommendationRequest(BaseModel):
    investment_amount: float = Field(gt=0)

    risk_profile: Literal[
        "conservative",
        "moderate",
        "aggressive",
    ]

    horizon: Literal["6m", "1y", "2y"]

    asset_classes: list[
        Literal["stocks", "etfs", "mutual_funds"]
    ] | None = None


class PersonalizedRecommendationItem(BaseModel):
    asset_class: str
    asset_name: str
    ticker: str
    allocation_percentage: float
    allocation_amount: float
    final_score: float
    rank: int
    reason: str | None = None

class PersonalizedRecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    investment_amount: float
    allocated_amount: float
    unallocated_amount: float

    risk_profile: str
    horizon: str

    recommendations: list[PersonalizedRecommendationItem]
