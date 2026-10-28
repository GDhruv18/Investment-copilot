from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
