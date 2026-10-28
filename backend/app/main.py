
from fastapi import FastAPI

from backend.app.core.config import settings
from backend.app.api.v1.endpoints import (
    health,
    recommendations,
    stocks,
    etfs,
)

app = FastAPI(
    title=settings.APP_NAME,
    description="Investment research and recommendation platform.",
    version="1.0.0",
)

app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    recommendations.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    stocks.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    etfs.router,
    prefix=settings.API_V1_PREFIX,
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Investment Copilot API",
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }
