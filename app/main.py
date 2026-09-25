from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes_chatbot import router as chatbot_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_face import router as face_router
from app.api.routes_product import router as product_router
from app.api.routes_sentiment import router as sentiment_router
from app.config import settings
from app.observability import RequestLoggingMiddleware
from app.schemas.api import HealthResponse, ReadinessResponse

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Reconstructed internship project combining computer vision, face recognition, "
        "sentiment analysis, retail chatbot services, and a FastAPI gateway.\n\n"
        "Interactive API documentation is available at /docs and /redoc. "
        "OpenAPI JSON is available at /openapi.json."
    ),
    contact={"name": "Smart Retail AI Project"},
    license_info={"name": "Educational / Internship Reconstruction"},
    openapi_tags=[
        {"name": "System", "description": "Health, readiness, and service metadata."},
        {"name": "Product Classification", "description": "Image-based retail product classification."},
        {"name": "Face Recognition", "description": "Lightweight face verification using the registered demo face database."},
        {"name": "Sentiment Analysis", "description": "Retail customer-text sentiment classification."},
        {"name": "Retail Chatbot", "description": "Intent-based retail customer support responses."},
        {"name": "Dashboard", "description": "Aggregated in-process API request statistics."},
    ],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(product_router)
app.include_router(face_router)
app.include_router(sentiment_router)
app.include_router(chatbot_router)
app.include_router(dashboard_router)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service health check",
    description="Returns basic service health and application version information.",
)
def health():
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Model readiness check",
    description="Checks whether the required product, sentiment, and face model artifacts are available.",
)
def readiness():
    checks = {
        "product_model": settings.product_model_path.exists(),
        "sentiment_model": settings.sentiment_model_path.exists(),
        "face_database": settings.face_db_path.exists(),
    }
    ready = all(checks.values())
    return {"status": "ready" if ready else "not_ready", "checks": checks}
