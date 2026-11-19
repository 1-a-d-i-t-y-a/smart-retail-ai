from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.core import increment
from app.schemas.api import SentimentRequest, SentimentResponse
from app.services.sentiment_service import predict_sentiment

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sentiment Analysis"])


@router.post(
    "/analyze-sentiment",
    response_model=SentimentResponse,
    summary="Analyze retail customer review sentiment",
    description="Classifies customer feedback into positive, neutral, or negative categories with confidence scores.",
    responses={
        200: {"description": "Sentiment analyzed successfully."},
        400: {"description": "Invalid or empty review text."},
        503: {"description": "Sentiment classification model is unavailable."},
    },
)
def analyze_sentiment(request: SentimentRequest):
    try:
        result = predict_sentiment(request.text)
        increment("sentiment_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Sentiment model missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model is unavailable. Please ensure models are built.",
        ) from exc
    except RuntimeError as exc:
        logger.error("Sentiment model runtime failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment classification model runtime error.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in analyze_sentiment: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentiment analysis service error.",
        ) from exc
