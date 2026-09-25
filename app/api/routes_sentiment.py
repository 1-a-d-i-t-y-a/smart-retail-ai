from fastapi import APIRouter, HTTPException

from app.core import increment
from app.schemas.api import SentimentRequest, SentimentResponse
from app.services.sentiment_service import predict_sentiment

router = APIRouter(tags=["Sentiment Analysis"])


@router.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        result = predict_sentiment(request.text)
        increment("sentiment_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Sentiment analysis service error.") from exc
