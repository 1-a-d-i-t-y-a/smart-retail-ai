from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.core import increment
from app.schemas.api import ChatbotRequest, ChatbotResponse
from app.services.chatbot_service import reply

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retail Chatbot"])


@router.post(
    "/chatbot",
    response_model=ChatbotResponse,
    summary="Query retail conversational support assistant",
    description="Processes user queries across 9 core retail intents: greeting, order_status, shipping, returns, refunds, store_info, products, payment, support, and fallback.",
    responses={
        200: {"description": "Retail query processed successfully."},
        400: {"description": "Invalid message format or unprocessable query."},
    },
)
def chatbot(request: ChatbotRequest):
    try:
        result = reply(request.message)
        increment("chatbot_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in chatbot route: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retail chatbot service error.",
        ) from exc
