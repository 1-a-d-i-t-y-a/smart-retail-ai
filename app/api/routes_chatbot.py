from fastapi import APIRouter, HTTPException

from app.core import increment
from app.schemas.api import ChatbotRequest, ChatbotResponse
from app.services.chatbot_service import reply

router = APIRouter(tags=["Retail Chatbot"])


@router.post("/chatbot", response_model=ChatbotResponse)
def chatbot(request: ChatbotRequest):
    try:
        result = reply(request.message)
        increment("chatbot_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Retail chatbot service error.") from exc
