from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank.")
        return value


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


class ChatbotRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ChatbotResponse(BaseModel):
    intent: str
    response: str


class ProductResponse(BaseModel):
    product: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class FaceResponse(BaseModel):
    matched: bool
    person: str | None = None
    distance: float | None = None
    similarity: float | None = None
    bbox: list[int] | None = None
    message: str


class DashboardStats(BaseModel):
    total_requests: int
    product_requests: int
    face_requests: int
    sentiment_requests: int
    chatbot_requests: int
