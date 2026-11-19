from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.config import settings
from app.services.sentiment_service import load_sentiment_model


def test_sentiment_positive(client: TestClient):
    response = client.post(
        "/analyze-sentiment",
        json={"text": "Excellent quality, extremely happy with my purchase!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0


def test_sentiment_negative(client: TestClient):
    response = client.post(
        "/analyze-sentiment",
        json={"text": "Terrible product, arrived broken and completely damaged."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0


def test_sentiment_neutral(client: TestClient):
    response = client.post(
        "/analyze-sentiment",
        json={"text": "The item is standard, neither good nor bad."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("positive", "neutral", "negative")
    assert 0.0 <= data["confidence"] <= 1.0


def test_sentiment_empty_string_validation(client: TestClient):
    response = client.post("/analyze-sentiment", json={"text": ""})
    assert response.status_code == 422


def test_sentiment_whitespace_validation(client: TestClient):
    response = client.post("/analyze-sentiment", json={"text": "    \t \n   "})
    assert response.status_code == 422


def test_sentiment_oversized_string_validation(client: TestClient):
    huge_text = "great product! " * 400  # > 5000 chars
    response = client.post("/analyze-sentiment", json={"text": huge_text})
    assert response.status_code == 422


def test_sentiment_missing_model(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "sentiment_model_path", Path("nonexistent_sentiment_model.joblib"))
    load_sentiment_model.cache_clear()
    try:
        response = client.post("/analyze-sentiment", json={"text": "Solid experience overall."})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]
    finally:
        load_sentiment_model.cache_clear()
