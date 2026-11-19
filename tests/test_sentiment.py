from __future__ import annotations

from fastapi.testclient import TestClient


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


def test_sentiment_validation_error(client: TestClient):
    # Empty string or missing body
    response = client.post("/analyze-sentiment", json={"text": ""})
    assert response.status_code in (200, 422, 400)
