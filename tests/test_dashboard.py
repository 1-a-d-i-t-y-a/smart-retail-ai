from __future__ import annotations

from fastapi.testclient import TestClient
from app.core import reset, snapshot


def test_dashboard_stats_structure(client: TestClient):
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "product_requests" in data
    assert "face_requests" in data
    assert "sentiment_requests" in data
    assert "chatbot_requests" in data
    assert isinstance(data["total_requests"], int)


def test_dashboard_counter_increments_on_success(client: TestClient):
    reset()
    init_stats = snapshot()
    assert init_stats["total_requests"] == 0
    assert init_stats["chatbot_requests"] == 0

    res = client.post("/chatbot", json={"message": "hello"})
    assert res.status_code == 200

    new_stats = snapshot()
    assert new_stats["total_requests"] == 1
    assert new_stats["chatbot_requests"] == 1


def test_dashboard_does_not_increment_on_bad_request(client: TestClient):
    reset()
    res = client.post("/chatbot", json={"message": ""})
    assert res.status_code == 422

    stats = snapshot()
    assert stats["total_requests"] == 0
    assert stats["chatbot_requests"] == 0
