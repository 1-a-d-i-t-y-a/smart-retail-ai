from __future__ import annotations

from fastapi.testclient import TestClient


def test_dashboard_stats(client: TestClient):
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "product_requests" in data
    assert "face_requests" in data
    assert "sentiment_requests" in data
    assert "chatbot_requests" in data
    assert isinstance(data["total_requests"], int)
