from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.config import settings


def test_root_redirect(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "service" in data


def test_readiness_endpoint_all_models_present(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["product_model"] is True
    assert data["checks"]["sentiment_model"] is True
    assert data["checks"]["face_database"] is True


def test_readiness_endpoint_model_missing(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "product_model_path", Path("missing_model.joblib"))
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["product_model"] is False
