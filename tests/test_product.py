from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.services.product_service import load_product_model


def test_classify_product_valid_image(client: TestClient, sample_image_bytes: bytes):
    files = {"file": ("test_product.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "product" in data
    assert "confidence" in data
    assert data["product"] in ["bottle", "mug", "headphones", "shoe"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_classify_product_invalid_content_type(client: TestClient):
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 400
    assert "upload a valid image file" in response.json()["detail"]


def test_classify_product_corrupt_image(client: TestClient, invalid_image_bytes: bytes):
    files = {"file": ("corrupt.jpg", invalid_image_bytes, "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 400
    assert "Could not decode image" in response.json()["detail"] or "header validation failed" in response.json()["detail"]


def test_classify_product_empty_image(client: TestClient):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_classify_product_oversized_upload(client: TestClient, monkeypatch):
    """Uploading a payload that exceeds the size limit must return HTTP 413."""
    monkeypatch.setattr(settings, "max_image_bytes", 50)
    large_payload = b"X" * 200
    files = {"file": ("big.jpg", large_payload, "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_classify_product_undersized_dimensions(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "min_image_dimension", 64)
    tiny = np.ones((16, 16, 3), dtype=np.uint8) * 120
    _, enc = cv2.imencode(".png", tiny)
    files = {"file": ("tiny.png", enc.tobytes(), "image/png")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 400
    assert "smaller than minimum allowed" in response.json()["detail"]


def test_classify_product_missing_model(client: TestClient, sample_image_bytes: bytes, monkeypatch):
    monkeypatch.setattr(settings, "product_model_path", Path("nonexistent_product_model.joblib"))
    load_product_model.cache_clear()
    try:
        files = {"file": ("sample.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/classify-product", files=files)
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]
    finally:
        load_product_model.cache_clear()
