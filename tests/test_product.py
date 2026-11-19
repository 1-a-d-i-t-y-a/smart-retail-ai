from __future__ import annotations

import io
from fastapi.testclient import TestClient


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


def test_classify_product_corrupt_image(client: TestClient, invalid_image_bytes: bytes):
    files = {"file": ("corrupt.jpg", invalid_image_bytes, "image/jpeg")}
    response = client.post("/classify-product", files=files)
    assert response.status_code == 400
