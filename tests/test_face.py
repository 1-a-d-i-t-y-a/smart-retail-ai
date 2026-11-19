from __future__ import annotations

from fastapi.testclient import TestClient


def test_face_recognition_invalid_content_type(client: TestClient):
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 400


def test_face_recognition_corrupt_image(client: TestClient, invalid_image_bytes: bytes):
    files = {"file": ("corrupt.jpg", invalid_image_bytes, "image/jpeg")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 400


def test_face_recognition_valid_image(client: TestClient, sample_image_bytes: bytes):
    files = {"file": ("face.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "matched" in data
    assert "message" in data
