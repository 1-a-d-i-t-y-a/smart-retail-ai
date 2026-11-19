from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.config import settings
from app.services.face_service import load_face_db

ROOT = Path(__file__).resolve().parents[1]


def test_face_recognition_invalid_content_type(client: TestClient):
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 400


def test_face_recognition_corrupt_image(client: TestClient, invalid_image_bytes: bytes):
    files = {"file": ("corrupt.jpg", invalid_image_bytes, "image/jpeg")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 400


def test_face_recognition_registered_match(client: TestClient):
    ref_img_path = ROOT / "data" / "training" / "face" / "person_01_reference.png"
    if not ref_img_path.exists():
        ref_img_path = ROOT / "artifacts" / "face" / "reference_annotated.png"

    with open(ref_img_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("registered.png", img_bytes, "image/png")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is True
    assert data["person"] == "person_01"
    assert data["similarity"] >= 0.50
    assert data["bbox"] is not None


def test_face_recognition_unregistered_negative(client: TestClient):
    neg_path = ROOT / "data" / "evaluation" / "face" / "negative_other_face.png"
    if not neg_path.exists():
        neg_path = ROOT / "artifacts" / "face" / "negative_other_face_annotated.png"

    with open(neg_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("unregistered.png", img_bytes, "image/png")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is False
    assert data["person"] is None


def test_face_recognition_no_face_detected(client: TestClient):
    no_face_path = ROOT / "data" / "evaluation" / "face" / "no_face_distractor.png"
    if no_face_path.exists():
        with open(no_face_path, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 200  # fallback

    files = {"file": ("distractor.png", img_bytes, "image/png")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is False
    assert "No face detected" in data["message"]


def test_face_recognition_oversized_upload(client: TestClient, monkeypatch):
    """Uploading a payload that exceeds the size limit must return HTTP 413."""
    monkeypatch.setattr(settings, "max_image_bytes", 50)
    large_payload = b"X" * 200
    files = {"file": ("big.png", large_payload, "image/png")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_face_recognition_missing_database(client: TestClient, sample_image_bytes: bytes, monkeypatch):
    """When the face DB is missing, any image that hits the DB lookup should return 503."""
    monkeypatch.setattr(settings, "face_db_path", Path("nonexistent_face_db.pkl"))
    load_face_db.cache_clear()
    try:
        # Use the registered reference image — it triggers the DB lookup which will fail.
        ref_img_path = ROOT / "data" / "training" / "face" / "person_01_reference.png"
        if not ref_img_path.exists():
            ref_img_path = ROOT / "artifacts" / "face" / "reference_annotated.png"
        with open(ref_img_path, "rb") as f:
            img_bytes = f.read()
        files = {"file": ("face.png", img_bytes, "image/png")}
        response = client.post("/recognize-face", files=files)
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]
    finally:
        load_face_db.cache_clear()
