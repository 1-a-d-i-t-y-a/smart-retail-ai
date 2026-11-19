from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a valid test image (128x128 JPEG)."""
    img = np.ones((128, 128, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (40, 40), (88, 100), (50, 120, 220), -1)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def invalid_image_bytes() -> bytes:
    """Generate corrupted non-image bytes."""
    return b"not_a_real_image_payload_binary_data"
