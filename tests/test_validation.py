from __future__ import annotations

import asyncio
import io
import cv2
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.config import settings
from app.utils.validation import PayloadTooLargeError, read_bounded_upload, validate_image_payload


def test_validate_image_payload_success(sample_image_bytes: bytes):
    img = validate_image_payload("image/jpeg", sample_image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape[0] >= settings.min_image_dimension
    assert img.shape[1] >= settings.min_image_dimension


def test_validate_image_payload_missing_mime():
    with pytest.raises(ValueError, match="upload a valid image file"):
        validate_image_payload(None, b"some content")


def test_validate_image_payload_invalid_mime():
    with pytest.raises(ValueError, match="upload a valid image file"):
        validate_image_payload("text/plain", b"some content")


def test_validate_image_payload_empty_bytes():
    with pytest.raises(ValueError, match="uploaded image is empty"):
        validate_image_payload("image/jpeg", b"")


def test_validate_image_payload_oversized(monkeypatch):
    monkeypatch.setattr(settings, "max_image_bytes", 100)
    with pytest.raises(ValueError, match="upload limit"):
        validate_image_payload("image/jpeg", b"A" * 200)


def test_validate_image_payload_corrupt_bytes():
    with pytest.raises(ValueError, match="Could not decode image|header validation failed"):
        validate_image_payload("image/jpeg", b"NOT_A_VALID_JPEG_HEADER_BYTES")


def test_validate_image_payload_undersized_dimension(monkeypatch):
    monkeypatch.setattr(settings, "min_image_dimension", 50)
    # Generate 20x20 image
    tiny = np.ones((20, 20, 3), dtype=np.uint8) * 128
    _, enc = cv2.imencode(".png", tiny)
    with pytest.raises(ValueError, match="smaller than minimum allowed"):
        validate_image_payload("image/png", enc.tobytes())


def test_validate_image_payload_oversized_dimension(monkeypatch):
    monkeypatch.setattr(settings, "max_image_dimension", 100)
    # Generate 120x120 image
    big = np.ones((120, 120, 3), dtype=np.uint8) * 128
    _, enc = cv2.imencode(".png", big)
    with pytest.raises(ValueError, match="exceed maximum allowed"):
        validate_image_payload("image/png", enc.tobytes())


# --- read_bounded_upload tests ---

def _make_upload_file(data: bytes, chunk_size: int = 65536):
    """Create a mock UploadFile that yields chunks."""
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    chunks.append(b"")  # EOF sentinel

    async def _read(size=-1):
        if chunks:
            return chunks.pop(0)
        return b""

    mock = MagicMock()
    mock.read = _read
    return mock


def test_read_bounded_upload_within_limit():
    data = b"A" * 1000
    mock_file = _make_upload_file(data)
    result = asyncio.get_event_loop().run_until_complete(
        read_bounded_upload(mock_file, max_bytes=2000)
    )
    assert result == data


def test_read_bounded_upload_exceeds_limit():
    data = b"B" * 200_000
    mock_file = _make_upload_file(data)
    with pytest.raises(PayloadTooLargeError, match="limit"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=100)
        )


def test_read_bounded_upload_empty_file():
    mock_file = _make_upload_file(b"")
    with pytest.raises(ValueError, match="empty"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=1024)
        )
