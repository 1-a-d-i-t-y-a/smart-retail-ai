from __future__ import annotations

import asyncio
import io
import warnings
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.utils.validation import (
    ALLOWED_MIME_TYPES,
    MIME_TO_PILLOW_FORMATS,
    PayloadTooLargeError,
    read_bounded_upload,
    validate_image_payload,
)


def _create_image_bytes(format_name: str, size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def test_validate_image_payload_success(sample_image_bytes: bytes):
    img = validate_image_payload("image/jpeg", sample_image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape[0] >= settings.min_image_dimension
    assert img.shape[1] >= settings.min_image_dimension


def test_validate_image_payload_normalized_mime(sample_image_bytes: bytes):
    # Normalized MIME with whitespace and charset parameter
    img = validate_image_payload("  IMAGE/JPEG; charset=utf-8  ", sample_image_bytes)
    assert isinstance(img, np.ndarray)


def test_validate_image_payload_png_success():
    png_bytes = _create_image_bytes("PNG", (32, 32))
    img = validate_image_payload("image/png", png_bytes)
    assert isinstance(img, np.ndarray)


def test_validate_image_payload_webp_success():
    webp_bytes = _create_image_bytes("WEBP", (32, 32))
    img = validate_image_payload("image/webp", webp_bytes)
    assert isinstance(img, np.ndarray)


def test_validate_image_payload_bmp_success():
    bmp_bytes = _create_image_bytes("BMP", (32, 32))
    img = validate_image_payload("image/bmp", bmp_bytes)
    assert isinstance(img, np.ndarray)


def test_validate_image_payload_missing_mime():
    with pytest.raises(ValueError, match="upload a valid image file"):
        validate_image_payload(None, b"some content")


def test_validate_image_payload_invalid_mime():
    with pytest.raises(ValueError, match="upload a valid image file"):
        validate_image_payload("text/plain", b"some content")


def test_validate_image_payload_svg_rejected():
    # Explicitly test that image/svg+xml is rejected by the strict allowlist
    with pytest.raises(ValueError, match="upload a valid image file"):
        validate_image_payload("image/svg+xml", b"<svg></svg>")


def test_validate_image_payload_mime_format_mismatch():
    # PNG bytes passed with image/jpeg MIME header
    png_bytes = _create_image_bytes("PNG", (32, 32))
    with pytest.raises(ValueError, match="Image format mismatch"):
        validate_image_payload("image/jpeg", png_bytes)


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
    tiny_bytes = _create_image_bytes("PNG", (20, 20))
    with pytest.raises(ValueError, match="smaller than minimum allowed"):
        validate_image_payload("image/png", tiny_bytes)


def test_validate_image_payload_oversized_dimension(monkeypatch):
    monkeypatch.setattr(settings, "max_image_dimension", 100)
    big_bytes = _create_image_bytes("PNG", (120, 120))
    with pytest.raises(ValueError, match="exceed maximum allowed"):
        validate_image_payload("image/png", big_bytes)


def test_validate_image_payload_decompression_bomb(monkeypatch):
    # Simulate a decompression bomb by lowering Pillow's MAX_IMAGE_PIXELS
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 500)
    img_bytes = _create_image_bytes("PNG", (40, 40))  # 1600 pixels > 500
    with pytest.raises(ValueError, match="decompression bomb"):
        validate_image_payload("image/png", img_bytes)


# --- read_bounded_upload tests ---

def _make_upload_file(data: bytes, chunk_size: int = 65536):
    """Create a mock UploadFile that yields chunks and counts read calls."""
    read_calls = []

    async def _read(size=-1):
        read_calls.append(size)
        if not data:
            return b""
        start = sum(read_calls[:-1])
        if start >= len(data):
            return b""
        end = start + size if size > 0 else len(data)
        return data[start:end]

    mock = MagicMock()
    mock.read = _read
    mock.read_calls = read_calls
    return mock


def test_read_bounded_upload_exact_limit():
    # Exactly at max_bytes
    data = b"A" * 1024
    mock_file = _make_upload_file(data)
    result = asyncio.get_event_loop().run_until_complete(
        read_bounded_upload(mock_file, max_bytes=1024)
    )
    assert result == data


def test_read_bounded_upload_one_byte_over():
    # Exactly max_bytes + 1
    data = b"B" * 1025
    mock_file = _make_upload_file(data)
    with pytest.raises(PayloadTooLargeError, match="limit"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=1024)
        )


def test_read_bounded_upload_stops_reading_immediately():
    # Huge payload (1 MB), max_bytes = 100
    # The reader must stop as soon as it reads 101 bytes, never reading the rest
    data = b"C" * 1_000_000
    mock_file = _make_upload_file(data, chunk_size=65536)
    with pytest.raises(PayloadTooLargeError, match="limit"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=100)
        )
    # Total bytes requested must not exceed max_bytes + 1 = 101 bytes
    total_requested = sum(mock_file.read_calls)
    assert total_requested <= 101


def test_read_bounded_upload_empty_file():
    mock_file = _make_upload_file(b"")
    with pytest.raises(ValueError, match="empty"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=1024)
        )


def test_read_bounded_upload_invalid_max_bytes():
    mock_file = _make_upload_file(b"content")
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=0)
        )
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.get_event_loop().run_until_complete(
            read_bounded_upload(mock_file, max_bytes=-5)
        )
