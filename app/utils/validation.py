from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.config import settings

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# Sentinel exception for 413 responses (payload too large).
class PayloadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


async def read_bounded_upload(file, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, stopping early if the payload is oversized.

    Reads at most ``max_bytes + 1`` bytes.  If the accumulated length exceeds
    ``max_bytes`` the full body is NOT buffered — a :class:`PayloadTooLargeError`
    is raised immediately so the route can return HTTP 413.

    Args:
        file: A FastAPI ``UploadFile`` instance.
        max_bytes: Maximum permitted body size in bytes.

    Returns:
        The full body bytes when it fits within the limit.

    Raises:
        PayloadTooLargeError: When the body exceeds *max_bytes*.
        ValueError: When the body is empty.
    """
    CHUNK = 65_536  # 64 KiB
    buf = bytearray()
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise PayloadTooLargeError(
                f"Upload exceeds the {max_bytes / (1024 * 1024):.1f} MB limit."
            )
    if len(buf) == 0:
        raise ValueError("The uploaded image is empty.")
    return bytes(buf)


def validate_image_payload(content_type: str | None, payload: bytes) -> np.ndarray:
    """Validate an uploaded image payload before passing it to ML inference pipelines.

    Checks:
    1. Content-Type header matches an image MIME type.
    2. Non-empty payload.
    3. Payload size does not exceed ``settings.max_image_bytes`` (pre-read guard).
    4. Pillow header inspection for decompression-bomb protection.
    5. Image successfully decodes as a valid 2D/3D matrix via OpenCV.
    6. Dimensions conform to configured minimum and maximum bounds.

    Note:
        When routes use :func:`read_bounded_upload`, the size check in step 3 is
        redundant but kept as a defence-in-depth fallback for direct callers.
    """
    if not content_type or (not content_type.startswith("image/") and content_type not in ALLOWED_MIME_TYPES):
        raise ValueError("Please upload a valid image file (JPEG, PNG, WebP).")

    if not payload or len(payload) == 0:
        raise ValueError("The uploaded image is empty.")

    if len(payload) > settings.max_image_bytes:
        max_mb = settings.max_image_bytes / (1024 * 1024)
        raise ValueError(f"Image exceeds the {max_mb:.1f} MB upload limit.")

    # --- Pillow header inspection (decompression-bomb protection) ---
    # PIL.Image.MAX_IMAGE_PIXELS defaults to ~178 MP; Images exceeding it raise
    # a DecompressionBombWarning (or Error at 2×).  We catch both and re-raise
    # as a structured ValueError so routes return 400 rather than 500.
    try:
        pil_img = Image.open(io.BytesIO(payload))
        pil_img.verify()  # Does NOT decode pixel data, reads headers only.
    except Exception as exc:
        raise ValueError(
            f"Image header validation failed: {exc}"
        ) from exc

    # --- OpenCV full decode ---
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image: file may be corrupted or in an unsupported format.")

    h, w = img.shape[:2]
    if w < settings.min_image_dimension or h < settings.min_image_dimension:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are smaller than minimum allowed "
            f"({settings.min_image_dimension}x{settings.min_image_dimension})."
        )

    if w > settings.max_image_dimension or h > settings.max_image_dimension:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed "
            f"({settings.max_image_dimension}x{settings.max_image_dimension})."
        )

    return img
