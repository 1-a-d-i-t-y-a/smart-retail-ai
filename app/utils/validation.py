from __future__ import annotations

import cv2
import numpy as np

from app.config import settings

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


def validate_image_payload(content_type: str | None, payload: bytes) -> np.ndarray:
    """Validate an uploaded image payload before passing it to ML inference pipelines.

    Checks:
    1. Content-Type header matches an image MIME type.
    2. Non-empty payload.
    3. Payload size does not exceed `settings.max_image_bytes`.
    4. Image successfully decodes as a valid 2D/3D image matrix.
    5. Dimensions conform to configured minimum and maximum bounds.
    """
    if not content_type or (not content_type.startswith("image/") and content_type not in ALLOWED_MIME_TYPES):
        raise ValueError("Please upload a valid image file (JPEG, PNG, WebP).")

    if not payload or len(payload) == 0:
        raise ValueError("The uploaded image is empty.")

    if len(payload) > settings.max_image_bytes:
        max_mb = settings.max_image_bytes / (1024 * 1024)
        raise ValueError(f"Image exceeds the {max_mb:.1f} MB upload limit.")

    # Decode image buffer safely
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
