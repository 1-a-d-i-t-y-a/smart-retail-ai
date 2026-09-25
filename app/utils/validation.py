from __future__ import annotations

MAX_IMAGE_BYTES = 5 * 1024 * 1024


def validate_image_payload(content_type: str | None, payload: bytes) -> None:
    """Validate an uploaded image before passing it to an ML service."""
    if not content_type or not content_type.startswith("image/"):
        raise ValueError("Please upload an image file.")
    if not payload:
        raise ValueError("The uploaded image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 5 MB upload limit.")
