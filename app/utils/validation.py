from __future__ import annotations

import io
import warnings

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

MIME_TO_PILLOW_FORMATS: dict[str, set[str]] = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
    "image/bmp": {"BMP"},
}


class PayloadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


async def read_bounded_upload(file, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, reading at most ``max_bytes + 1`` bytes.

    Stops reading immediately as soon as an extra byte proves the limit was
    exceeded, preventing buffering of the remaining upload.

    Args:
        file: A FastAPI ``UploadFile`` or asynchronous file-like object.
        max_bytes: Maximum permitted body size in bytes.

    Returns:
        The full body bytes when it fits within the limit.

    Raises:
        PayloadTooLargeError: When the body exceeds *max_bytes*.
        ValueError: When *max_bytes* <= 0 or the body is empty.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")

    CHUNK_SIZE = 65_536  # 64 KiB
    target_max = max_bytes + 1
    buf = bytearray()

    while len(buf) < target_max:
        to_read = min(CHUNK_SIZE, target_max - len(buf))
        chunk = await file.read(to_read)
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

    Execution sequence:
    1. Validate normalized Content-Type against explicit allowlist.
    2. Validate non-empty payload.
    3. Validate byte-size limit.
    4. Open image using Pillow to inspect header.
    5. Check format compatibility between Content-Type and detected format.
    6. Read width and height from header without full pixel matrix decoding.
    7. Reject dimensions below minimum (10x10) or above maximum (4096x4096).
    8. Trap decompression-bomb warnings and errors as validation failures.
    9. Call Pillow verify() safely.
    10. Decode image via OpenCV.
    11. Confirm OpenCV decoding succeeded.

    Returns:
        Decoded BGR NumPy image array from cv2.imdecode.

    Raises:
        ValueError: When any validation rule fails.
    """
    if not content_type:
        raise ValueError("Please upload a valid image file (JPEG, PNG, WebP, BMP).")

    # Normalize content type: lowercase, strip params after ';'
    raw_ct = content_type.strip().lower()
    normalized_ct = raw_ct.split(";")[0].strip()

    if normalized_ct not in ALLOWED_MIME_TYPES:
        raise ValueError("Please upload a valid image file (JPEG, PNG, WebP, BMP).")

    if not payload or len(payload) == 0:
        raise ValueError("The uploaded image is empty.")

    if len(payload) > settings.max_image_bytes:
        max_mb = settings.max_image_bytes / (1024 * 1024)
        raise ValueError(f"Image exceeds the {max_mb:.1f} MB upload limit.")

    # Pillow header inspection and decompression-bomb protection
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            warnings.simplefilter("error", Image.DecompressionBombError)

            pil_img = Image.open(io.BytesIO(payload))

            for w in caught_warnings:
                if issubclass(w.category, Image.DecompressionBombWarning):
                    raise ValueError(f"Image decompression bomb warning: {w.message}")

            detected_format = (pil_img.format or "").upper()
            allowed_formats = MIME_TO_PILLOW_FORMATS.get(normalized_ct, set())
            if detected_format not in allowed_formats:
                raise ValueError(
                    f"Image format mismatch: Content-Type '{normalized_ct}' "
                    f"does not match detected format '{detected_format}'."
                )

            # Inspect dimensions from header before decoding
            w_dim, h_dim = pil_img.size
            if w_dim < settings.min_image_dimension or h_dim < settings.min_image_dimension:
                raise ValueError(
                    f"Image dimensions ({w_dim}x{h_dim}) are smaller than minimum allowed "
                    f"({settings.min_image_dimension}x{settings.min_image_dimension})."
                )

            if w_dim > settings.max_image_dimension or h_dim > settings.max_image_dimension:
                raise ValueError(
                    f"Image dimensions ({w_dim}x{h_dim}) exceed maximum allowed "
                    f"({settings.max_image_dimension}x{settings.max_image_dimension})."
                )

            pil_img.verify()

    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"Image decompression bomb detected: {exc}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Image header validation failed: {exc}") from exc

    # Decode image using OpenCV only after all header checks succeed
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image: file may be corrupted or in an unsupported format.")

    return img
