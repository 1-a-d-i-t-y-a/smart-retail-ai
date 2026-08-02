"""Image preprocessing utilities used across the CV modules."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load a BGR image from disk. Raises FileNotFoundError if missing."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize to `(width, height)` using bilinear interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR / RGB image to single-channel grayscale."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to the [0, 1] float range."""
    return img.astype("float32") / 255.0


def augment_image(img: np.ndarray, flip: bool = True, rotate_deg: float = 0.0) -> np.ndarray:
    """Very small augmentation helper: horizontal flip + optional rotation."""
    out = img.copy()
    if flip:
        out = cv2.flip(out, 1)
    if rotate_deg:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return out
