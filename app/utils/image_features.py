from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 128


def hog_descriptor(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is empty")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )
    return hog.compute(gray).reshape(-1)
