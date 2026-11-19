from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import joblib
import numpy as np

from app.config import settings
from app.utils.image_features import hog_descriptor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_product_model():
    model_path = settings.product_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Product classification model not found at {model_path}. "
            "Please train or build model artifacts using scripts/build_all_models.py."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:
        logger.exception("Failed to load product model from %s: %s", model_path, exc)
        raise RuntimeError(f"Corrupt or incompatible product model artifact: {exc}") from exc


def predict_product(image_bytes: bytes) -> dict:
    bundle = load_product_model()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image buffer.")

    feat = hog_descriptor(img).reshape(1, -1)
    probs = bundle["pipeline"].predict_proba(feat)[0]
    idx = int(np.argmax(probs))
    return {
        "product": bundle["classes"][idx],
        "confidence": float(probs[idx]),
    }
