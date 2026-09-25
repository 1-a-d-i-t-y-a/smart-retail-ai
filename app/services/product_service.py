from __future__ import annotations

from pathlib import Path

from functools import lru_cache

import joblib
import numpy as np

from app.utils.image_features import hog_descriptor

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "product_classifier.joblib"


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


def predict_product(image_bytes: bytes) -> dict:
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    bundle = _load_model()
    feat = hog_descriptor(img).reshape(1, -1)
    probs = bundle["pipeline"].predict_proba(feat)[0]
    idx = int(np.argmax(probs))
    return {"product": bundle["classes"][idx], "confidence": float(probs[idx])}
