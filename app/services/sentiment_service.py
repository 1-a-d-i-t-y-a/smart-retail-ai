from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_sentiment_model():
    model_path = settings.sentiment_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Sentiment classification model not found at {model_path}. "
            "Please train or build model artifacts using scripts/build_all_models.py."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:
        logger.exception("Failed to load sentiment model from %s: %s", model_path, exc)
        raise RuntimeError(f"Corrupt or incompatible sentiment model artifact: {exc}") from exc


def predict_sentiment(text: str) -> dict:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text must not be empty or blank.")

    model = load_sentiment_model()
    probs = model.predict_proba([cleaned])[0]
    idx = int(np.argmax(probs))
    label = str(model.classes_[idx])
    return {
        "sentiment": label,
        "confidence": float(probs[idx]),
    }
