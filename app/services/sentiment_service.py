from __future__ import annotations

from pathlib import Path

from functools import lru_cache

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sentiment_classifier.joblib"


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> dict:
    model = _load_model()
    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    label = str(model.classes_[idx])
    return {"sentiment": label, "confidence": float(probs[idx])}
