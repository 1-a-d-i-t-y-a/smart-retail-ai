from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config import settings

logger = logging.getLogger(__name__)


def _get_cascade_detector() -> cv2.CascadeClassifier | None:
    # 1. Configured / local cascade path
    if settings.cascade_path.exists():
        detector = cv2.CascadeClassifier(str(settings.cascade_path))
        if not detector.empty():
            return detector

    # 2. System OpenCV cascade path
    system_cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if system_cascade.exists():
        detector = cv2.CascadeClassifier(str(system_cascade))
        if not detector.empty():
            return detector

    logger.warning("No valid Haar Cascade XML found for face detection.")
    return None


def _detect_largest_face(gray: np.ndarray):
    detector = _get_cascade_detector()
    if detector is None or detector.empty():
        return None
    try:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])
    except Exception as exc:
        logger.warning("Error running detectMultiScale: %s", exc)
        return None


def _normalize_face(gray_face: np.ndarray) -> np.ndarray:
    gray_face = cv2.resize(gray_face, (160, 160), interpolation=cv2.INTER_AREA)
    gray_face = cv2.equalizeHist(gray_face)
    return gray_face


def _embedding(gray_face: np.ndarray) -> dict[str, np.ndarray]:
    face = _normalize_face(gray_face)
    return {"image": face.astype(np.float32) / 255.0}


def _similarity(query: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    q = query.astype(np.uint8)
    r = reference.astype(np.uint8)
    ssim_score = float(ssim(q, r, data_range=255))
    corr = float(np.corrcoef(q.flatten(), r.flatten())[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    corr_score = (corr + 1.0) / 2.0
    combined = 0.70 * ssim_score + 0.30 * corr_score
    return combined, ssim_score, corr_score


@lru_cache(maxsize=1)
def load_face_db() -> dict:
    db_path = settings.face_db_path
    if not db_path.exists():
        raise FileNotFoundError(
            f"Face database file {db_path} not found. "
            "Please initialize the face database using scripts/build_all_models.py."
        )
    try:
        with open(db_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.exception("Failed to load face database from %s: %s", db_path, exc)
        raise RuntimeError(f"Corrupt face database file: {exc}") from exc


def recognize_face(image_bytes: bytes) -> dict:
    """Verify query face image against registered customer database.

    Educational/Reconstruction Scope:
    This module uses OpenCV Haar Cascade face detection and dual Structural
    Similarity (SSIM) + Pearson correlation matching. It is intended for
    demonstration and retail customer check-in simulation, not biometric security.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image buffer.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = _detect_largest_face(gray)
    if box is None:
        return {
            "matched": False,
            "person": None,
            "distance": None,
            "similarity": None,
            "bbox": None,
            "message": "No face detected in query image.",
        }

    x, y, w, h = box
    query = _normalize_face(gray[y:y + h, x:x + w])
    db = load_face_db()

    scores = {}
    for name, record in db.items():
        if name in {"threshold", "reference_bbox"}:
            continue
        ref = np.clip(record["image"] * 255.0, 0, 255).astype(np.uint8)
        combined, _, _ = _similarity(query, ref)
        scores[name] = combined

    if not scores:
        return {
            "matched": False,
            "person": None,
            "distance": None,
            "similarity": None,
            "bbox": [int(x), int(y), int(w), int(h)],
            "message": "Face database contains no registered individuals.",
        }

    person, similarity = max(scores.items(), key=lambda item: item[1])
    threshold = float(db.get("threshold", 0.50))
    matched = similarity >= threshold
    distance = max(0.0, 1.0 - similarity)

    return {
        "matched": bool(matched),
        "person": person if matched else None,
        "distance": float(round(distance, 4)),
        "similarity": float(round(similarity, 4)),
        "bbox": [int(x), int(y), int(w), int(h)],
        "message": "Face matched" if matched else "Face detected but no registered match",
    }
