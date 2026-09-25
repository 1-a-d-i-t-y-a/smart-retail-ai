from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "models" / "face_db.pkl"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _detect_largest_face(gray: np.ndarray):
    detector = cv2.CascadeClassifier(CASCADE)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


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


def build_face_db(reference_image: Path, threshold: float = 0.52) -> None:
    image = cv2.imread(str(reference_image))
    if image is None:
        raise ValueError(f"Cannot read reference image: {reference_image}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    box = _detect_largest_face(gray)
    if box is None:
        raise ValueError("No face detected in reference image")
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    db = {
        "person_01": _embedding(face),
        "threshold": float(threshold),
        "reference_bbox": [int(x), int(y), int(w), int(h)],
    }
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


@lru_cache(maxsize=1)
def _load_face_db() -> dict:
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def recognize_face(image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = _detect_largest_face(gray)
    if box is None:
        return {
            "matched": False,
            "person": None,
            "distance": None,
            "similarity": None,
            "bbox": None,
            "message": "No face detected",
        }

    x, y, w, h = box
    query = _normalize_face(gray[y:y + h, x:x + w])
    db = _load_face_db()

    scores = {}
    for name, record in db.items():
        if name in {"threshold", "reference_bbox"}:
            continue
        ref = np.clip(record["image"] * 255.0, 0, 255).astype(np.uint8)
        combined, _, _ = _similarity(query, ref)
        scores[name] = combined

    person, similarity = max(scores.items(), key=lambda item: item[1])
    threshold = float(db.get("threshold", 0.52))
    matched = similarity >= threshold
    distance = 1.0 - similarity
    return {
        "matched": bool(matched),
        "person": person if matched else None,
        "distance": float(distance),
        "similarity": float(similarity),
        "bbox": [int(x), int(y), int(w), int(h)],
        "message": "Face matched" if matched else "Face detected but no registered match",
    }
