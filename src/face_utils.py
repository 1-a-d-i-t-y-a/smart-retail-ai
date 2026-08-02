"""Face detection + lightweight embedding-based customer recognition.

Uses OpenCV's Haar-cascade detector plus a pixel-histogram embedding (fast,
zero external deps beyond OpenCV). Swap `_embed()` with FaceNet / DeepFace for
production-grade accuracy.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_utils import load_image, resize_image, to_grayscale


@dataclass
class FaceRecognizer:
    """In-memory customer face database with disk persistence."""

    cascade_path: str = field(
        default_factory=lambda: cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    embedding_size: Tuple[int, int] = (100, 100)
    db: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Haar cascades require cv2.CascadeClassifier which is unavailable in
        # some preview OpenCV builds. Fall back to a "whole-image" detector so
        # the pipeline still works (and can be swapped for FaceNet later).
        self._detector = None
        if hasattr(cv2, "CascadeClassifier"):
            try:
                det = cv2.CascadeClassifier(self.cascade_path)
                if not det.empty():
                    self._detector = det
            except Exception:
                self._detector = None

    # ---------- detection ----------
    def detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._detector is None:
            h, w = img.shape[:2]
            return [(0, 0, w, h)]  # treat the whole image as one face
        gray = to_grayscale(img)
        faces = self._detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            h, w = img.shape[:2]
            return [(0, 0, w, h)]
        return [tuple(map(int, f)) for f in faces]

    def crop_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect_faces(img)
        if not faces:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        return img[y : y + h, x : x + w]

    # ---------- embedding ----------
    def _embed(self, face_img: np.ndarray) -> np.ndarray:
        face = resize_image(to_grayscale(face_img), self.embedding_size)
        # Histogram of pixel intensities — simple but stable embedding
        hist = cv2.calcHist([face], [0], None, [64], [0, 256]).flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        return hist.astype("float32")

    # ---------- API ----------
    def enroll(self, customer_id: str, img: np.ndarray) -> bool:
        face = self.crop_face(img)
        if face is None:
            return False
        self.db[customer_id] = self._embed(face)
        return True

    def identify(self, img: np.ndarray, threshold: float = 0.75) -> Tuple[Optional[str], float]:
        face = self.crop_face(img)
        if face is None or not self.db:
            return None, 0.0
        emb = self._embed(face)
        best_id, best_score = None, -1.0
        for cid, ref in self.db.items():
            score = float(np.dot(emb, ref))  # cosine on unit vectors
            if score > best_score:
                best_id, best_score = cid, score
        return (best_id if best_score >= threshold else None), best_score

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.db, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.db = pickle.load(f)

    # ---------- convenience ----------
    def enroll_from_folder(self, root: str) -> int:
        """Enrol every sub-folder under `root` as a customer id."""
        count = 0
        for cid in sorted(os.listdir(root)):
            sub = os.path.join(root, cid)
            if not os.path.isdir(sub):
                continue
            for name in os.listdir(sub):
                try:
                    img = load_image(os.path.join(sub, name))
                    if self.enroll(cid, img):
                        count += 1
                        break  # one representative image is enough for the demo
                except Exception:
                    continue
        return count
