#!/usr/bin/env python3
"""
scripts/train_baseline_models.py
--------------------------------
Reproducible script to build baseline machine learning model artifacts
for the Smart Retail AI platform:
1. Sentiment Analysis: TF-IDF + Logistic Regression
2. Product Image Classification: HOG Features + RBF Support Vector Machine
3. Face Verification: Face Database Pickle with reference embeddings
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Project root
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def train_sentiment_model() -> Path:
    """Train and persist baseline sentiment model."""
    print("Training Sentiment Analysis baseline model...")
    texts = [
        # Positive
        "Excellent product, highly recommended!",
        "Great quality and fast delivery. Very satisfied.",
        "Loved the customer support, five stars!",
        "Amazing experience, the item works perfectly.",
        "Very good quality fabric and fits well.",
        "Best purchase ever, totally worth the price.",
        "Super happy with the order, will buy again.",
        "Fantastic service and friendly store staff.",
        # Neutral
        "Product arrived on time, standard packaging.",
        "It is okay, nothing special but does the job.",
        "Average quality, neither good nor bad.",
        "The item matches the description.",
        "Delivery took four days, as expected.",
        "Standard retail experience.",
        "Neutral opinion, works as stated.",
        # Negative
        "Terrible quality, arrived damaged and broken.",
        "Worst customer service, nobody answers the phone.",
        "Defective item, do not buy this!",
        "Completely dissatisfied with this purchase.",
        "Very late delivery and poor packaging.",
        "Horrible experience, requesting an immediate refund.",
        "Fell apart after one day of use.",
        "Waste of money, totally unusable.",
    ]
    labels = (
        ["positive"] * 8
        + ["neutral"] * 7
        + ["negative"] * 8
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(C=1.0, max_iter=200, random_state=42)),
    ])

    pipeline.fit(texts, labels)

    out_path = MODELS_DIR / "sentiment_classifier.joblib"
    joblib.dump(pipeline, out_path)
    print(f" Saved sentiment model to {out_path}")
    return out_path


def train_product_model() -> Path:
    """Train and persist baseline HOG+SVM product classifier."""
    print("Training Product Classification baseline model...")
    # Import HOG descriptor from app
    from app.utils.image_features import hog_descriptor

    classes = ["bottle", "mug", "headphones", "shoe"]
    X, y = [], []

    # Generate synthetic images with characteristic geometric shapes
    np.random.seed(42)
    for class_idx, class_name in enumerate(classes):
        for sample_i in range(25):
            img = np.ones((128, 128, 3), dtype=np.uint8) * 240
            noise = np.random.randint(-15, 15, (128, 128, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            if class_name == "bottle":
                # Vertical elongated cylinder
                cv2.rectangle(img, (48, 40), (80, 110), (50, 120, 200), -1)
                cv2.rectangle(img, (56, 20), (72, 40), (40, 100, 180), -1)
            elif class_name == "mug":
                # Square body + handle
                cv2.rectangle(img, (40, 45), (88, 105), (180, 80, 60), -1)
                cv2.circle(img, (95, 75), 18, (180, 80, 60), 6)
            elif class_name == "headphones":
                # Headband arc + earcups
                cv2.ellipse(img, (64, 55), (40, 30), 0, 180, 360, (40, 40, 40), 6)
                cv2.circle(img, (26, 75), 14, (30, 30, 30), -1)
                cv2.circle(img, (102, 75), 14, (30, 30, 30), -1)
            elif class_name == "shoe":
                # Horizontal sole + upper
                pts = np.array([[20, 95], [108, 95], [105, 80], [70, 70], [45, 60], [25, 75]], np.int32)
                cv2.fillPoly(img, [pts], (60, 160, 60))

            feat = hog_descriptor(img)
            X.append(feat)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    svm = SVC(kernel="rbf", C=5.0, probability=True, random_state=42)
    svm.fit(X, y)

    bundle = {
        "pipeline": svm,
        "classes": classes,
    }

    out_path = MODELS_DIR / "product_classifier.joblib"
    joblib.dump(bundle, out_path)
    print(f" Saved product model to {out_path}")
    return out_path


def build_demo_face_db() -> Path:
    """Build and persist baseline face verification database."""
    print("Building Face Verification baseline database...")
    # 160x160 normalized face synthetic reference
    np.random.seed(101)
    face_canvas = np.ones((160, 160), dtype=np.uint8) * 180
    # Simulate face oval, eyes, nose, mouth
    cv2.ellipse(face_canvas, (80, 80), (55, 70), 0, 0, 360, 120, -1)
    cv2.circle(face_canvas, (60, 65), 8, 40, -1)
    cv2.circle(face_canvas, (100, 65), 8, 40, -1)
    cv2.line(face_canvas, (80, 75), (80, 95), 60, 3)
    cv2.ellipse(face_canvas, (80, 115), (20, 8), 0, 0, 180, 50, -1)
    face_canvas = cv2.equalizeHist(face_canvas)

    normalized_face = face_canvas.astype(np.float32) / 255.0

    db = {
        "person_01": {"image": normalized_face},
        "threshold": 0.50,
        "reference_bbox": [25, 25, 110, 110],
    }

    out_path = MODELS_DIR / "face_db.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(db, f)
    print(f" Saved face database to {out_path}")
    return out_path


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_sentiment_model()
    train_product_model()
    build_demo_face_db()
    print("All baseline model artifacts successfully generated!")


if __name__ == "__main__":
    main()
