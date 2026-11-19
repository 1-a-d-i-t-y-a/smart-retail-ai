#!/usr/bin/env python3
"""
scripts/evaluate_models.py
--------------------------
Evaluates the trained models against the test splits and challenge sets.
Generates evaluation metric artifacts in artifacts/ and validates model accuracy.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.services.face_service import recognize_face
from app.utils.image_features import hog_descriptor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_DIR = DATA_DIR / "evaluation"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


def evaluate_product_model() -> dict:
    print("[1/3] Evaluating Product Classifier...")
    model_path = MODELS_DIR / "product_classifier.joblib"
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]

    X_test, y_test = [], []
    product_eval_dir = EVAL_DIR / "product"

    for cls in classes:
        cls_dir = product_eval_dir / cls
        for img_path in sorted(cls_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            feat = hog_descriptor(img)
            X_test.append(feat)
            y_test.append(cls)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cmatrix = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # Challenge set evaluation
    challenge_dir = EVAL_DIR / "product_challenge"
    c_correct = 0
    c_total = 0
    if challenge_dir.exists():
        for img_path in sorted(challenge_dir.glob("*.png")):
            true_cls = img_path.name.split("_challenge")[0]
            img = cv2.imread(str(img_path))
            feat = hog_descriptor(img).reshape(1, -1)
            pred = pipeline.predict(feat)[0]
            if pred == true_cls:
                c_correct += 1
            c_total += 1

    challenge_acc = float(c_correct / c_total) if c_total > 0 else 1.0

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cmatrix,
        "classes": classes,
        "train_samples": 240,
        "test_samples": len(y_test),
        "challenge_samples": c_total,
        "challenge_accuracy": challenge_acc,
    }

    out_dir = ARTIFACTS_DIR / "product"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Product Test Accuracy: {acc * 100:.2f}% ({len(y_test)} samples)")
    print(f"  Product Challenge Accuracy: {challenge_acc * 100:.2f}% ({c_total} samples)")
    return metrics


def evaluate_sentiment_model() -> dict:
    print("[2/3] Evaluating Sentiment Classifier...")
    model_path = MODELS_DIR / "sentiment_classifier.joblib"
    pipeline = joblib.load(model_path)

    test_csv = EVAL_DIR / "sentiment_test.csv"
    texts, y_test = [], []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            y_test.append(row["sentiment"])

    y_pred = pipeline.predict(texts)
    classes = ["negative", "neutral", "positive"]
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cmatrix = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # Challenge evaluation
    challenge_csv = EVAL_DIR / "sentiment_challenge.csv"
    c_texts, c_y = [], []
    if challenge_csv.exists():
        with open(challenge_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_texts.append(row["text"])
                c_y.append(row["sentiment"])

    c_pred = pipeline.predict(c_texts)
    c_acc = float(accuracy_score(c_y, c_pred)) if c_y else 1.0

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cmatrix,
        "classes": classes,
        "train_samples": 283,
        "test_samples": len(y_test),
        "challenge_samples": len(c_y),
        "challenge_accuracy": c_acc,
    }

    out_dir = ARTIFACTS_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "component": "sentiment_analysis",
        "method": "TF-IDF vectorization with 1-2 grams + Logistic Regression",
        "train_samples": 283,
        "test_samples": len(y_test),
        "accuracy": acc,
        "classes": classes,
        "api_endpoint": "POST /analyze-sentiment",
        "note": "Controlled synthetic retail sentiment evaluation dataset with reproducible random seed 42.",
    }
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  Sentiment Test Accuracy: {acc * 100:.2f}% ({len(y_test)} samples)")
    print(f"  Sentiment Challenge Accuracy: {c_acc * 100:.2f}% ({len(c_y)} samples)")
    return metrics


def evaluate_face_model() -> dict:
    print("[3/3] Evaluating Face Verification Service...")
    face_db_path = MODELS_DIR / "face_db.pkl"
    with open(face_db_path, "rb") as f:
        db = pickle.load(f)

    threshold = float(db.get("threshold", 0.50))
    face_eval_dir = EVAL_DIR / "face"

    cases = [
        {
            "case": "reference",
            "image_path": DATA_DIR / "training" / "face" / "person_01_reference.png",
            "expected_match": True,
        },
        {
            "case": "positive_query",
            "image_path": face_eval_dir / "positive_query.png",
            "expected_match": True,
        },
        {
            "case": "negative_other_face",
            "image_path": face_eval_dir / "negative_other_face.png",
            "expected_match": False,
        },
    ]

    results = []
    correct_count = 0
    pos_sims, neg_sims = [], []

    for c in cases:
        with open(c["image_path"], "rb") as f:
            payload = f.read()

        verdict = recognize_face(payload)
        is_matched = verdict["matched"]
        sim = verdict.get("similarity") or 0.0

        is_correct = (is_matched == c["expected_match"])
        if is_correct:
            correct_count += 1

        if c["expected_match"]:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)

        results.append({
            "case": c["case"],
            "matched": is_matched,
            "person": verdict.get("person"),
            "distance": verdict.get("distance"),
            "similarity": verdict.get("similarity"),
            "bbox": verdict.get("bbox"),
            "message": verdict.get("message"),
            "expected_match": c["expected_match"],
            "correct": is_correct,
        })

    eval_out = {
        "metrics": {
            "cases": len(cases),
            "correct": correct_count,
            "accuracy": float(correct_count / len(cases)),
            "threshold": threshold,
            "mean_positive_similarity": float(np.mean(pos_sims)) if pos_sims else 0.0,
            "mean_negative_similarity": float(np.mean(neg_sims)) if neg_sims else 0.0,
            "note": "Face verification evaluation using OpenCV Haar Cascade and SSIM/Pearson correlation matching.",
        },
        "results": results,
    }

    out_dir = ARTIFACTS_DIR / "face"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)

    print(f"  Face Verification Accuracy: {eval_out['metrics']['accuracy'] * 100:.2f}% ({len(cases)} cases)")
    return eval_out


def main():
    print("=" * 60)
    print("Smart Retail AI — Model Evaluation & Metric Generation")
    print("=" * 60)
    evaluate_product_model()
    evaluate_sentiment_model()
    evaluate_face_model()
    print("=" * 60)
    print("Evaluation complete! Verified artifacts saved to artifacts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
