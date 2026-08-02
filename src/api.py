"""Unified FastAPI backend that serves all four retail-AI models.

Endpoints
---------
GET  /health              — liveness probe
POST /classify-product    — multipart image → predicted category
POST /analyze-sentiment   — JSON {text} → sentiment label + confidence
POST /chatbot             — JSON {query} → FAQ reply
POST /identify-customer   — multipart image → recognised customer id
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chatbot import RetailChatbot
from .face_utils import FaceRecognizer
from .sentiment_model import SentimentModel

app = FastAPI(
    title="AI Smart Retail & Customer Intelligence Platform",
    description="Unified inference API for CV, DL and NLP retail modules.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- lazy singletons ----------
_MODEL_DIR = os.getenv("MODEL_DIR", "models")
_chatbot: Optional[RetailChatbot] = None
_sentiment: Optional[SentimentModel] = None
_face: Optional[FaceRecognizer] = None
_product_model = None
_product_labels: Optional[dict] = None


def get_chatbot() -> RetailChatbot:
    global _chatbot
    if _chatbot is None:
        path = os.path.join(_MODEL_DIR, "chatbot.joblib")
        _chatbot = RetailChatbot.load(path) if os.path.exists(path) else RetailChatbot()
    return _chatbot


def get_sentiment() -> SentimentModel:
    global _sentiment
    if _sentiment is None:
        path = os.path.join(_MODEL_DIR, "sentiment_model.joblib")
        if not os.path.exists(path):
            raise HTTPException(503, "Sentiment model not trained. Run notebook 4 first.")
        _sentiment = SentimentModel().load(path)
    return _sentiment


def get_face_recognizer() -> FaceRecognizer:
    global _face
    if _face is None:
        _face = FaceRecognizer()
        db_path = os.path.join(_MODEL_DIR, "face_db.pkl")
        if os.path.exists(db_path):
            _face.load(db_path)
    return _face


def get_product_model():
    global _product_model, _product_labels
    if _product_model is None:
        import tensorflow as tf

        model_path = os.path.join(_MODEL_DIR, "product_classifier_model.h5")
        labels_path = os.path.join(_MODEL_DIR, "product_labels.json")
        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            raise HTTPException(503, "Product model not trained. Run notebook 3 first.")
        _product_model = tf.keras.models.load_model(model_path)
        with open(labels_path) as f:
            _product_labels = {int(k): v for k, v in json.load(f).items()}
    return _product_model, _product_labels


# ---------- schemas ----------
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


# ---------- helpers ----------
async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode uploaded image.")
    return img


# ---------- routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "modules": {
            "chatbot": True,
            "sentiment": os.path.exists(os.path.join(_MODEL_DIR, "sentiment_model.joblib")),
            "product_classifier": os.path.exists(
                os.path.join(_MODEL_DIR, "product_classifier_model.h5")
            ),
            "face_db": os.path.exists(os.path.join(_MODEL_DIR, "face_db.pkl")),
        },
    }


@app.post("/chatbot")
def chatbot(req: ChatRequest):
    reply, score = get_chatbot().reply(req.query)
    return {"reply": reply, "confidence": round(score, 4)}


@app.post("/analyze-sentiment")
def analyze_sentiment(req: SentimentRequest):
    label, proba = get_sentiment().predict(req.text)
    return {"sentiment": label, "confidence": round(proba, 4)}


@app.post("/classify-product")
async def classify_product(file: UploadFile = File(...)):
    img = await _read_image(file)
    model, labels = get_product_model()
    img_rs = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    probs = model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"category": labels[idx], "confidence": round(float(probs[idx]), 4)}


@app.post("/identify-customer")
async def identify_customer(file: UploadFile = File(...)):
    img = await _read_image(file)
    fr = get_face_recognizer()
    if not fr.db:
        raise HTTPException(503, "Face DB is empty. Enrol customers via notebook 2 first.")
    cid, score = fr.identify(img)
    return {
        "customer_id": cid,
        "recognized": cid is not None,
        "similarity": round(score, 4),
    }
