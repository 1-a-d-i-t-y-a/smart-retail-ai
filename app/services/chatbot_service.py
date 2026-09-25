from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = BASE_DIR / "data" / "chatbot" / "retail_knowledge.json"

with KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

# Phrase-weighted deterministic intent routing keeps the reconstruction local,
# reproducible, and independent of external LLM/API credentials.
INTENT_PATTERNS: dict[str, list[tuple[str, int]]] = {
    "greeting": [
        ("hello", 3), ("hi", 2), ("hey", 2), ("good morning", 3),
        ("good afternoon", 3), ("good evening", 3), ("namaste", 3)
    ],
    "order_status": [
        ("order status", 6), ("track my order", 6), ("track order", 6),
        ("where is my order", 6), ("order tracking", 6), ("order update", 5),
        ("my order", 3), ("order", 2)
    ],
    "shipping": [
        ("delivery", 5), ("shipping", 5), ("deliver", 4), ("arrive", 4),
        ("dispatch", 4), ("how long", 3), ("delivery time", 6)
    ],
    "returns": [
        ("return", 6), ("refund", 6), ("exchange", 6), ("replace", 5),
        ("send back", 6), ("money back", 6)
    ],
    "store_info": [
        ("store hours", 6), ("opening hours", 6), ("open today", 5),
        ("closing time", 6), ("timing", 4), ("location", 5), ("address", 5),
        ("where is the store", 6)
    ],
    "products": [
        ("recommend", 5), ("product", 4), ("catalog", 5), ("headphones", 5),
        ("shoe", 5), ("bottle", 5), ("mug", 5), ("features", 3),
        ("compare", 4), ("buy", 3)
    ],
    "payment": [
        ("payment", 6), ("pay", 4), ("upi", 5), ("card", 4),
        ("cash on delivery", 7), ("cod", 5)
    ],
    "support": [
        ("support", 5), ("customer service", 6), ("contact", 4),
        ("help me", 4), ("complaint", 5), ("issue", 3), ("problem", 3)
    ],
}

RESPONSES = {
    "greeting": "Hello! I can help with products, orders, delivery, returns, store information, payments, and customer support.",
    "order_status": KNOWLEDGE["shipping"]["tracking"],
    "shipping": KNOWLEDGE["shipping"]["delivery"],
    "returns": KNOWLEDGE["returns"]["policy"],
    "store_info": KNOWLEDGE["store"]["hours"] + " " + KNOWLEDGE["store"]["location"],
    "products": KNOWLEDGE["products"]["recommendation"] + " " + KNOWLEDGE["products"]["catalog"],
    "payment": KNOWLEDGE["payment"]["methods"],
    "support": KNOWLEDGE["store"]["contact"],
    "fallback": "I can help with products, order status, delivery, returns, store information, payments, and general customer support. Please rephrase your question with a little more detail.",
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def classify_message(message: str) -> dict:
    """Classify a retail message using weighted phrase matching."""
    text = _normalise(message)
    scores: dict[str, int] = {intent: 0 for intent in INTENT_PATTERNS}
    matches: dict[str, list[str]] = {intent: [] for intent in INTENT_PATTERNS}

    for intent, patterns in INTENT_PATTERNS.items():
        for phrase, weight in patterns:
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                scores[intent] += weight
                matches[intent].append(phrase)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Greeting-led messages with only a generic support phrase stay greetings.
    if scores["greeting"] > 0 and all(
        scores[intent] == 0 for intent in scores if intent not in {"greeting", "support"}
    ):
        intent = "greeting"
        best_score = scores["greeting"]
    elif best_score == 0:
        intent = "fallback"
    else:
        intent = best_intent

    return {
        "intent": intent,
        "score": best_score,
        "matched_terms": matches.get(intent, []),
        "response": RESPONSES[intent],
    }


def reply(message: str) -> dict:
    result = classify_message(message)
    return {"intent": result["intent"], "response": result["response"]}
