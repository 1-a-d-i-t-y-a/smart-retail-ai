from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE = {
    "shipping": {
        "tracking": "Use your order ID with the order-status service to retrieve real-time shipment updates.",
        "delivery": "Standard retail delivery is estimated at 3–7 business days depending on destination and shipping tier.",
    },
    "returns": {
        "policy": "Return or exchange requests should be raised through the support workflow with your order ID within 30 days.",
    },
    "refunds": {
        "policy": "Refunds are credited to the original payment method within 5–7 business days once returned merchandise is inspected and approved.",
    },
    "store": {
        "hours": "Store hours are 10:00 AM to 8:00 PM, Monday through Saturday.",
        "location": "The demo retail store is configured as a local flagship store; location coordinates are available via the store directory.",
        "contact": "For escalated assistance, please contact customer-support@smartretail.local with your order ID.",
    },
    "products": {
        "recommendation": "I can help narrow your choice across product category, features, and customer sentiment.",
        "catalog": "The catalog currently showcases Bottle, Mug, Headphones, and Shoe categories.",
    },
    "payment": {
        "methods": "Supported retail payment channels include Credit/Debit Cards, UPI, Net Banking, and Cash on Delivery.",
    },
}


def _load_knowledge() -> dict:
    knowledge_path = settings.retail_knowledge_path
    if knowledge_path.exists():
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # merge with defaults if any keys missing
                for k, v in DEFAULT_KNOWLEDGE.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception as exc:
            logger.warning("Failed to read knowledge file from %s: %s. Using fallback defaults.", knowledge_path, exc)
    return DEFAULT_KNOWLEDGE


KNOWLEDGE = _load_knowledge()

INTENT_PATTERNS: dict[str, list[tuple[str, int]]] = {
    "greeting": [
        ("hello", 3), ("hi", 2), ("hey", 2), ("good morning", 3),
        ("good afternoon", 3), ("good evening", 3), ("greetings", 3), ("namaste", 3),
    ],
    "order_status": [
        ("order status", 6), ("track my order", 6), ("track order", 6),
        ("where is my order", 6), ("order tracking", 6), ("order update", 5),
        ("package status", 5), ("my order", 3), ("order", 2),
    ],
    "shipping": [
        ("delivery", 5), ("shipping", 5), ("dispatch", 5), ("deliver", 4),
        ("courier", 4), ("arrive", 4), ("how long to deliver", 6), ("delivery time", 6),
    ],
    "returns": [
        ("return policy", 6), ("return", 5), ("exchange", 6), ("replace", 5),
        ("send back", 6), ("return item", 6), ("exchange item", 6),
    ],
    "refunds": [
        ("refund policy", 6), ("refund", 6), ("money back", 6), ("reimbursement", 6),
        ("when will i get my refund", 7), ("refund status", 6),
    ],
    "store_info": [
        ("store hours", 6), ("opening hours", 6), ("closing time", 6),
        ("open today", 5), ("timing", 4), ("address", 5), ("location", 5),
        ("where is the store", 6), ("store location", 6),
    ],
    "products": [
        ("recommend", 5), ("product", 4), ("catalog", 5), ("headphones", 5),
        ("shoe", 5), ("bottle", 5), ("mug", 5), ("inventory", 4),
        ("features", 3), ("compare", 4), ("buy", 3),
    ],
    "payment": [
        ("payment method", 6), ("payment", 5), ("pay", 4), ("upi", 5),
        ("credit card", 5), ("cash on delivery", 7), ("cod", 5),
        ("billing", 4), ("payment options", 6),
    ],
    "support": [
        ("customer support", 6), ("customer service", 6), ("talk to human", 6),
        ("contact support", 6), ("help desk", 5), ("complaint", 5),
        ("contact", 4), ("issue", 3), ("problem", 3), ("help", 2),
    ],
}

RESPONSES = {
    "greeting": "Hello! Welcome to Smart Retail AI. I can assist with product recommendations, order tracking, shipping, returns, refunds, payments, store hours, and support.",
    "order_status": KNOWLEDGE["shipping"]["tracking"],
    "shipping": KNOWLEDGE["shipping"]["delivery"],
    "returns": KNOWLEDGE["returns"]["policy"],
    "refunds": KNOWLEDGE["refunds"]["policy"],
    "store_info": f"{KNOWLEDGE['store']['hours']} {KNOWLEDGE['store']['location']}",
    "products": f"{KNOWLEDGE['products']['recommendation']} {KNOWLEDGE['products']['catalog']}",
    "payment": KNOWLEDGE["payment"]["methods"],
    "support": KNOWLEDGE["store"]["contact"],
    "fallback": (
        "I didn't quite catch that. You can ask about product recommendations, order status, "
        "shipping, returns, refunds, payment options, store hours, or contact customer support."
    ),
}


def _normalise(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def classify_message(message: str) -> dict:
    """Classify retail user queries into one of 9 distinct intents + fallback."""
    text = _normalise(message)
    if not text:
        return {
            "intent": "fallback",
            "score": 0,
            "matched_terms": [],
            "response": RESPONSES["fallback"],
        }

    scores: dict[str, int] = {intent: 0 for intent in INTENT_PATTERNS}
    matches: dict[str, list[str]] = {intent: [] for intent in INTENT_PATTERNS}

    for intent, patterns in INTENT_PATTERNS.items():
        for phrase, weight in patterns:
            # Word boundary search for multi-word or single-word phrases
            pattern = rf"\b{re.escape(phrase)}\b"
            if re.search(pattern, text):
                scores[intent] += weight
                matches[intent].append(phrase)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Greeting resolution: If greeting is present and only generic support matched, retain greeting
    if scores["greeting"] > 0 and all(
        scores[intent] == 0 for intent in scores if intent not in {"greeting", "support"}
    ):
        final_intent = "greeting"
        final_score = scores["greeting"]
    elif best_score == 0:
        final_intent = "fallback"
        final_score = 0
    else:
        final_intent = best_intent
        final_score = best_score

    return {
        "intent": final_intent,
        "score": final_score,
        "matched_terms": matches.get(final_intent, []),
        "response": RESPONSES[final_intent],
    }


def reply(message: str) -> dict:
    result = classify_message(message)
    return {"intent": result["intent"], "response": result["response"]}
