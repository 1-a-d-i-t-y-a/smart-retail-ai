from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("Hello, can you help me?", "greeting"),
        ("Namaste! Good morning", "greeting"),
        ("Where is my order?", "order_status"),
        ("Can I track my order update?", "order_status"),
        ("How long will shipping and delivery take?", "shipping"),
        ("Can I return this shoe if the size does not fit?", "returns"),
        ("I want a refund for my damaged item", "refunds"),
        ("When will I get my refund back?", "refunds"),
        ("What are your store hours and address?", "store_info"),
        ("Can you recommend high quality headphones?", "products"),
        ("Can I pay with credit card or UPI?", "payment"),
        ("I have a complaint, connect me to customer service", "support"),
        ("Quantum physics teleportation algorithm", "fallback"),
    ],
)
def test_chatbot_intents(client: TestClient, message: str, expected_intent: str):
    response = client.post("/chatbot", json={"message": message})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == expected_intent
    assert len(data["response"]) > 0


def test_chatbot_punctuation_normalization(client: TestClient):
    response = client.post("/chatbot", json={"message": "!!!HELLO??? WORLD!!!"})
    assert response.status_code == 200
    assert response.json()["intent"] == "greeting"


def test_chatbot_empty_message_validation(client: TestClient):
    response = client.post("/chatbot", json={"message": ""})
    assert response.status_code == 422


def test_chatbot_whitespace_validation(client: TestClient):
    response = client.post("/chatbot", json={"message": "    \n\t  "})
    assert response.status_code == 422
