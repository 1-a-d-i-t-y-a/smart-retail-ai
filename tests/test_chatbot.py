from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("Hello, can you help me?", "greeting"),
        ("Can you recommend headphones?", "products"),
        ("Where is my order?", "order_status"),
        ("How long will delivery take?", "shipping"),
        ("I want a refund", "returns"),
        ("What are the store hours?", "store_info"),
        ("Can I pay by UPI?", "payment"),
        ("I have a complaint", "support"),
        ("Tell me a random story please", "fallback"),
    ],
)
def test_chatbot_intents(client: TestClient, message: str, expected_intent: str):
    response = client.post("/chatbot", json={"message": message})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == expected_intent
    assert len(data["response"]) > 0


def test_chatbot_empty_message_validation(client: TestClient):
    response = client.post("/chatbot", json={"message": ""})
    assert response.status_code == 422
