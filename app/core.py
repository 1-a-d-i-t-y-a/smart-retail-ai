from __future__ import annotations

from threading import Lock

COUNTERS = {
    "total_requests": 0,
    "product_requests": 0,
    "face_requests": 0,
    "sentiment_requests": 0,
    "chatbot_requests": 0,
}
_LOCK = Lock()


def increment(name: str) -> None:
    with _LOCK:
        COUNTERS["total_requests"] += 1
        COUNTERS[name] += 1


def snapshot() -> dict:
    with _LOCK:
        return dict(COUNTERS)
