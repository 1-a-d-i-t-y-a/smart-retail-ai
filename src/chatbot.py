"""Retrieval-based retail FAQ chatbot (TF-IDF + cosine similarity)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_FAQS: List[Tuple[str, str]] = [
    ("What are your store hours?",
     "Our stores are open Mon-Sat 10am-9pm and Sun 11am-7pm."),
    ("How do I return a product?",
     "Returns are accepted within 30 days of purchase with a valid receipt. Visit any store or start a return online."),
    ("What is your shipping policy?",
     "Standard shipping takes 3-5 business days and is free for orders over $50."),
    ("Do you offer international shipping?",
     "Yes, we ship to 40+ countries. Duties and taxes are calculated at checkout."),
    ("How can I track my order?",
     "You will receive a tracking email once your order ships. You can also view status in your account dashboard."),
    ("What payment methods do you accept?",
     "We accept Visa, Mastercard, Amex, PayPal, Apple Pay, and Google Pay."),
    ("Do you price match?",
     "Yes, we price match identical items from major retailers within 14 days of purchase."),
    ("How do I contact customer support?",
     "Reach us at support@retailai.com or call 1-800-555-0199 (Mon-Fri, 9am-6pm)."),
    ("Do you have a loyalty program?",
     "Yes! Sign up for RetailAI Rewards to earn 1 point per $1 spent and unlock exclusive discounts."),
    ("How do I change or cancel my order?",
     "Orders can be modified or cancelled within 1 hour of placement from your account page."),
]


@dataclass
class RetailChatbot:
    faqs: List[Tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_FAQS))
    vectorizer: TfidfVectorizer = field(default=None)
    matrix: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        self.fit()

    def fit(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        questions = [q for q, _ in self.faqs]
        self.matrix = self.vectorizer.fit_transform(questions)

    def reply(self, query: str, threshold: float = 0.15) -> Tuple[str, float]:
        vec = self.vectorizer.transform([query])
        sims = cosine_similarity(vec, self.matrix)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score < threshold:
            return (
                "I'm not sure I understood that. Could you rephrase, or contact support@retailai.com?",
                score,
            )
        return self.faqs[idx][1], score

    def save(self, path: str = "models/chatbot.joblib") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"faqs": self.faqs}, path)

    @classmethod
    def load(cls, path: str = "models/chatbot.joblib") -> "RetailChatbot":
        data = joblib.load(path)
        return cls(faqs=data["faqs"])
