"""TF-IDF + Logistic Regression sentiment classifier (Positive / Neutral / Negative)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


_TOKEN_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class SentimentModel:
    pipeline: Optional[Pipeline] = None

    def train(self, df: pd.DataFrame, text_col: str = "review", label_col: str = "sentiment"):
        X = df[text_col].astype(str).apply(clean_text)
        y = df[label_col].astype(str)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self.pipeline.fit(X_tr, y_tr)
        report = classification_report(y_te, self.pipeline.predict(X_te), output_dict=True, zero_division=0)
        return report

    def predict(self, text: str) -> Tuple[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained/loaded.")
        clean = clean_text(text)
        label = self.pipeline.predict([clean])[0]
        proba = float(np.max(self.pipeline.predict_proba([clean])[0]))
        return str(label), proba

    def save(self, path: str = "models/sentiment_model.joblib") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str = "models/sentiment_model.joblib") -> "SentimentModel":
        self.pipeline = joblib.load(path)
        return self
