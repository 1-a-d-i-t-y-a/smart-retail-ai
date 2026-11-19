from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    app_name: str = os.getenv(
        "SMART_RETAIL_APP_NAME",
        "AI-Powered Smart Retail & Customer Intelligence Platform",
    )
    app_version: str = os.getenv("SMART_RETAIL_APP_VERSION", "1.2.0")
    environment: str = os.getenv("SMART_RETAIL_ENV", "development")
    log_level: str = os.getenv("SMART_RETAIL_LOG_LEVEL", "INFO")
    max_image_bytes: int = int(os.getenv("SMART_RETAIL_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
    min_image_dimension: int = int(os.getenv("SMART_RETAIL_MIN_IMAGE_DIM", "10"))
    max_image_dimension: int = int(os.getenv("SMART_RETAIL_MAX_IMAGE_DIM", "4096"))
    product_model_path: Path = Path(
        os.getenv("SMART_RETAIL_PRODUCT_MODEL", str(ROOT / "models" / "product_classifier.joblib"))
    )
    sentiment_model_path: Path = Path(
        os.getenv("SMART_RETAIL_SENTIMENT_MODEL", str(ROOT / "models" / "sentiment_classifier.joblib"))
    )
    face_db_path: Path = Path(
        os.getenv("SMART_RETAIL_FACE_DB", str(ROOT / "models" / "face_db.pkl"))
    )
    cascade_path: Path = Path(
        os.getenv("SMART_RETAIL_CASCADE_PATH", str(ROOT / "data" / "haarcascades" / "haarcascade_frontalface_default.xml"))
    )
    retail_knowledge_path: Path = Path(
        os.getenv("SMART_RETAIL_KNOWLEDGE_PATH", str(ROOT / "data" / "chatbot" / "retail_knowledge.json"))
    )


settings = Settings()
