from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv(
        "SMART_RETAIL_APP_NAME",
        "AI-Powered Smart Retail & Customer Intelligence Platform",
    )
    app_version: str = os.getenv("SMART_RETAIL_APP_VERSION", "1.1.0")
    environment: str = os.getenv("SMART_RETAIL_ENV", "development")
    log_level: str = os.getenv("SMART_RETAIL_LOG_LEVEL", "INFO")
    max_image_bytes: int = int(os.getenv("SMART_RETAIL_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
    product_model_path: Path = ROOT / "models" / "product_classifier.joblib"
    sentiment_model_path: Path = ROOT / "models" / "sentiment_classifier.joblib"
    face_db_path: Path = ROOT / "models" / "face_db.pkl"


settings = Settings()
