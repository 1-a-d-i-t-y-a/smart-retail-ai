from __future__ import annotations

import os
from pathlib import Path

from app.config import Settings, settings


def test_default_settings():
    assert settings.app_name == "AI-Powered Smart Retail & Customer Intelligence Platform"
    assert settings.app_version == "1.2.0"
    assert settings.max_image_bytes == 5 * 1024 * 1024
    assert settings.min_image_dimension == 10
    assert settings.max_image_dimension == 4096
    assert isinstance(settings.product_model_path, Path)
    assert isinstance(settings.sentiment_model_path, Path)
    assert isinstance(settings.face_db_path, Path)


def test_custom_settings_environment_overrides(monkeypatch):
    monkeypatch.setenv("SMART_RETAIL_APP_NAME", "Custom Retail Test Platform")
    monkeypatch.setenv("SMART_RETAIL_APP_VERSION", "2.0.0-test")
    monkeypatch.setenv("SMART_RETAIL_MAX_IMAGE_BYTES", "1048576")
    monkeypatch.setenv("SMART_RETAIL_MIN_IMAGE_DIM", "20")
    monkeypatch.setenv("SMART_RETAIL_MAX_IMAGE_DIM", "2048")

    custom = Settings(
        app_name=os.getenv("SMART_RETAIL_APP_NAME"),
        app_version=os.getenv("SMART_RETAIL_APP_VERSION"),
        max_image_bytes=int(os.getenv("SMART_RETAIL_MAX_IMAGE_BYTES")),
        min_image_dimension=int(os.getenv("SMART_RETAIL_MIN_IMAGE_DIM")),
        max_image_dimension=int(os.getenv("SMART_RETAIL_MAX_IMAGE_DIM")),
    )

    assert custom.app_name == "Custom Retail Test Platform"
    assert custom.app_version == "2.0.0-test"
    assert custom.max_image_bytes == 1048576
    assert custom.min_image_dimension == 20
    assert custom.max_image_dimension == 2048
