from __future__ import annotations

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core import increment
from app.schemas.api import ProductResponse
from app.services.product_service import predict_product
from app.utils.validation import PayloadTooLargeError, read_bounded_upload, validate_image_payload
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Product Classification"])


@router.post(
    "/classify-product",
    response_model=ProductResponse,
    summary="Classify retail product image",
    description="Extracts HOG feature descriptors and classifies image into Bottle, Mug, Headphones, or Shoe.",
    responses={
        200: {"description": "Product classification succeeded."},
        400: {"description": "Invalid image payload or corrupted file."},
        413: {"description": "Upload exceeds the configured size limit."},
        503: {"description": "Product classification model not loaded or unavailable."},
    },
)
async def classify_product(file: UploadFile = File(...)):
    try:
        payload = await read_bounded_upload(file, settings.max_image_bytes)
        validate_image_payload(file.content_type, payload)
        result = predict_product(payload)
        increment("product_requests")
        return result
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Product model missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product classification model is unavailable. Please ensure models are built.",
        ) from exc
    except RuntimeError as exc:
        logger.error("Product model runtime failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product classification model runtime error.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in classify_product: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Product classification service error.",
        ) from exc
