from __future__ import annotations

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core import increment
from app.schemas.api import FaceResponse
from app.services.face_service import recognize_face
from app.utils.validation import validate_image_payload

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Face Recognition"])


@router.post(
    "/recognize-face",
    response_model=FaceResponse,
    summary="Verify customer face image",
    description="Detects face via Haar Cascade and compares against registered customer templates using SSIM and Pearson correlation.",
    responses={
        200: {"description": "Face verification completed (matched or unmatched)."},
        400: {"description": "Invalid image payload or corrupted file."},
        503: {"description": "Face database not initialized or unavailable."},
    },
)
async def recognize_face_route(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        validate_image_payload(file.content_type, payload)
        result = recognize_face(payload)
        increment("face_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Face database missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition database is unavailable. Please build models first.",
        ) from exc
    except RuntimeError as exc:
        logger.error("Face service runtime failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service runtime error.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in recognize_face_route: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face recognition service error.",
        ) from exc
