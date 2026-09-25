from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core import increment
from app.schemas.api import FaceResponse
from app.services.face_service import recognize_face
from app.utils.validation import validate_image_payload

router = APIRouter(tags=["Face Recognition"])


@router.post("/recognize-face", response_model=FaceResponse)
async def recognize_face_route(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        validate_image_payload(file.content_type, payload)
        result = recognize_face(payload)
        increment("face_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Face recognition service error.") from exc
