from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core import increment
from app.schemas.api import ProductResponse
from app.services.product_service import predict_product
from app.utils.validation import validate_image_payload

router = APIRouter(tags=["Product Classification"])


@router.post("/classify-product", response_model=ProductResponse)
async def classify_product(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        validate_image_payload(file.content_type, payload)
        result = predict_product(payload)
        increment("product_requests")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Product classification service error.") from exc
