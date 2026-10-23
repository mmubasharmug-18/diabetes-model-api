import time
import uuid

from fastapi import APIRouter, HTTPException

from app.model_service import ModelNotLoaded
from app.schemas import PredictRequest, PredictResponse
from app.state import audit, service

router = APIRouter(tags=["predict"])


def log_failure(request_id, features, http_status, message, started):
    audit.record(
        request_id=request_id,
        status="failure",
        http_status=http_status,
        input_data=features,
        error=message,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        model_version=service.version,
    )


@router.post("/predict", response_model=PredictResponse, summary="Predict diabetes from 8 measurements")
def predict(data: PredictRequest):
    request_id = uuid.uuid4().hex
    features = data.model_dump()
    started = time.perf_counter()

    try:
        result = service.predict(features)
    except ModelNotLoaded as exc:
        log_failure(request_id, features, 503, str(exc), started)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log_failure(request_id, features, 500, f"{type(exc).__name__}: {exc}", started)
        raise HTTPException(status_code=500, detail=f"Prediction failed. See the audit log for request {request_id}.")

    audit.record(
        request_id=request_id,
        status="success",
        http_status=200,
        input_data=features,
        prediction=result["prediction"],
        probability=result["probability"]["diabetes"],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        model_version=service.version,
    )
    return PredictResponse(request_id=request_id, model_version=service.version, **result)
