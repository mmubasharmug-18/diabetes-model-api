from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas import HealthOut
from app.state import service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Is the service up and is the model loaded?")
def health():
    return HealthOut(
        status="ok",
        model_loaded=service.loaded,
        model_version=service.version if service.loaded else None,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
