import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.handlers import validation_error_handler
from app.routes import audit as audit_routes
from app.routes import health, predict
from app.state import audit, service

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    audit.init()
    if not service.load():
        logger.warning("No model found at %s. Run: python -m ml.train", service.model_path)
    yield


app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0",
    description="Serves a trained diabetes model and logs every /predict request to SQLite.",
    lifespan=lifespan,
)

app.add_exception_handler(RequestValidationError, validation_error_handler)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(audit_routes.router)
