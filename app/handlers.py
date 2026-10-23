import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import format_errors
from app.state import audit, service


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = format_errors(exc.errors())
    content = {"detail": details}

    if request.url.path == "/predict":
        request_id = uuid.uuid4().hex
        audit.record(
            request_id=request_id,
            status="failure",
            http_status=400,
            input_data=exc.body,
            error="; ".join(item["message"] for item in details),
            model_version=service.version,
        )
        content["request_id"] = request_id

    return JSONResponse(status_code=400, content=content)
