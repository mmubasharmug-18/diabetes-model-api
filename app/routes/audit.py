from typing import Literal, Optional

from fastapi import APIRouter, Query

from app.schemas import AuditSummary, LogEntry
from app.state import audit

router = APIRouter(tags=["audit"])


@router.get("/audit", response_model=list[LogEntry], summary="Recent logged requests, newest first")
def audit_entries(
    limit: int = Query(20, ge=1, le=200, description="How many entries to return"),
    status: Optional[Literal["success", "failure"]] = Query(None, description="Only show this kind of request"),
):
    return audit.recent(limit, status)


@router.get("/audit/summary", response_model=AuditSummary, summary="Counts of successful and failed requests")
def audit_summary():
    return audit.summary()
