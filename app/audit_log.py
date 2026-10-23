import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("audit")

TABLE = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    timestamp TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'failure')),
    http_status INTEGER NOT NULL,
    input_json TEXT,
    prediction INTEGER,
    probability REAL,
    error TEXT,
    latency_ms REAL,
    model_version TEXT
)
"""

COLUMNS = "id, request_id, timestamp, endpoint, status, http_status, input_json, prediction, probability, error, latency_ms, model_version"


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)

    def configure(self, path) -> None:
        self.path = Path(path)
        self.init()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(TABLE)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_status ON request_log (status)")

    def record(
        self,
        *,
        status: str,
        http_status: int,
        request_id: Optional[str] = None,
        endpoint: str = "/predict",
        input_data: Any = None,
        prediction: Optional[int] = None,
        probability: Optional[float] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> bool:
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        input_json = json.dumps(input_data, default=str)[:4000]
        try:
            with closing(sqlite3.connect(self.path)) as conn, conn:
                conn.execute(
                    "INSERT INTO request_log (request_id, timestamp, endpoint, status, http_status, input_json,"
                    " prediction, probability, error, latency_ms, model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (request_id, stamp, endpoint, status, http_status, input_json,
                     prediction, probability, error, latency_ms, model_version),
                )
            return True
        except sqlite3.Error:
            logger.exception("could not write to the audit log")
            return False

    def recent(self, limit: int = 20, status: Optional[str] = None) -> list:
        query = f"SELECT {COLUMNS} FROM request_log"
        params = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with closing(sqlite3.connect(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._to_dict(row) for row in rows]

    def summary(self) -> dict:
        with closing(sqlite3.connect(self.path)) as conn:
            total, successes, avg_latency = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(status = 'success'), 0), AVG(CASE WHEN status = 'success' THEN latency_ms END)"
                " FROM request_log"
            ).fetchone()
            by_status = conn.execute(
                "SELECT http_status, COUNT(*) FROM request_log WHERE status = 'failure' GROUP BY http_status"
            ).fetchall()
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 3) if total else 0.0,
            "average_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "failures_by_status": {str(code): count for code, count in by_status},
        }

    @staticmethod
    def _to_dict(row) -> dict:
        item = dict(row)
        raw = item.pop("input_json")
        try:
            item["input"] = json.loads(raw) if raw is not None else None
        except ValueError:
            item["input"] = raw
        return item
