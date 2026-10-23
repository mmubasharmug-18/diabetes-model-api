import pytest

from app.audit_log import AuditLog


@pytest.fixture
def log(tmp_path):
    audit = AuditLog(tmp_path / "nested" / "audit.db")
    audit.init()
    return audit


def add_success(log, prediction=1, probability=0.8, latency=5.0):
    return log.record(status="success", http_status=200, request_id="r-ok", input_data={"age": 50},
                      prediction=prediction, probability=probability, latency_ms=latency, model_version="v1")


def add_failure(log, http_status=400, error="age is wrong"):
    return log.record(status="failure", http_status=http_status, request_id="r-bad",
                      input_data={"age": 5}, error=error, model_version="v1")


def test_init_creates_the_folder_and_the_file(log):
    assert log.path.exists()
    assert log.recent() == []


def test_init_can_run_twice(log):
    add_success(log)
    log.init()
    assert len(log.recent()) == 1


def test_success_is_stored_with_everything(log):
    assert add_success(log) is True
    [entry] = log.recent()
    assert entry["status"] == "success"
    assert entry["http_status"] == 200
    assert entry["endpoint"] == "/predict"
    assert entry["request_id"] == "r-ok"
    assert entry["input"] == {"age": 50}
    assert entry["prediction"] == 1
    assert entry["probability"] == 0.8
    assert entry["latency_ms"] == 5.0
    assert entry["model_version"] == "v1"
    assert entry["error"] is None
    assert entry["timestamp"].endswith("+00:00")


def test_failure_keeps_the_error_and_the_input(log):
    add_failure(log)
    [entry] = log.recent()
    assert entry["status"] == "failure"
    assert entry["error"] == "age is wrong"
    assert entry["input"] == {"age": 5}
    assert entry["prediction"] is None


def test_newest_entry_comes_first(log):
    add_success(log)
    add_failure(log)
    assert [e["status"] for e in log.recent()] == ["failure", "success"]


def test_limit(log):
    for _ in range(5):
        add_success(log)
    assert len(log.recent(limit=3)) == 3


def test_filter_by_status(log):
    add_success(log)
    add_failure(log)
    add_failure(log)
    assert len(log.recent(status="failure")) == 2
    assert len(log.recent(status="success")) == 1


def test_summary_counts(log):
    add_success(log, latency=4.0)
    add_success(log, latency=6.0)
    add_failure(log, 400)
    add_failure(log, 503)
    assert log.summary() == {
        "total": 4,
        "successes": 2,
        "failures": 2,
        "success_rate": 0.5,
        "average_latency_ms": 5.0,
        "failures_by_status": {"400": 1, "503": 1},
    }


def test_summary_of_an_empty_log(log):
    assert log.summary() == {
        "total": 0, "successes": 0, "failures": 0, "success_rate": 0.0,
        "average_latency_ms": None, "failures_by_status": {},
    }


def test_input_that_is_not_json_is_kept_as_text(log):
    log.record(status="failure", http_status=400, input_data="{not json", error="bad")
    assert log.recent()[0]["input"] == "{not json"


def test_very_long_input_is_cut_and_still_readable(log):
    log.record(status="failure", http_status=400, input_data="x" * 10000, error="bad")
    stored = log.recent()[0]["input"]
    assert isinstance(stored, str) and len(stored) < 5000


def test_an_invalid_status_is_refused_without_crashing(log):
    assert log.record(status="weird", http_status=200) is False
    assert log.recent() == []


def test_a_broken_database_path_does_not_crash(tmp_path):
    assert AuditLog(tmp_path).record(status="success", http_status=200) is False


def test_configure_switches_to_a_new_file(log, tmp_path):
    add_success(log)
    log.configure(tmp_path / "other.db")
    assert log.recent() == []
