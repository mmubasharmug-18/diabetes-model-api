from app.state import service
from tests.helpers import VALID, payload_with


def entries(client, **params):
    return client.get("/audit", params=params).json()


def test_successful_request_is_logged(client):
    reply = client.post("/predict", json=VALID).json()
    [entry] = entries(client)
    assert entry["request_id"] == reply["request_id"]
    assert entry["status"] == "success"
    assert entry["http_status"] == 200
    assert entry["endpoint"] == "/predict"
    assert entry["input"] == VALID
    assert entry["prediction"] == reply["prediction"]
    assert entry["probability"] == reply["probability"]["diabetes"]
    assert entry["model_version"] == reply["model_version"]
    assert entry["latency_ms"] >= 0
    assert entry["error"] is None


def test_validation_failure_is_logged_with_the_reason_and_the_input(client):
    reply = client.post("/predict", json=payload_with("glucose", 500)).json()
    [entry] = entries(client)
    assert entry["request_id"] == reply["request_id"]
    assert entry["status"] == "failure"
    assert entry["http_status"] == 400
    assert "glucose must be a number between 40 and 400" in entry["error"]
    assert entry["input"]["glucose"] == 500
    assert entry["prediction"] is None


def test_broken_json_is_logged_as_text(client):
    client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    [entry] = entries(client)
    assert entry["status"] == "failure"
    assert "not json" in entry["input"]


def test_model_not_loaded_is_logged_as_503(client):
    service.set_model(None)
    client.post("/predict", json=VALID)
    [entry] = entries(client)
    assert entry["http_status"] == 503
    assert "ml.train" in entry["error"]
    assert entry["input"] == VALID


def test_unexpected_error_is_logged_as_500(client, monkeypatch):
    def broken(features):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "predict", broken)
    client.post("/predict", json=VALID)
    [entry] = entries(client)
    assert entry["http_status"] == 500
    assert entry["error"] == "RuntimeError: boom"


def test_log_is_newest_first(client):
    client.post("/predict", json=VALID)
    client.post("/predict", json=payload_with("age", 5))
    client.post("/predict", json=VALID)
    assert [e["status"] for e in entries(client)] == ["success", "failure", "success"]


def test_every_request_gets_its_own_id(client):
    for _ in range(3):
        client.post("/predict", json=VALID)
    ids = [e["request_id"] for e in entries(client)]
    assert len(set(ids)) == 3


def test_filter_and_limit(client):
    for _ in range(3):
        client.post("/predict", json=VALID)
    client.post("/predict", json=payload_with("age", 5))
    assert len(entries(client, status="failure")) == 1
    assert len(entries(client, status="success")) == 3
    assert len(entries(client, limit=2)) == 2


def test_bad_audit_parameters_give_400(client):
    assert client.get("/audit", params={"limit": 0}).status_code == 400
    assert client.get("/audit", params={"limit": 500}).status_code == 400
    assert client.get("/audit", params={"status": "other"}).status_code == 400


def test_summary_counts(client):
    client.post("/predict", json=VALID)
    client.post("/predict", json=VALID)
    client.post("/predict", json=payload_with("age", 5))
    service.set_model(None)
    client.post("/predict", json=VALID)
    summary = client.get("/audit/summary").json()
    assert summary["total"] == 4
    assert summary["successes"] == 2
    assert summary["failures"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["failures_by_status"] == {"400": 1, "503": 1}
    assert summary["average_latency_ms"] >= 0


def test_summary_of_an_empty_log(client):
    summary = client.get("/audit/summary").json()
    assert summary["total"] == 0
    assert summary["average_latency_ms"] is None


def test_reading_the_log_does_not_add_to_it(client):
    client.post("/predict", json=VALID)
    client.get("/audit")
    client.get("/audit/summary")
    client.get("/health")
    assert client.get("/audit/summary").json()["total"] == 1
