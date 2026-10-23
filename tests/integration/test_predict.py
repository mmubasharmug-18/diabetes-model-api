import pytest

from app.state import service
from tests.helpers import BOUNDARY_OK, MISSING, OUT_OF_RANGE, VALID, WRONG_TYPES, payload_with, payload_without


def test_valid_request_gets_a_prediction(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("diabetes" if body["prediction"] == 1 else "no diabetes")
    assert set(body["probability"]) == {"no_diabetes", "diabetes"}
    assert abs(sum(body["probability"].values()) - 1) < 0.001
    assert len(body["request_id"]) == 32
    assert body["model_version"].startswith("rf-")


def test_same_input_gives_the_same_prediction(client):
    first = client.post("/predict", json=VALID).json()
    second = client.post("/predict", json=VALID).json()
    assert first["prediction"] == second["prediction"]
    assert first["probability"] == second["probability"]
    assert first["request_id"] != second["request_id"]


@pytest.mark.parametrize("field,value", BOUNDARY_OK)
def test_values_on_the_limits_are_accepted(client, field, value):
    assert client.post("/predict", json=payload_with(field, value)).status_code == 200


@pytest.mark.parametrize("field,value", OUT_OF_RANGE)
def test_out_of_range_gives_400(client, field, value):
    r = client.post("/predict", json=payload_with(field, value))
    assert r.status_code == 400
    assert r.json()["detail"][0]["field"] == field
    assert field in r.json()["detail"][0]["message"]


@pytest.mark.parametrize("field,value", WRONG_TYPES)
def test_wrong_data_type_gives_400(client, field, value):
    r = client.post("/predict", json=payload_with(field, value))
    assert r.status_code == 400
    assert r.json()["detail"][0]["field"] == field


@pytest.mark.parametrize("field", MISSING)
def test_missing_field_gives_400(client, field):
    r = client.post("/predict", json=payload_without(field))
    assert r.status_code == 400
    assert r.json()["detail"][0]["field"] == field
    assert "missing" in r.json()["detail"][0]["message"]


def test_all_problems_are_listed_together(client):
    bad = {"pregnancies": -1, "glucose": 0, "blood_pressure": 0, "skin_thickness": 500,
           "insulin": -5, "bmi": 200, "diabetes_pedigree_function": 9, "age": 5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert len(r.json()["detail"]) == 8


def test_invalid_json_gives_400(client):
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["detail"][0]["field"] == "body"


def test_empty_body_gives_400(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert r.json()["detail"][0]["message"] == "request body is required"


def test_list_instead_of_object_gives_400(client):
    r = client.post("/predict", json=[VALID])
    assert r.status_code == 400
    assert r.json()["detail"][0]["message"] == "request body must be a JSON object"


def test_a_400_response_carries_a_request_id(client):
    r = client.post("/predict", json=payload_with("age", 5))
    assert len(r.json()["request_id"]) == 32


def test_model_not_loaded_gives_503(client):
    service.set_model(None)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "ml.train" in r.json()["detail"]


def test_unexpected_error_gives_500(client, monkeypatch):
    def broken(features):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "predict", broken)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert "boom" not in r.json()["detail"]


def test_health_reports_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("rf-")
    service.set_model(None)
    assert client.get("/health").json()["model_loaded"] is False


def test_docs_and_schema_are_served(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/health", "/audit", "/audit/summary"} <= set(paths)
