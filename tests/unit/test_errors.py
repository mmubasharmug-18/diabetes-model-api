import pytest
from pydantic import ValidationError

from app.errors import field_name, format_errors, short
from app.schemas import PredictRequest
from tests.helpers import payload_with, payload_without


def api_errors(payload):
    with pytest.raises(ValidationError) as info:
        PredictRequest.model_validate(payload)
    return format_errors([{**e, "loc": ("body",) + e["loc"]} for e in info.value.errors()])


def test_message_states_the_rule_and_the_value():
    assert api_errors(payload_with("glucose", 500)) == [
        {"field": "glucose", "message": "glucose must be a number between 40 and 400 (got 500)"}
    ]


def test_whole_number_rule_is_worded_that_way():
    [error] = api_errors(payload_with("age", 15))
    assert error["message"].startswith("age must be a whole number between 21 and 120")


def test_missing_field_says_so():
    [error] = api_errors(payload_without("bmi"))
    assert error["message"].endswith("(field is missing)")


def test_every_wrong_field_is_reported_at_once():
    payload = {"pregnancies": -1, "glucose": 0, "blood_pressure": 0, "skin_thickness": 500,
               "insulin": -5, "bmi": 200, "diabetes_pedigree_function": 9, "age": 5}
    assert len(api_errors(payload)) == 8


def test_invalid_json_body():
    errors = format_errors([{"type": "json_invalid", "loc": ("body", 3), "msg": "JSON decode error"}])
    assert errors == [{"field": "body", "message": "request body must be valid JSON"}]


def test_missing_body():
    errors = format_errors([{"type": "missing", "loc": ("body",), "msg": "Field required"}])
    assert errors[0]["message"] == "request body is required"


def test_body_that_is_not_an_object():
    errors = format_errors([{"type": "model_attributes_type", "loc": ("body",), "msg": "bad", "input": []}])
    assert errors[0]["message"] == "request body must be a JSON object"


def test_query_parameters():
    errors = format_errors([
        {"type": "greater_than_equal", "loc": ("query", "limit"), "msg": "bad", "input": "0"},
        {"type": "literal_error", "loc": ("query", "status"), "msg": "bad", "input": "x"},
    ])
    assert errors[0]["message"] == "limit must be a whole number between 1 and 200 (got '0')"
    assert errors[1]["message"] == "status must be 'success' or 'failure' (got 'x')"


def test_unknown_field_falls_back_to_the_library_message():
    errors = format_errors([{"type": "value_error", "loc": ("body", "nickname"), "msg": "too odd", "input": 1}])
    assert errors[0]["message"].startswith("nickname is invalid: too odd")


def test_duplicate_errors_are_merged():
    one = {"type": "int_type", "loc": ("body", "age"), "msg": "x", "input": "a"}
    assert len(format_errors([one, dict(one)])) == 1


def test_helpers():
    assert field_name(("body", "age")) == "age"
    assert short("x" * 100).endswith("...") and len(short("x" * 100)) == 40
