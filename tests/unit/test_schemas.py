import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest
from tests.helpers import BOUNDARY_OK, MISSING, OUT_OF_RANGE, VALID, WRONG_TYPES, payload_with, payload_without


def failed_fields(payload):
    with pytest.raises(ValidationError) as info:
        PredictRequest.model_validate(payload)
    return {".".join(str(p) for p in e["loc"]) for e in info.value.errors()}


def test_valid_payload_is_accepted():
    request = PredictRequest.model_validate(VALID)
    assert request.glucose == 148.0
    assert request.age == 50


def test_whole_numbers_are_fine_for_decimal_fields():
    assert PredictRequest.model_validate(payload_with("bmi", 30)).bmi == 30.0


def test_extra_fields_are_ignored():
    request = PredictRequest.model_validate({**VALID, "nickname": "x"})
    assert not hasattr(request, "nickname")


@pytest.mark.parametrize("field,value", BOUNDARY_OK)
def test_values_on_the_limits_are_accepted(field, value):
    PredictRequest.model_validate(payload_with(field, value))


@pytest.mark.parametrize("field,value", OUT_OF_RANGE)
def test_out_of_range_is_rejected(field, value):
    assert failed_fields(payload_with(field, value)) == {field}


@pytest.mark.parametrize("field,value", WRONG_TYPES)
def test_wrong_types_are_rejected(field, value):
    assert failed_fields(payload_with(field, value)) == {field}


@pytest.mark.parametrize("field", MISSING)
def test_missing_fields_are_rejected(field):
    assert failed_fields(payload_without(field)) == {field}


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nan_and_infinity_are_rejected(value):
    assert failed_fields(payload_with("glucose", value)) == {"glucose"}
