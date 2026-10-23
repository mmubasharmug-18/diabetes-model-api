import pytest

from app.model_service import ModelNotLoaded, ModelService
from ml.train import save_model
from tests.helpers import VALID


@pytest.fixture
def service(trained):
    model, info = trained
    svc = ModelService("missing.joblib", "missing.json")
    svc.set_model(model, info)
    return svc


def test_predict_before_loading_raises():
    with pytest.raises(ModelNotLoaded):
        ModelService("missing.joblib", "missing.json").predict(VALID)


def test_result_has_label_and_probabilities(service):
    result = service.predict(VALID)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("diabetes", "no diabetes")
    assert set(result["probability"]) == {"no_diabetes", "diabetes"}


def test_probabilities_add_up_to_one(service):
    p = service.predict(VALID)["probability"]
    assert abs(p["no_diabetes"] + p["diabetes"] - 1) < 0.001


def test_label_follows_the_higher_probability(service):
    result = service.predict(VALID)
    higher_is_diabetes = result["probability"]["diabetes"] > result["probability"]["no_diabetes"]
    assert (result["prediction"] == 1) == higher_is_diabetes


def test_same_input_gives_the_same_answer(service):
    assert service.predict(VALID) == service.predict(VALID)


def test_unmeasured_values_given_as_zero_still_work(service):
    result = service.predict({**VALID, "insulin": 0, "skin_thickness": 0})
    assert result["prediction"] in (0, 1)


def test_high_glucose_scores_higher_than_low_glucose(service):
    low = service.predict({**VALID, "glucose": 70, "bmi": 22})["probability"]["diabetes"]
    high = service.predict({**VALID, "glucose": 200, "bmi": 40})["probability"]["diabetes"]
    assert high > low


def test_load_from_disk_gives_the_same_answers(trained, tmp_path):
    model, info = trained
    save_model(model, info, tmp_path / "m.joblib", tmp_path / "m.json")
    fresh = ModelService(tmp_path / "m.joblib", tmp_path / "m.json")
    assert fresh.load() is True
    assert fresh.version == info["version"]
    reference = ModelService("x", "y")
    reference.set_model(model, info)
    assert fresh.predict(VALID) == reference.predict(VALID)


def test_load_returns_false_when_the_file_is_missing(tmp_path):
    service = ModelService(tmp_path / "nope.joblib", tmp_path / "nope.json")
    assert service.load() is False
    assert service.loaded is False


def test_version_is_unknown_without_info():
    assert ModelService("a", "b").version == "unknown"
