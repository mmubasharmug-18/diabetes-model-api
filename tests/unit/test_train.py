import json

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURES, ZERO_AS_MISSING, zeros_to_nan
from ml.train import save_model, train_model
from tests.helpers import make_dataset


def test_training_returns_a_model_and_its_details(trained):
    model, info = trained
    assert hasattr(model, "predict_proba")
    assert info["version"].startswith("rf-")
    assert info["features"] == FEATURES
    assert info["rows"] == 300
    assert set(info["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in info["metrics"].values())


def test_model_beats_a_coin_flip_on_the_stand_in_data(trained):
    assert trained[1]["metrics"]["roc_auc"] > 0.6


def test_missing_column_is_reported():
    with pytest.raises(ValueError, match="Glucose"):
        train_model(make_dataset().drop(columns="Glucose"))


def test_zeros_become_missing_only_where_it_makes_sense():
    frame = pd.DataFrame({name: [0, 5] for name in FEATURES})
    cleaned = zeros_to_nan(frame)
    for name in ZERO_AS_MISSING:
        assert np.isnan(cleaned.loc[0, name])
    assert cleaned.loc[0, "Pregnancies"] == 0
    assert cleaned.loc[0, "Age"] == 0
    assert frame.loc[0, "Glucose"] == 0


def test_save_writes_both_files(trained, tmp_path):
    model, info = trained
    save_model(model, info, tmp_path / "out" / "m.joblib", tmp_path / "out" / "m.json")
    assert (tmp_path / "out" / "m.joblib").exists()
    assert json.loads((tmp_path / "out" / "m.json").read_text())["version"] == info["version"]
