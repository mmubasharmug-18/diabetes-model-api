import json
from pathlib import Path

import joblib
import pandas as pd

from ml.features import API_TO_COLUMN, FEATURES


class ModelNotLoaded(Exception):
    pass


class ModelService:
    def __init__(self, model_path, info_path):
        self.model_path = Path(model_path)
        self.info_path = Path(info_path)
        self.model = None
        self.info = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return self.info.get("version", "unknown")

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.model = joblib.load(self.model_path)
        self.info = json.loads(self.info_path.read_text()) if self.info_path.exists() else {}
        return True

    def set_model(self, model, info=None) -> None:
        self.model = model
        self.info = info or {}

    def predict(self, features: dict) -> dict:
        if self.model is None:
            raise ModelNotLoaded("Model is not loaded. Train it first with: python -m ml.train")

        row = {API_TO_COLUMN[name]: [value] for name, value in features.items()}
        frame = pd.DataFrame(row, columns=FEATURES)
        proba = self.model.predict_proba(frame)[0]
        classes = list(self.model.classes_)
        diabetes = float(proba[classes.index(1)])
        no_diabetes = float(proba[classes.index(0)])
        prediction = 1 if diabetes > no_diabetes else 0

        return {
            "prediction": prediction,
            "label": "diabetes" if prediction == 1 else "no diabetes",
            "probability": {"no_diabetes": round(no_diabetes, 4), "diabetes": round(diabetes, 4)},
        }
