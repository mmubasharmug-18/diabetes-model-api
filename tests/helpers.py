import copy

import numpy as np
import pandas as pd

from ml.features import FEATURES

VALID = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}

OUT_OF_RANGE = [
    ("pregnancies", -1),
    ("pregnancies", 21),
    ("glucose", 39),
    ("glucose", 401),
    ("blood_pressure", 19),
    ("blood_pressure", 201),
    ("skin_thickness", -1),
    ("skin_thickness", 101),
    ("insulin", -1),
    ("insulin", 901),
    ("bmi", 9.9),
    ("bmi", 70.1),
    ("diabetes_pedigree_function", -0.1),
    ("diabetes_pedigree_function", 3.1),
    ("age", 20),
    ("age", 121),
]

BOUNDARY_OK = [
    ("pregnancies", 0),
    ("pregnancies", 20),
    ("glucose", 40),
    ("glucose", 400),
    ("blood_pressure", 20),
    ("blood_pressure", 200),
    ("skin_thickness", 0),
    ("skin_thickness", 100),
    ("insulin", 0),
    ("insulin", 900),
    ("bmi", 10),
    ("bmi", 70),
    ("diabetes_pedigree_function", 0),
    ("diabetes_pedigree_function", 3),
    ("age", 21),
    ("age", 120),
]

WRONG_TYPES = [
    ("pregnancies", "6"),
    ("pregnancies", 6.5),
    ("pregnancies", True),
    ("pregnancies", None),
    ("glucose", "148"),
    ("glucose", True),
    ("glucose", None),
    ("glucose", [148]),
    ("blood_pressure", "72"),
    ("blood_pressure", None),
    ("skin_thickness", "35"),
    ("insulin", "0"),
    ("bmi", "33.6"),
    ("bmi", None),
    ("diabetes_pedigree_function", "0.6"),
    ("age", "50"),
    ("age", 50.5),
    ("age", None),
]

MISSING = list(VALID)


def payload_with(field, value):
    payload = copy.deepcopy(VALID)
    payload[field] = value
    return payload


def payload_without(field):
    payload = copy.deepcopy(VALID)
    del payload[field]
    return payload


def make_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "Pregnancies": rng.integers(0, 15, n),
        "Glucose": rng.normal(120, 30, n).clip(50, 250).round(),
        "BloodPressure": rng.normal(70, 12, n).clip(30, 120).round(),
        "SkinThickness": rng.integers(5, 60, n),
        "Insulin": rng.integers(15, 400, n),
        "BMI": rng.normal(32, 6, n).clip(15, 60).round(1),
        "DiabetesPedigreeFunction": rng.random(n).round(3),
        "Age": rng.integers(21, 80, n),
    })
    score = (data["Glucose"] - 125) / 20 + (data["BMI"] - 32) / 8 + rng.normal(0, 1, n)
    data["Outcome"] = (score > 0.3).astype(int)
    data.loc[rng.random(n) < 0.3, "Insulin"] = 0
    data.loc[rng.random(n) < 0.2, "SkinThickness"] = 0
    data.loc[rng.random(n) < 0.02, "Glucose"] = 0
    return data[FEATURES + ["Outcome"]]
