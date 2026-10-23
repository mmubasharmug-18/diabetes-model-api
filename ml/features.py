import numpy as np
import pandas as pd

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

API_TO_COLUMN = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "age": "Age",
}


def zeros_to_nan(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame
