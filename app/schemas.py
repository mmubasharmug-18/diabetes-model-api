from typing import Any, Optional

from pydantic import BaseModel, Field, StrictInt, field_validator

RULES = {
    "pregnancies": ("integer", 0, 20),
    "glucose": ("number", 40, 400),
    "blood_pressure": ("number", 20, 200),
    "skin_thickness": ("number", 0, 100),
    "insulin": ("number", 0, 900),
    "bmi": ("number", 10, 70),
    "diabetes_pedigree_function": ("number", 0, 3),
    "age": ("integer", 21, 120),
}


def bounds(name: str) -> dict:
    _, low, high = RULES[name]
    return {"ge": low, "le": high}


def rule_text(name: str) -> str:
    kind, low, high = RULES[name]
    article = "a whole number" if kind == "integer" else "a number"
    return f"{name} must be {article} between {low} and {high}"


class PredictRequest(BaseModel):
    pregnancies: StrictInt = Field(**bounds("pregnancies"), description="Number of pregnancies", examples=[6])
    glucose: float = Field(**bounds("glucose"), allow_inf_nan=False, description="Plasma glucose, mg/dL", examples=[148])
    blood_pressure: float = Field(**bounds("blood_pressure"), allow_inf_nan=False, description="Diastolic blood pressure, mm Hg", examples=[72])
    skin_thickness: float = Field(**bounds("skin_thickness"), allow_inf_nan=False, description="Skin fold thickness, mm (0 = not measured)", examples=[35])
    insulin: float = Field(**bounds("insulin"), allow_inf_nan=False, description="2-hour serum insulin, mu U/ml (0 = not measured)", examples=[0])
    bmi: float = Field(**bounds("bmi"), allow_inf_nan=False, description="Body mass index", examples=[33.6])
    diabetes_pedigree_function: float = Field(**bounds("diabetes_pedigree_function"), allow_inf_nan=False, description="Family history score", examples=[0.627])
    age: StrictInt = Field(**bounds("age"), description="Age in years", examples=[50])

    @field_validator(
        "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree_function",
        mode="before",
    )
    @classmethod
    def only_real_numbers(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value


class Probabilities(BaseModel):
    no_diabetes: float
    diabetes: float


class PredictResponse(BaseModel):
    request_id: str
    prediction: int
    label: str
    probability: Probabilities
    model_version: str


class LogEntry(BaseModel):
    id: int
    request_id: Optional[str] = None
    timestamp: str
    endpoint: str
    status: str
    http_status: int
    input: Optional[Any] = None
    prediction: Optional[int] = None
    probability: Optional[float] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    model_version: Optional[str] = None


class AuditSummary(BaseModel):
    total: int
    successes: int
    failures: int
    success_rate: float
    average_latency_ms: Optional[float] = None
    failures_by_status: dict[str, int]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: str
