import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.joblib"))
MODEL_INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", BASE_DIR / "models" / "model_info.json"))
AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", BASE_DIR / "logs" / "audit.db"))
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "diabetes.csv"))
