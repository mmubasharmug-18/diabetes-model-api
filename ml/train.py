import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from app import config
from ml.features import FEATURES, zeros_to_nan

TARGET = "Outcome"


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("zeros", FunctionTransformer(zeros_to_nan)),
        ("fill", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
        )),
    ])


def train_model(df: pd.DataFrame, seed: int = 42):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    model = build_pipeline(seed).fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    trained_at = datetime.now(timezone.utc)
    info = {
        "version": "rf-" + trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "features": FEATURES,
        "metrics": {
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "roc_auc": round(roc_auc_score(y_test, prob), 3),
        },
    }
    return model, info


def save_model(model, info: dict, model_path: Path, info_path: Path) -> None:
    model_path, info_path = Path(model_path), Path(info_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    info_path.write_text(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the diabetes model and save it to models/")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="path to diabetes.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Could not find {data_path}. Download diabetes.csv from Kaggle and put it there.")

    model, info = train_model(pd.read_csv(data_path))
    save_model(model, info, config.MODEL_PATH, config.MODEL_INFO_PATH)

    print("model version:", info["version"])
    print("rows used    :", info["rows"])
    for name, value in info["metrics"].items():
        print(f"{name:10s}   : {value}")
    print("saved to     :", config.MODEL_PATH)


if __name__ == "__main__":
    main()
