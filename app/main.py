from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import joblib
import numpy as np
from pathlib import Path
import shap
import logging
from datetime import datetime

app = FastAPI(title="Thyroid ML API", version="1.0.0")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/predictor", response_class=HTMLResponse)
async def predictor(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thyroid_api")


# Input Schema


class InputData(BaseModel):
    features: List[float] = Field(..., min_length=21, max_length=21)

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != 21:
            raise ValueError("Exactly 21 features required.")
        return v


class BatchInput(BaseModel):
    samples: List[InputData]



# Load Artifacts


BASE_DIR = Path(__file__).resolve().parent.parent

imputer = joblib.load(BASE_DIR / "artifacts/imputer.pkl")
scaler = joblib.load(BASE_DIR / "artifacts/scaler.pkl")
selector = joblib.load(BASE_DIR / "artifacts/selector.pkl")
model = joblib.load(BASE_DIR / "artifacts/model.pkl")

MODEL_VERSION = "v1.0.0"

LABEL_MAP = {
    1: "Normal",
    2: "Hyperthyroid",
    3: "Hypothyroid"
}


# SHAP Setup (XGBoost inside stack)


xgb_model = None
for name, estimator in model.named_estimators_.items():
    if name == "xgb":
        xgb_model = estimator

explainer = shap.TreeExplainer(xgb_model)

# Feature Names
feature_names = [
    "Age","Sex","On Thyroxine","Query on Thyroxine",
    "On Antithyroid Medication","Sick","Pregnant",
    "Thyroid Surgery","I131 Treatment","Query Hypothyroid",
    "Query Hyperthyroid","Lithium","Goitre","Tumor",
    "Hypopituitary","Psych","TSH","T3","TT4","T4U","FTI"
]

selected_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_indices]



# Core Prediction Logic


def run_inference(features: List[float]):

    # -----------------------------
    # Preprocess
    # -----------------------------
    X = np.array(features, dtype=float).reshape(1, -1)
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = selector.transform(X)

    # -----------------------------
    # Final Stacked Model Prediction
    # -----------------------------
    probs = model.predict_proba(X)[0]
    pred_index = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_index + 1]
    confidence = float(probs[pred_index])

    # -----------------------------
    # Base Model Predictions
    # -----------------------------
    base_model_outputs = {}

    for name, estimator in model.named_estimators_.items():
        try:
            if hasattr(estimator, "predict_proba"):
                base_probs = estimator.predict_proba(X)[0]
                base_pred = int(np.argmax(base_probs))
                base_model_outputs[name] = {
                    "prediction": LABEL_MAP[base_pred + 1],
                    "confidence": float(base_probs[base_pred])
                }
            else:
                base_pred = int(estimator.predict(X)[0]) - 1
                base_model_outputs[name] = {
                    "prediction": LABEL_MAP[base_pred + 1],
                    "confidence": None
                }
        except Exception:
            continue

    # -----------------------------
    # SHAP Explanation (Robust)
    # -----------------------------
    shap_values = explainer.shap_values(X)

    # Case 1: New SHAP returns 3D ndarray
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_for_class = shap_values[0, :, pred_index]

    # Case 2: Old SHAP returns list of arrays
    elif isinstance(shap_values, list):
        shap_for_class = shap_values[pred_index][0]

    else:
        shap_for_class = shap_values[0]

    feature_contributions = list(
        zip(selected_feature_names, shap_for_class)
    )

    feature_contributions = sorted(
        feature_contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    positive = [
        {"feature": n, "impact": float(v)}
        for n, v in feature_contributions if v > 0
    ]

    negative = [
        {"feature": n, "impact": float(v)}
        for n, v in feature_contributions if v < 0
    ]

    # -----------------------------
    # Risk Flag
    # -----------------------------
    risk_flag = "High Confidence" if confidence > 0.90 else "Moderate Confidence"

    logger.info({
        "prediction": pred_label,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {
        "prediction_code": pred_index + 1,
        "prediction_label": pred_label,
        "confidence": confidence,
        "risk_level": risk_flag,
        "probabilities": {
            "Normal": float(probs[0]),
            "Hyperthyroid": float(probs[1]),
            "Hypothyroid": float(probs[2])
        },
        "explanation": {
            "all_selected_features_sorted": [
                {"feature": n, "impact": float(v)}
                for n, v in feature_contributions
            ],
            "push_towards_prediction": positive,
            "push_against_prediction": negative
        },
        "base_model_predictions": base_model_outputs,
        "model_version": MODEL_VERSION
    }


# Endpoints


@app.post("/predict")
def predict(data: InputData):
    return run_inference(data.features)



@app.post("/predict-batch")
async def predict_batch(batch: BatchInput):
    results = [run_inference(sample.features) for sample in batch.samples]
    return {"results": results}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }