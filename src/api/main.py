from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import FraudPredictionRequest, FraudPredictionResponse
from src.features.engineer import engineer_features

app = FastAPI(
    title="Fraud Detection API",
    description="API for scoring transaction fraud risk using a trained XGBoost pipeline.",
    version="1.0.0",
)

MODEL_PATH = Path("models/fraud_model.pkl")
COLUMNS_PATH = Path("models/expected_columns.pkl")
HIGH_CARD_PATH = Path("models/high_card_cols.pkl")
FREQ_MAPS_PATH = Path("models/frequency_maps.pkl")

THRESHOLD = 0.85

pipeline = joblib.load(MODEL_PATH)
expected_columns = joblib.load(COLUMNS_PATH)
high_card_cols = joblib.load(HIGH_CARD_PATH)
frequency_maps = joblib.load(FREQ_MAPS_PATH)


def build_features(payload: FraudPredictionRequest) -> pd.DataFrame:
    """
    Build a single-row dataframe matching the schema used during training.
    High-cardinality categorical columns are frequency-encoded using
    training-time mappings. Missing columns are filled with NaN.
    """
    payload_dict = payload.model_dump()

    # Start with all expected columns as NaN
    row = {col: np.nan for col in expected_columns}

    # Fill provided request values where names match expected columns
    for key, value in payload_dict.items():
        if key in row:
            row[key] = value

    df = pd.DataFrame([row])

    # Apply shared feature engineering logic
    df = engineer_features(df)

    # Apply frequency encoding to high-cardinality columns
    for col in high_card_cols:
        if col in df.columns:
            freq_map = frequency_maps.get(col, {})
            df[col] = df[col].map(freq_map).fillna(0.0)

    # Reindex to exact training-time schema
    df = df.reindex(columns=expected_columns)

    return df


@app.get("/")
def root() -> dict:
    return {
        "message": "Fraud Detection API is running",
        "model": "XGBoost Pipeline",
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=FraudPredictionResponse)
def predict(payload: FraudPredictionRequest) -> FraudPredictionResponse:
    try:
        features = build_features(payload)

        fraud_probability = float(pipeline.predict_proba(features)[:, 1][0])
        fraud_prediction = int(fraud_probability >= THRESHOLD)
        decision = "review" if fraud_probability >= THRESHOLD else "approve"

        return FraudPredictionResponse(
            fraud_probability=round(fraud_probability, 6),
            fraud_prediction=fraud_prediction,
            threshold_used=THRESHOLD,
            decision=decision,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
