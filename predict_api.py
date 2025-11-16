"""FastAPI app for fraud prediction."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import Dict

app = FastAPI(title='Fraud Detection API')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

class Transaction(BaseModel):
    features: Dict[str, float]

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError('model.pkl or scaler.pkl not found. Train model first using src/train.py')
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@app.on_event('startup')
def startup_event():
    try:
        app.state.model, app.state.scaler = load_model_and_scaler()
    except FileNotFoundError:
        app.state.model, app.state.scaler = None, None

@app.post('/predict')
def predict(tx: Transaction):
    if app.state.model is None or app.state.scaler is None:
        raise HTTPException(status_code=503, detail='Model not available. Train and place model.pkl and scaler.pkl into src/')
    # Order features alphabetically to ensure consistent ordering
    feature_items = sorted(tx.features.items())
    feature_names, feature_vals = zip(*feature_items)
    arr = np.array(feature_vals, dtype=float).reshape(1, -1)
    arr_scaled = app.state.scaler.transform(arr)
    prob = app.state.model.predict_proba(arr_scaled)[0,1]
    return {'is_fraud_probability': float(prob), 'feature_order': list(feature_names)}
