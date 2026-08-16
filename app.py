from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os

# Create FastAPI application
app = FastAPI(
    title="Mobile Network Quality Prediction API",
    description="ML-based mobile network quality prediction",
    version="1.0"
)

# Load trained ML files (checking both current folder and parent folder)
MODEL_PATH = "logistic_regression_model.pkl" if os.path.exists("logistic_regression_model.pkl") else "../ml_model/logistic_regression_model.pkl"
SCALER_PATH = "scaler.pkl" if os.path.exists("scaler.pkl") else "../ml_model/scaler.pkl"
FEATURES_PATH = "feature_columns.pkl" if os.path.exists("feature_columns.pkl") else "../ml_model/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)


# Input data schema (works whether app sends 5 or all fields)
class NetworkData(BaseModel):
    signal_strength: float
    download_speed: float
    upload_speed: float
    latency: float
    jitter: float
    network_type: Optional[str] = None
    carrier: Optional[str] = None
    band: Optional[str] = None
    congestion: Optional[str] = None
    ping: Optional[float] = None
    handover_count: Optional[float] = None
    dropped_connection: Optional[bool] = None


@app.get("/")
def home():
    return {
        "message": "Mobile Network Quality Prediction API is running"
    }


@app.post("/predict")
def predict(data: NetworkData):

    # Create input DataFrame with the 5 required features
    input_data = pd.DataFrame([{
        "Signal Strength (dBm)": data.signal_strength,
        "Download Speed (Mbps)": data.download_speed,
        "Upload Speed (Mbps)": data.upload_speed,
        "Latency (ms)": data.latency,
        "Jitter (ms)": data.jitter
    }])

    # Ensure column order matches the trained scaler & model
    input_data = input_data[feature_columns]

    # Scale the numerical input
    input_scaled = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(input_scaled)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    probability_dict = {
        str(classes[i]): round(float(probabilities[i]) * 100, 2)
        for i in range(len(classes))
    }

    return {
        "network_quality": prediction,
        "probabilities": probability_dict
    }
