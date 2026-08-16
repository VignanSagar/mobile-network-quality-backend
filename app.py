from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Mobile Network Quality Prediction API",
    description="ML-based mobile network quality prediction",
    version="1.0"
)

# Load trained ML files (Updated paths for Cloud Deployment)
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# Input data received from Android app
class NetworkData(BaseModel):
    signal_strength: float
    download_speed: float
    upload_speed: float
    latency: float
    jitter: float
    network_type: str
    carrier: str
    band: str
    congestion: str
    ping: float
    handover_count: float
    dropped_connection: bool


@app.get("/")
def home():
    return {
        "message": "Mobile Network Quality Prediction API is running"
    }


@app.post("/predict")
def predict(data: NetworkData):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Signal Strength (dBm)": data.signal_strength,
        "Download Speed (Mbps)": data.download_speed,
        "Upload Speed (Mbps)": data.upload_speed,
        "Latency (ms)": data.latency,
        "Jitter (ms)": data.jitter,
        "Network Type": data.network_type,
        "Carrier": data.carrier,
        "Band": data.band,
        "Network Congestion Level": data.congestion,
        "Ping to Google (ms)": data.ping,
        "Handover Count": data.handover_count,
        "Dropped Connection": data.dropped_connection
    }])

    # Convert categorical values to encoded columns
    input_encoded = pd.get_dummies(input_data)

    # Make sure input has exactly the same columns as training data
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Prediction probabilities
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
