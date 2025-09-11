# Imports
from fastapi import FastAPI
import joblib
import pandas as pd
import datetime

# Load trained model
model = joblib.load("f1_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "F1 Prediction API is running!"}

# Prediction function
@app.post("/predict")
def predict():
    data = pd.DataFrame([{
        "Continent": "Europe",
        "Team": "Ferrari",
        "Laps": 70,
        "Year": 2026
    }])

    # Predict
    pred_seconds = model.predict(data)[0]
    pred_hhmmss = str(datetime.timedelta(seconds=int(pred_seconds)))

    return {
        "predicted_time_seconds": float(pred_seconds),
        "predicted_time_hhmmss": pred_hhmmss
    }
