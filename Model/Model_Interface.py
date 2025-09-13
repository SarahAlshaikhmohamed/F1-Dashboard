from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import datetime

# Load trained model
model = joblib.load("f1_model.pkl")

app = FastAPI()

# Define request body schema
class PredictionInput(BaseModel):
    continent: str
    team: str
    laps: int
    year: int

@app.get("/")
def home():
    return {"message": "F1 Prediction API is running!"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input to DataFrame
    data = pd.DataFrame([{
        "Continent": input_data.continent,
        "Team": input_data.team,
        "Laps": input_data.laps,
        "Year": input_data.year
    }])

    # Predict
    pred_seconds = model.predict(data)[0]
    pred_hhmmss = str(datetime.timedelta(seconds=int(pred_seconds)))

    return {
        "predicted_time_seconds": float(pred_seconds),
        "predicted_time_hhmmss": pred_hhmmss
    }