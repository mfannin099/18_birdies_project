# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Declaring FastAPI instance
app = FastAPI(title="Golf Handicap Prediction")
# Load the trained model
model = joblib.load('sim_golf_model_NO_STR.pkl')

@app.get("/")
def read_root():
    return {"Handicap Predicton": "Returns low,mid,high"}

# Request Body Schema
class ModelInput(BaseModel):
    # course_name: str
    hole_yardage: float
    hole_par: float
    player_score: float

class ModelPrediction(BaseModel):
    predicted_handicap: str

@app.get("/predict")  # Allowing GET requests with query parameters
def predict(hole_yardage: float, hole_par: float, player_score: float):
    # Convert input data to a NumPy array
    input_data = np.array([[hole_yardage, hole_par, player_score]])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_str = str(prediction[0])

    return {"predicted_handicap": prediction_str}

## Example url: http://127.0.0.1:8000/predict?hole_yardage=7000&hole_par=72&player_score=90

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

