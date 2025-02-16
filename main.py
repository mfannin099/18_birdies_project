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
    return {"Handicap Predicton": "Returns low, mid, high"}

# Request Body Schema
class ModelInput(BaseModel):
    # course_name: str
    hole_yardage: float
    hole_par: float
    player_score: float

class ModelPrediction(BaseModel):
    predicted_handicap: str

@app.post("/predict", response_model=ModelPrediction)
def predict(data: ModelInput):
    # Convert the input data to a numpy array
    input_data = np.array(
        [[#data.course_name, 
        data.hole_yardage, 
        data.hole_par, 
        data.player_score]]
    )

    predicton = model.predict(input_data)
    prediction_str = str(predicton[0])

    return ModelPrediction(predicted_handicap=prediction_str)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

