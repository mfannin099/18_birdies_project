# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Declaring FastAPI instance
app = FastAPI()
# Load the trained model
model = joblib.load('sim_golf_model.pkl')

@app.get("/")
def read_root():
    return {"Hello": "World TESTTTT"}

# Request Body Schema
class ModelInput(BaseModel):
    course_name: str
    hole_yardage: float
    hole_par: float
    player_score: float

class ModelPrediction(BaseModel):
    predicted_class_name: str

@app.post("/predict", response_model=ModelPrediction)
def predict(data: ModelInput):
    # Convert the input data to a numpy array
    input_data = np.array(
        [[data.course_name, 
        data.hole_yardage, 
        data.hole_par, 
        data.player_score]]
    )

    predicted_class_name = model.predict(input_data)

    return ModelPrediction(predicted_class_name=predicted_class_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



# @app.post("/") #/predict
# def predict(data: RequestBody):
#     # Convert input to DataFrame
#     test_data = pd.DataFrame([data.dict()])

#     # Make a prediction
#     prediction = model.predict(test_data)[0]

#     # Return the prediction result
#     return {"predicted_class": prediction}




# @app.post('/predict')
# def predict(data: RequestBody):
#     # Convert input into a DataFrame
#     test_data = pd.DataFrame([data.dict()])  

#     # Predicting the class
#     prediction = model.predict(test_data)[0]
    
#     # Return the prediction result
#     return {"predicted_class": prediction}
