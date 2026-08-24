# FastAPI service with two prediction endpoints:
#  - POST /predict/strokes: a trained LinearRegression over scoring-category
#    counts + course par (see train.py / data.py). Run api/train.py first to
#    produce api/models/strokes_model.pkl.
#  - POST /predict/random-walk: wraps score_analysis_app.utils.make_random_walk
#    directly -- no training involved, it's a per-request stochastic
#    simulation seeded by whatever score history you send in.
import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "score_analysis_app"))
from utils import make_random_walk  # noqa: E402

app = FastAPI(title="Golf Score Prediction API")

MODEL_PATH = Path(__file__).parent / "models" / "strokes_model.pkl"
_model_bundle = joblib.load(MODEL_PATH)
model = _model_bundle["model"]
FEATURES = _model_bundle["features"]


class StrokesPredictionInput(BaseModel):
    aces: int = Field(0, ge=0)
    albatrosses: int = Field(0, ge=0)
    eagles: int = Field(0, ge=0)
    birdies: int = Field(0, ge=0)
    pars: int = Field(0, ge=0)
    bogeys: int = Field(0, ge=0)
    double_bogey_or_worse: int = Field(0, ge=0)
    course_par: float


class StrokesPredictionOutput(BaseModel):
    predicted_total_strokes: float


@app.post("/predict/strokes", response_model=StrokesPredictionOutput)
def predict_strokes(payload: StrokesPredictionInput):
    row = pd.DataFrame([{
        "Aces": payload.aces,
        "Albatrosses": payload.albatrosses,
        "Eagles": payload.eagles,
        "Birdies": payload.birdies,
        "Pars": payload.pars,
        "Bogeys": payload.bogeys,
        "Double Bogey or Worse": payload.double_bogey_or_worse,
        "Course Par": payload.course_par,
    }])[FEATURES]

    prediction = model.predict(row)[0]
    return StrokesPredictionOutput(predicted_total_strokes=round(float(prediction), 1))


class RandomWalkInput(BaseModel):
    total_strokes_history: List[float] = Field(..., min_length=2)


class RandomWalkOutput(BaseModel):
    random_walk: List[float]


@app.post("/predict/random-walk", response_model=RandomWalkOutput)
def predict_random_walk(payload: RandomWalkInput):
    df = pd.DataFrame({"Total Strokes": payload.total_strokes_history})
    df = make_random_walk(df)
    return RandomWalkOutput(random_walk=[round(v, 1) for v in df["Random Walk"].tolist()])
