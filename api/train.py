# Trains and pickles the Total Strokes regression model served by main.py.
# Run this whenever data.py's generator changes: `uv run python api/train.py`
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from data import CATEGORIES, generate_training_data

FEATURES = CATEGORIES + ["Course Par"]
MODEL_PATH = Path(__file__).parent / "models" / "strokes_model.pkl"


def train():
    df = generate_training_data()
    X = df[FEATURES]
    y = df["Total Strokes"]

    model = LinearRegression()
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

    print(f"R^2 on training data: {model.score(X, y):.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
