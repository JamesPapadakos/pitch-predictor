from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "dylan_cease_v2.joblib"
)


def predict_pitch_probabilities(situation):
    """Return pitch codes and probabilities for a game situation."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python src/train_model.py to create it."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    # Match the feature names and order used when the model was trained.
    inputs = pd.DataFrame([situation])[bundle["features"]]
    probabilities = model.predict_proba(inputs)[0]

    return sorted(
        zip(model.classes_, probabilities),
        key=lambda result: result[1],
        reverse=True,
    )