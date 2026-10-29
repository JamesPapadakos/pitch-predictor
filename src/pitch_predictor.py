import joblib
import pandas as pd

from src.model import MODEL_PATH


def predict_pitch_probabilities(situation):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python -m scripts.train first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    # Preserve the feature order expected by the saved pipeline.
    inputs = pd.DataFrame([situation])[bundle["features"]]
    probabilities = model.predict_proba(inputs)[0]

    return sorted(
        zip(model.classes_, probabilities),
        key=lambda result: result[1],
        reverse=True,
    )