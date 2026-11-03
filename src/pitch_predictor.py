import joblib
import pandas as pd

from src.model import (
    DEFAULT_PITCHER_ID,
    MODELS_DIR,
    pitcher_model_path,
)


def predict_pitch_probabilities(
    situation,
    pitcher_id=DEFAULT_PITCHER_ID,
    ):
    path = pitcher_model_path(pitcher_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Saved model not found for pitcher {pitcher_id}. "
            f"Run python -m scripts.train --pitcher-id {pitcher_id} first."
        )

    bundle = joblib.load(path)

    #Prevent a mislabeled file from using another pitcher's model.
    if bundle["pitcher_id"] != pitcher_id:
        raise ValueError("Saved model pitcher ID does not match the request.")

    model = bundle["model"]

    #Preserve the feature order expected by the saved pipeline.
    inputs = pd.DataFrame([situation])[bundle["features"]]
    probabilities = model.predict_proba(inputs)[0]

    return sorted(
        zip(model.classes_, probabilities),
        key=lambda result: result[1],
        reverse=True,
    )

def list_available_pitchers():
    pitchers = []

    for path in sorted(MODELS_DIR.glob("pitcher_*_v2.joblib")):
        bundle = joblib.load(path)
        pitcher_id = bundle["pitcher_id"]

        # Ensure the file belongs to the pitcher described inside it.
        if path.name != f"pitcher_{pitcher_id}_v2.joblib":
            raise ValueError(f"Pitcher ID does not match model filename: {path.name}")

        pitchers.append({
            "pitcher_id": pitcher_id,
            "name": bundle["pitcher_name"],
            "pitch_types": bundle["pitch_types"],
            "training_start": bundle["training_start"],
            "training_end": bundle["training_end"],
        })

    return sorted(pitchers, key=lambda pitcher: pitcher["name"])