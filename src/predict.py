from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "dylan_cease_v1.joblib"
)

PITCH_NAMES = {
    "CH": "Changeup",
    "FF": "Four-seam fastball",
    "KC": "Knuckle curve",
    "SI": "Sinker",
    "SL": "Slider",
    "ST": "Sweeper",
}


def main():
    if not MODEL_PATH.exists():
        print("Run test_model.py first to save the model.")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    situation = {
        "balls": 1,
        "strikes": 2,
        "stand": "L",
        "inning": 6,
        "outs_when_up": 1,
        "previous_pitch": "FF",
    }

    # Use the saved feature order to match the model's training inputs.
    inputs = pd.DataFrame([situation])[bundle["features"]]
    probabilities = model.predict_proba(inputs)[0]

    # Match probability columns to the model's own class labels.
    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda result: result[1],
        reverse=True,
    )

    print("Dylan Cease — estimated next-pitch probabilities")
    print(
        f"Count: {situation['balls']}-{situation['strikes']} | "
        f"Batter: {situation['stand']} | "
        f"Inning: {situation['inning']} | "
        f"Outs: {situation['outs_when_up']}"
    )
    print(f"Previous pitch: {situation['previous_pitch']}\n")

    for pitch_type, probability in results:
        name = PITCH_NAMES.get(pitch_type, pitch_type)
        print(f"{name}: {probability:.1%}")


if __name__ == "__main__":
    main()