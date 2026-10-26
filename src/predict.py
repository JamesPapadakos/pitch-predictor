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

def get_number(prompt, minimum, maximum=None):
    while True:
        response = input(prompt).strip()

        try:
            number = int(response)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if number < minimum or (maximum is not None and number > maximum):
            if maximum is None:
                print(f"Please enter a number of at least {minimum}.")
            else:
                print(f"Please enter a number from {minimum} to {maximum}.")
            continue

        return number

def get_choice(prompt, choices):
    while True:
        response = input(prompt).strip().upper()

        if response in choices:
            return response

        print(f"Please enter one of: {', '.join(choices)}")

def main():
    if not MODEL_PATH.exists():
        print("Run test_model.py first to save the model.")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    print("Enter the current game situation:")

    balls = get_number("Balls (0–3): ", 0, 3)
    strikes = get_number("Strikes (0–2): ", 0, 2)

    stand = get_choice(
        "Batter's side (L or R): ",
        ["L", "R"],
    )

    inning = get_number("Inning (1 or higher): ", 1)
    outs = get_number("Outs (0–2): ", 0, 2)

    if balls == 0 and strikes == 0:
        # A 0–0 count normally starts a new plate appearance.
        previous_pitch = "START"
        print("Previous pitch: START")
    else:
        print("\nPrevious pitch codes:")
        for code, name in PITCH_NAMES.items():
            print(f"  {code}: {name}")
        print("  UNKNOWN: pitch type isn't known")

        previous_pitch = get_choice(
            "Previous pitch: ",
            list(PITCH_NAMES) + ["UNKNOWN"],
        )

    situation = {
        "balls": balls,
        "strikes": strikes,
        "stand": stand,
        "inning": inning,
        "outs_when_up": outs,
        "previous_pitch": previous_pitch,
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