from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    pitches = pd.read_csv(DATA_DIR / "dylan_cease_2025.csv")
    original_count = len(pitches)

    # Without the actual pitch type, a row has no training answer.
    pitches = pitches.dropna(subset=["pitch_type"]).copy()

    pitches = pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    # Keep dates and game IDs so we can later test on newer games
    # without splitting one game's pitches across training and testing.
    columns = [
        "game_date",
        "game_pk",
        "balls",
        "strikes",
        "stand",
        "inning",
        "outs_when_up",
        "pitch_type",
    ]

    model_data = pitches[columns].copy()
    output_path = DATA_DIR / "dylan_cease_2025_prepared.csv"
    model_data.to_csv(output_path, index=False)

    print(f"Removed {original_count - len(model_data)} unlabeled records.")
    print(f"Saved {len(model_data):,} labeled pitches.")
    print("\nFirst five rows:")
    print(model_data.head().to_string(index=False))

    print("\nTarget pitch types:")
    print(model_data["pitch_type"].value_counts())


if __name__ == "__main__":
    main()