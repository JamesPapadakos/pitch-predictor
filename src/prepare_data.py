from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    pitches = pd.read_csv(DATA_DIR / "dylan_cease_2025.csv")
    original_count = len(pitches)

    pitches = pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    # Stay within each plate appearance so sequences don't cross batters.
    previous_pitch = pitches.groupby(
        ["game_pk", "at_bat_number"]
    )["pitch_type"].shift(1)

    # Distinguish a new plate appearance from an unrecorded pitch type.
    pitches["previous_pitch"] = previous_pitch.fillna("UNKNOWN")
    pitches.loc[pitches["pitch_number"] == 1, "previous_pitch"] = "START"

    # Build sequences before removing unlabeled rows; otherwise "previous"
    # could accidentally refer to a pitch two positions earlier.
    pitches = pitches.dropna(subset=["pitch_type"]).copy()

    # Base columns contain runner IDs; occupancy is enough for this first model.
    for base in ["1b", "2b", "3b"]:
        pitches[f"runner_on_{base}"] = pitches[f"on_{base}"].notna().astype(int)

    # Use the pitching team's perspective so the sign has a consistent meaning.
    # Pre-pitch scores avoid revealing what happens on the pitch being predicted.
    pitches["score_difference"] = pitches["fld_score"] - pitches["bat_score"]

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
        "previous_pitch",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "score_difference",
        "pitch_type",
    ]

    model_data = pitches[columns].copy()
    output_path = DATA_DIR / "dylan_cease_2025_prepared.csv"
    model_data.to_csv(output_path, index=False)

    print(f"Removed {original_count - len(model_data)} unlabeled records.")
    print(f"Saved {len(model_data):,} labeled pitches.")

    print("\nSequence preview:")
    print(
        model_data[
            ["balls", "strikes", "previous_pitch", "pitch_type"]
        ].head(12).to_string(index=False)
    )

if __name__ == "__main__":
    main()