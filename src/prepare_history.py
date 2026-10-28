from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEASONS = range(2021, 2026)


def main():
    seasons = []

    for year in SEASONS:
        file_path = DATA_DIR / f"dylan_cease_{year}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {file_path.name}. Run download_history.py first."
            )

        pitches = pd.read_csv(file_path)
        pitches = pitches[pitches["game_type"] == "R"].copy()
        pitches["season"] = year
        seasons.append(pitches)

    history = pd.concat(seasons, ignore_index=True)

    history = history.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    # Duplicate pitch records would give some examples extra influence.
    pitch_identifier = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitcher",
    ]

    if history.duplicated(subset=pitch_identifier).any():
        raise ValueError(
            "Duplicate pitch identifiers found. Inspect them before training."
        )

    # Keep sequences within the same pitcher and plate appearance.
    previous_pitch = history.groupby(
        ["game_pk", "pitcher", "at_bat_number"]
    )["pitch_type"].shift(1)

    history["previous_pitch"] = previous_pitch.fillna("UNKNOWN")
    history.loc[
        history["pitch_number"] == 1, "previous_pitch"
    ] = "START"

    # Removing missing labels earlier could skip the true previous pitch.
    original_count = len(history)
    history = history.dropna(subset=["pitch_type"]).copy()

    # Retain identifiers for chronological splits and later diagnostics.
    columns = [
        "season",
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "balls",
        "strikes",
        "stand",
        "inning",
        "outs_when_up",
        "previous_pitch",
        "pitch_type",
    ]

    prepared = history[columns].copy()

    features = [
        "balls",
        "strikes",
        "stand",
        "inning",
        "outs_when_up",
        "previous_pitch",
    ]

    # Stop rather than silently train on incomplete model inputs.
    missing = prepared[features].isna().sum()
    if missing.any():
        raise ValueError(
            f"Missing feature values:\n{missing[missing > 0]}"
        )

    output_path = DATA_DIR / "dylan_cease_2021_2025_prepared.csv"
    prepared.to_csv(output_path, index=False)

    print(f"Removed {original_count - len(prepared)} unlabeled records.")
    print(f"Saved {len(prepared):,} labeled pitches to {output_path.name}")

    print("\nPrepared data by season:")
    print(
        prepared.groupby("season").agg(
            pitches=("pitch_type", "size"),
            games=("game_pk", "nunique"),
        ).to_string()
    )

    print("\nPitch types included:")
    print(prepared["pitch_type"].value_counts().to_string())

    print("\nFirst five rows:")
    print(prepared.head().to_string(index=False))


if __name__ == "__main__":
    main()