from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    file_path = DATA_DIR / "dylan_cease_2025.csv"

    if not file_path.exists():
        print("Run download_data.py first to download the pitches.")
        return

    pitches = pd.read_csv(file_path)

    print(f"Total pitches: {len(pitches):,}")
    print(f"Total games: {pitches['game_pk'].nunique()}")
    print(
        f"Date range: {pitches['game_date'].min()} "
        f"to {pitches['game_date'].max()}"
    )

    # Missing labels cannot teach the model which pitch was thrown.
    # Missing inputs may need different handling during preparation.
    important_columns = [
        "pitch_type",
        "batter",
        "stand",
        "balls",
        "strikes",
        "inning",
        "outs_when_up",
    ]

    print("\nMissing values in potential model columns:")
    print(pitches[important_columns].isna().sum())

    # Percentages let us compare counts even when sample sizes differ.
    pitch_percentages = pd.crosstab(
        index=[pitches["balls"], pitches["strikes"]],
        columns=pitches["pitch_name"],
        normalize="index",
    ) * 100

    print("\nPitch selection by count (%):")
    print(pitch_percentages.round(1).to_string())

    # A percentage based on very few pitches can be misleading.
    print("\nNumber of pitches at each count:")
    print(
        pitches.groupby(["balls", "strikes"])
        .size()
        .to_string(name="pitches")
    )


if __name__ == "__main__":
    main()