from pathlib import Path

import pandas as pd
from pybaseball import statcast_pitcher


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PITCHER_ID = 656302
SEASONS = range(2021, 2026)


def load_season(year):
    file_path = DATA_DIR / f"dylan_cease_{year}.csv"

    # Reuse downloads, including our existing 2025 file, to avoid
    # repeated requests and preserve the data behind earlier results.
    if file_path.exists():
        print(f"Using saved data for {year}")
        return pd.read_csv(file_path)

    print(f"Downloading {year}...")

    pitches = statcast_pitcher(
        start_dt=f"{year}-03-01",
        end_dt=f"{year}-11-01",
        player_id=PITCHER_ID,
    )

    if pitches.empty:
        raise ValueError(f"No data returned for {year}. No file saved.")

    pitches = pitches[pitches["game_type"] == "R"].copy()

    if pitches.empty:
        raise ValueError(f"No regular-season records found for {year}.")

    pitches = pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    # Retain the original columns for future feature experiments.
    pitches.to_csv(file_path, index=False)

    print(f"Saved {len(pitches):,} records for {year}")
    return pitches


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seasons = []

    for year in SEASONS:
        pitches = load_season(year).copy()
        pitches["season"] = year
        seasons.append(pitches)

    history = pd.concat(seasons, ignore_index=True)

    summary = history.groupby("season").agg(
        records=("game_pk", "size"),
        games=("game_pk", "nunique"),
        labeled_pitches=("pitch_type", "count"),
    )

    print("\nData available by season:")
    print(summary.to_string())

    # Compare proportions because seasons can contain different pitch totals.
    labeled = history.dropna(subset=["pitch_type"])

    usage = pd.crosstab(
        labeled["season"],
        labeled["pitch_type"],
        normalize="index",
    ) * 100

    print("\nPitch usage by season (% of labeled pitches):")
    print(usage.round(1).to_string())

    # Percentages alone can hide very small samples.
    three_zero = labeled[
        (labeled["balls"] == 3) & (labeled["strikes"] == 0)
    ]

    print("\nActual pitch counts at 3–0, by season:")
    if three_zero.empty:
        print("No labeled 3–0 pitches found.")
    else:
        print(
            pd.crosstab(
                three_zero["season"],
                three_zero["pitch_type"],
            ).to_string()
        )


if __name__ == "__main__":
    main()