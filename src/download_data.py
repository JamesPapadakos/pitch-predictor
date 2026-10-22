from pathlib import Path

from pybaseball import statcast_pitcher


# Locate the data folder relative to this script.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    print("Downloading Dylan Cease's 2025 pitches...")

    pitches = statcast_pitcher(
        start_dt="2025-03-01",
        end_dt="2025-11-01",
        player_id=656302, #identifier for Dylan Cease
    )
    #Avoid empty download
    if pitches.empty:
        print("No pitches returned. No file saved.")
        return

    # Keep regular-season games and put pitches in order.
    pitches = pitches[pitches["game_type"] == "R"].copy()
    pitches = pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "dylan_cease_2025.csv"
    pitches.to_csv(output_path, index=False)

    print(f"Saved {len(pitches):,} pitches to {output_path}")

    print("\nFirst five pitches:")
    print(
        pitches[
            ["game_date", "pitch_name", "balls", "strikes", "inning", "batter"]
        ].head().to_string(index=False)
    )

    print("\nPitch counts:")
    print(pitches["pitch_name"].value_counts())


if __name__ == "__main__":
    main()