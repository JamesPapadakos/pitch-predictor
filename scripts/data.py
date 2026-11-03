import argparse

import pandas as pd

from src.data import CURRENT_RAW_NAME, pitcher_data_dir, prepare_pitches
from src.model import AS_OF, DATA_DIR, DEFAULT_PITCHER_ID


def load_season(pitcher_id, year, end_date, folder):
    path = folder / f"raw_{year}_through_{end_date}.csv"

    if path.exists():
        print(f"Using saved data for {year}")
        return pd.read_csv(path)

    # Reuse Cease's existing downloads during the folder migration.
    if pitcher_id == DEFAULT_PITCHER_ID:
        legacy_name = (
            CURRENT_RAW_NAME
            if year == AS_OF.year
            else f"dylan_cease_{year}.csv"
        )
        legacy_path = DATA_DIR / legacy_name

        if legacy_path.exists():
            print(f"Using existing Cease data for {year}")
            return pd.read_csv(legacy_path)

    from pybaseball import statcast_pitcher

    print(f"Downloading {year} for pitcher {pitcher_id}...")

    raw = statcast_pitcher(
        start_dt=f"{year}-03-01",
        end_dt=end_date,
        player_id=pitcher_id,
    )

    if raw.empty:
        print(f"No pitches returned for {year}.")
        return None

    raw = raw[raw["game_type"] == "R"].copy()

    if raw.empty:
        print(f"No regular-season pitches for {year}.")
        return None

    raw.to_csv(path, index=False)
    return raw


def main(pitcher_id=DEFAULT_PITCHER_ID):
    folder = pitcher_data_dir(pitcher_id)
    folder.mkdir(parents=True, exist_ok=True)

    first_year = AS_OF.year - 4
    seasons = []

    for year in range(first_year, AS_OF.year + 1):
        end_date = (
            (AS_OF - pd.Timedelta(days=1)).date().isoformat()
            if year == AS_OF.year
            else f"{year}-11-01"
        )

        raw = load_season(
            pitcher_id,
            year,
            end_date,
            folder,
        )

        # Rookies and injured pitchers may not have data in every season.
        if raw is not None:
            seasons.append(raw)

    if not seasons:
        raise ValueError(
            f"No regular-season data found for pitcher {pitcher_id}."
        )

    raw = pd.concat(seasons, ignore_index=True)
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    raw = raw[
        raw["game_date"].dt.year.between(first_year, AS_OF.year)
        & (raw["game_date"] < AS_OF)
    ].copy()

    prepared = prepare_pitches(raw, pitcher_id=pitcher_id)

    output_path = (
        folder / f"prepared_{first_year}_{AS_OF.year}.csv"
    )
    prepared["pitcher"] = pitcher_id
    prepared.to_csv(output_path, index=False)

    summary = prepared.groupby("season").agg(
        pitches=("pitch_type", "size"),
        games=("game_pk", "nunique"),
    )

    print(f"\nPitcher ID: {pitcher_id}")
    print(summary.to_string())
    print(f"\nSaved {len(prepared):,} labeled pitches to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitcher-id",
        type=int,
        default=DEFAULT_PITCHER_ID,
        help="MLB player ID. Defaults to Dylan Cease.",
    )
    args = parser.parse_args()

    if args.pitcher_id <= 0:
        parser.error("--pitcher-id must be a positive integer.")

    main(pitcher_id=args.pitcher_id)