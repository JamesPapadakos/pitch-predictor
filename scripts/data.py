import pandas as pd

from src.data import (
    CURRENT_RAW_NAME,
    PREPARED_NAME,
    prepare_pitches,
)
from src.model import AS_OF, DATA_DIR


def main():
    # Downloading is only needed by this command, not the running app.
    from pybaseball import statcast_pitcher

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seasons = []

    for year in range(AS_OF.year - 4, AS_OF.year + 1):
        if year == AS_OF.year:
            filename = CURRENT_RAW_NAME
            end_date = (
                AS_OF - pd.Timedelta(days=1)
            ).date().isoformat()
        else:
            filename = f"dylan_cease_{year}.csv"
            end_date = f"{year}-11-01"

        path = DATA_DIR / filename

        # Reuse the same source files behind our earlier results.
        if path.exists():
            print(f"Using saved data for {year}")
            raw = pd.read_csv(path)
        else:
            print(f"Downloading {year}...")

            raw = statcast_pitcher(
                start_dt=f"{year}-03-01",
                end_dt=end_date,
                player_id=656302,
            )

            if raw.empty:
                raise ValueError(f"No data returned for {year}.")

            raw = raw[raw["game_type"] == "R"].copy()

            if raw.empty:
                raise ValueError(
                    f"No regular-season data returned for {year}."
                )

            raw.to_csv(path, index=False)

        seasons.append(raw)

    raw = pd.concat(seasons, ignore_index=True)
    prepared = prepare_pitches(raw)

    prepared = prepared[
        prepared["game_date"].dt.year.between(
            AS_OF.year - 4, AS_OF.year
        )
        & (prepared["game_date"] < AS_OF)
    ].copy()

    expected_seasons = set(
        range(AS_OF.year - 4, AS_OF.year + 1)
    )

    if set(prepared["season"]) != expected_seasons:
        raise ValueError(
            "Expected regular-season data from 2022 through 2026."
        )

    output_path = DATA_DIR / PREPARED_NAME
    prepared.to_csv(output_path, index=False)

    summary = prepared.groupby("season").agg(
        pitches=("pitch_type", "size"),
        games=("game_pk", "nunique"),
    )

    print("\nPrepared data by season:")
    print(summary.to_string())
    print(
        f"\nSaved {len(prepared):,} labeled pitches "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()