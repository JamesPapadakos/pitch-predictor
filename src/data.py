import pandas as pd
import json
from urllib.request import urlopen

from src.model import (
    AS_OF,
    DATA_DIR,
    DEFAULT_PITCHER_ID,
    FEATURES,
    PITCH_TYPES,
    TARGET,
)

CURRENT_RAW_NAME = "dylan_cease_2026_through_2026-10-27.csv"

SORT_COLUMNS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

PITCH_KEYS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

def pitcher_data_dir(pitcher_id, data_dir=DATA_DIR):
    if pitcher_id <= 0:
        raise ValueError("Pitcher ID must be a positive integer.")

    return data_dir / "pitchers" / str(pitcher_id)

def get_pitcher_name(pitcher_id, data_dir=DATA_DIR):
    folder = pitcher_data_dir(pitcher_id, data_dir)
    path = folder / "player.json"

    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"

        with urlopen(url, timeout=15) as response:
            payload = json.load(response)

        people = payload.get("people", [])

        if len(people) != 1:
            raise ValueError(
                f"Could not identify MLB player {pitcher_id}."
            )

        record = {
            "pitcher_id": people[0].get("id"),
            "pitcher_name": people[0].get("fullName"),
        }

    name = record.get("pitcher_name")

    if (
        record.get("pitcher_id") != pitcher_id
        or not isinstance(name, str)
        or not name.strip()
    ):
        raise ValueError(
            f"Invalid player metadata for pitcher {pitcher_id}."
        )

    # Cache only validated metadata so later training can work offline.
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

    return name.strip()

def validate_prepared(data):
    required = ["game_date", *FEATURES, TARGET]

    if data.empty or data[required].isna().any().any():
        raise ValueError(
            "Data is empty or has missing model inputs or labels."
        )

    if data.duplicated(PITCH_KEYS).any():
        raise ValueError("Duplicate pitches found.")

    unknown_types = set(data[TARGET]) - set(PITCH_TYPES)

    if unknown_types:
        raise ValueError(
            f"Unrecognized pitch types: {sorted(unknown_types)}"
        )


def prepare_pitches(raw, pitcher_id=DEFAULT_PITCHER_ID):
    pitches = raw.loc[raw["game_type"] == "R"].copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    if (
        pitches["pitcher"].isna().any()
        or not pitches["pitcher"].eq(pitcher_id).all()
    ):
        raise ValueError(
            f"Expected only pitches from pitcher {pitcher_id}."
        )

    if pitches.duplicated(PITCH_KEYS).any():
        raise ValueError("Duplicate pitch identifiers found.")

    pitches = pitches.sort_values(SORT_COLUMNS).copy()

    # Shift before dropping labels so an unknown pitch isn't skipped.
    previous = pitches.groupby(
        ["game_pk", "pitcher", "at_bat_number"]
    )[TARGET].shift(1)

    pitches["previous_pitch"] = previous.fillna("UNKNOWN")
    pitches.loc[
        pitches["pitch_number"] == 1, "previous_pitch"
    ] = "START"

    pitches["season"] = pitches["game_date"].dt.year

    columns = [
        "season",
        *SORT_COLUMNS,
        *FEATURES,
        TARGET,
    ]

    prepared = pitches.dropna(subset=[TARGET])[columns].copy()
    validate_prepared(prepared)

    return prepared


def load_training_data(
    data_dir=DATA_DIR,
    pitcher_id=DEFAULT_PITCHER_ID,
):
    folder = pitcher_data_dir(pitcher_id, data_dir)
    first_year = AS_OF.year - 4

    path = folder / f"prepared_{first_year}_{AS_OF.year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared data for pitcher {pitcher_id}. "
            f"Run python -m scripts.data --pitcher-id {pitcher_id} first."
        )

    data = pd.read_csv(path, parse_dates=["game_date"])

    # Check the contents as well as the folder to prevent pitcher mix-ups.
    if (
        "pitcher" not in data.columns
        or data["pitcher"].isna().any()
        or not data["pitcher"].eq(pitcher_id).all()
    ):
        raise ValueError(
            f"Prepared data must contain only pitcher {pitcher_id}. "
            "Regenerate it with the data command."
        )

    data = data[
        data["game_date"].dt.year.between(first_year, AS_OF.year)
        & (data["game_date"] < AS_OF)
    ].copy()

    data["season"] = data["game_date"].dt.year
    validate_prepared(data)

    # Five seasons is a maximum; some pitchers have a shorter MLB history.
    return data.sort_values(SORT_COLUMNS)