import pandas as pd

from src.model import AS_OF, DATA_DIR, FEATURES, PITCH_TYPES, TARGET


PREPARED_NAME = "dylan_cease_2022_2026_prepared.csv"
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


def prepare_pitches(raw):
    pitches = raw.loc[raw["game_type"] == "R"].copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    if (
        pitches["pitcher"].isna().any()
        or not pitches["pitcher"].eq(656302).all()
    ):
        raise ValueError(
            "Expected only Dylan Cease's pitches (MLB ID 656302)."
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


def load_training_data(data_dir=DATA_DIR):
    prepared_path = data_dir / PREPARED_NAME

    if prepared_path.exists():
        data = pd.read_csv(
            prepared_path,
            parse_dates=["game_date"],
        )
    else:
        # Support existing downloads while moving to one prepared snapshot.
        history_path = (
            data_dir / "dylan_cease_2021_2025_prepared.csv"
        )
        current_path = data_dir / CURRENT_RAW_NAME

        if not history_path.exists() or not current_path.exists():
            raise FileNotFoundError(
                "Missing training data. Expected the prepared snapshot "
                "or the existing historical and 2026 files."
            )

        history = pd.read_csv(
            history_path,
            parse_dates=["game_date"],
        )
        current = prepare_pitches(pd.read_csv(current_path))

        data = pd.concat(
            [history, current],
            ignore_index=True,
        )

    # Count the prediction year as one of the five calendar seasons.
    data = data[
        data["game_date"].dt.year.between(
            AS_OF.year - 4, AS_OF.year
        )
        & (data["game_date"] < AS_OF)
    ].copy()

    data["season"] = data["game_date"].dt.year

    validate_prepared(data)

    expected_seasons = set(range(AS_OF.year - 4, AS_OF.year + 1))

    if set(data["season"]) != expected_seasons:
        raise ValueError(
            "Expected data from every season, 2022 through 2026."
        )

    return data.sort_values(SORT_COLUMNS)