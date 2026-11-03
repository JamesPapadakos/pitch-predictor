import argparse

from src.data import get_pitcher_name, load_training_data
from src.model import (
    DEFAULT_PITCHER_ID,
    fit_model,
    pitcher_model_path,
    save_model,
)


def main(pitcher_id=DEFAULT_PITCHER_ID):
    data = load_training_data(pitcher_id=pitcher_id)
    pitcher_name = get_pitcher_name(pitcher_id)

    if data["pitch_type"].nunique() < 2:
        raise ValueError(
            "Training requires at least two distinct pitch types."
        )

    model = fit_model(data)

    save_model(
        model,
        data,
        pitcher_name=pitcher_name,
        pitcher_id=pitcher_id,
    )

    summary = data.groupby("season").agg(
        pitches=("pitch_type", "size"),
        games=("game_pk", "nunique"),
    )

    print(f"Pitcher: {pitcher_name} (ID {pitcher_id})")
    print(summary.to_string())
    print(f"\nTraining pitches: {len(data):,}")
    print(
        f"Training dates: "
        f"{data['game_date'].min().date()} to "
        f"{data['game_date'].max().date()}"
    )
    print(f"Saved model: {pitcher_model_path(pitcher_id)}")
    print("Deployment fit completed; this is not an accuracy evaluation.")


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