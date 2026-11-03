import argparse

from src.evaluation import inspect_predictions, main
from src.model import DEFAULT_PITCHER_ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitcher-id",
        type=int,
        default=DEFAULT_PITCHER_ID,
        help="MLB player ID. Defaults to Dylan Cease.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Inspect saved predictions without retraining.",
    )
    args = parser.parse_args()

    if args.pitcher_id <= 0:
        parser.error("--pitcher-id must be a positive integer.")

    if not args.diagnostics_only:
        main(pitcher_id=args.pitcher_id)

    inspect_predictions(pitcher_id=args.pitcher_id)