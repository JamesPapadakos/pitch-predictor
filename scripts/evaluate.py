import argparse

from src.evaluation import inspect_predictions, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Inspect saved predictions without retraining.",
    )
    args = parser.parse_args()

    if not args.diagnostics_only:
        main()

    inspect_predictions()