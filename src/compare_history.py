import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from validate_model import build_model


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURES = [
    "balls",
    "strikes",
    "stand",
    "inning",
    "outs_when_up",
    "previous_pitch",
]

PITCH_TYPES = ["CH", "FC", "FF", "KC", "SI", "SL", "ST"]

EXPERIMENTS = [
    ("One season", 1, False),
    ("Three seasons", 3, False),
    ("Five seasons", 5, False),
    ("Five seasons weighted", 5, True),
]


FOREST_SETTINGS = [
    ("Current: depth 6, leaf 20", 6, 20),
    ("Depth 10, leaf 20", 10, 20),
    ("Depth 6, leaf 5", 6, 5),
    ("Depth 10, leaf 5", 10, 5),
]


def main(forest_settings=False):
    # Hold data and recency weighting fixed to isolate forest settings.
    experiments = (
        [(name, 5, True, depth, leaf) for name, depth, leaf in FOREST_SETTINGS]
        if forest_settings
        else [(name, years, weighted, 6, 20)
              for name, years, weighted in EXPERIMENTS]
    )
    data = pd.read_csv(
        DATA_DIR / "dylan_cease_2021_2025_prepared.csv",
        parse_dates=["game_date"],
    )
    data = data.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    unknown_types = set(data["pitch_type"]) - set(PITCH_TYPES)
    if unknown_types:
        raise ValueError(f"Unrecognized pitch types: {unknown_types}")

    games_2025 = (
        data.loc[data["season"] == 2025, "game_pk"]
        .drop_duplicates()
        .to_list()
    )

    if len(games_2025) != 32:
        raise ValueError("These windows expect 32 games in 2025.")

    results = []

    for round_number, train_end in enumerate([10, 15, 20], start=1):
        validation_games = games_2025[train_end:train_end + 5]
        validation = data[data["game_pk"].isin(validation_games)]
        cutoff = validation["game_date"].min()

        print(
            f"\nRound {round_number}: "
            f"validate 2025 games {train_end + 1}–{train_end + 5}"
        )

        for name, season_count, weighted, depth, leaf in experiments:
            first_season = 2025 - season_count + 1

            # Every option sees only pitches available before this window.
            training = data[
                (data["season"] >= first_season)
                & (data["game_date"] < cutoff)
            ]

            # Rebuild preprocessing and the forest for each experiment.
            model = build_model()
            model.set_params(
                classifier__max_depth=depth,
                classifier__min_samples_leaf=leaf,
            )
            fit_options = {}

            if weighted:
                age_days = (
                    cutoff - training["game_date"]
                ).dt.total_seconds() / 86400

                # One-year half-life: a pitch 365 days old has half the weight.
                weights = 0.5 ** (age_days / 365)

                # Keep average weight at one while preserving relative recency.
                weights = weights / weights.mean()
                fit_options["classifier__sample_weight"] = weights.to_numpy()

            model.fit(
                training[FEATURES],
                training["pitch_type"],
                **fit_options,
            )

            predictions = model.predict(validation[FEATURES])

            # Some training windows may lack a pitch type seen in another year.
            probabilities = pd.DataFrame(
                model.predict_proba(validation[FEATURES]),
                columns=model.classes_,
            ).reindex(columns=PITCH_TYPES, fill_value=0)

            # Use the same floor for every option, including unseen classes.
            probabilities = probabilities.clip(lower=1e-12)
            probabilities = probabilities.div(
                probabilities.sum(axis=1), axis=0
            )

            accuracy = accuracy_score(
                validation["pitch_type"], predictions
            )
            loss = log_loss(
                validation["pitch_type"],
                probabilities.to_numpy(),
                labels=PITCH_TYPES,
            )

            results.append({
                "round": round_number,
                "experiment": name,
                "training_pitches": len(training),
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "validation_pitches": len(validation),
                "accuracy": accuracy,
                "log_loss": loss,
            })

            print(
                f"{name}: "
                f"{len(training):,} training pitches | "
                f"accuracy {accuracy:.1%} | log loss {loss:.3f}"
            )

    results = pd.DataFrame(results)

    print("\nCombined validation results:")
    for name, group in results.groupby("experiment", sort=False):
        # Pool by pitch count because windows contain different pitch totals.
        weights = (
            group["validation_pitches"]
            / group["validation_pitches"].sum()
        )

        accuracy = (group["accuracy"] * weights).sum()
        loss = (group["log_loss"] * weights).sum()

        print(f"{name}: accuracy {accuracy:.1%} | log loss {loss:.3f}")

    filename = (
        "forest_settings_comparison.csv" if forest_settings
        else "history_comparison.csv"
    )
    results.to_csv(DATA_DIR / filename, index=False)
    print(f"\nSaved experiment results to data/{filename}")
    print("No 2026 data used. Saved application model unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronological model comparisons.")
    parser.add_argument(
        "--forest-settings",
        action="store_true",
        help="Compare forest settings using five seasons with recency weighting.",
    )
    args = parser.parse_args()
    main(forest_settings=args.forest_settings)