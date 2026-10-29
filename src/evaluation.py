import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

from src.data import load_training_data
from src.model import (
    DATA_DIR,
    FEATURES,
    PITCH_TYPES,
    fit_model,
    recency_weights,
)


def forest_probabilities(model, pitches):
    # Keep probability columns consistent when training lacks a pitch type.
    return pd.DataFrame(
        model.predict_proba(pitches[FEATURES]),
        columns=model.classes_,
    ).reindex(columns=PITCH_TYPES, fill_value=0)


def situational_probabilities(training, pitches, cutoff):
    training = training.copy()

    # Give the baseline the same recency information as the forest.
    training["weight"] = recency_weights(
        training, cutoff
    ).to_numpy()

    overall = (
        training.groupby("pitch_type")["weight"]
        .sum()
        .reindex(PITCH_TYPES, fill_value=0)
    )
    overall = overall / overall.sum()

    situation_columns = ["balls", "strikes", "stand"]
    lookup = {}

    for situation, group in training.groupby(situation_columns):
        counts = (
            group.groupby("pitch_type")["weight"]
            .sum()
            .reindex(PITCH_TYPES, fill_value=0)
        )

        # Small groups borrow 10 pitches' worth of overall frequencies.
        lookup[situation] = (
            (counts + 10 * overall) / (counts.sum() + 10)
        ).to_numpy()

    return pd.DataFrame(
        [
            lookup.get(situation, overall.to_numpy())
            for situation in pitches[situation_columns].itertuples(
                index=False,
                name=None,
            )
        ],
        columns=PITCH_TYPES,
    )


def score(actual, probabilities):
    probabilities = probabilities[PITCH_TYPES]
    predictions = probabilities.idxmax(axis=1).to_numpy()

    # Apply the same floor to every predictor, including unseen classes.
    probabilities = probabilities.clip(lower=1e-12)
    probabilities = probabilities.div(
        probabilities.sum(axis=1),
        axis=0,
    )

    return {
        "accuracy": accuracy_score(actual, predictions),
        "log_loss": log_loss(
            actual,
            probabilities.to_numpy(),
            labels=PITCH_TYPES,
        ),
    }

def main():
    data = load_training_data()
    history = data[data["season"] < 2026].copy()
    current = data[data["season"] == 2026].copy()

    frozen_cutoff = pd.Timestamp("2026-01-01")
    frozen = fit_model(history, frozen_cutoff)

    collected = {
        "Frozen forest": [],
        "Updated forest": [],
        "Frozen situational baseline": [],
        "Updated situational baseline": [],
    }

    actual = []
    per_game = []
    pitch_results = []

    games = current["game_pk"].drop_duplicates().to_list()

    for number, game_id in enumerate(games, start=1):
        game = current[current["game_pk"] == game_id]
        cutoff = game["game_date"].min()

        # Exclude the entire prediction date, including the current game.
        training = data[data["game_date"] < cutoff].copy()
        updated = fit_model(training, cutoff)

        predictors = {
            "Frozen forest": forest_probabilities(frozen, game),
            "Updated forest": forest_probabilities(updated, game),
            "Frozen situational baseline": situational_probabilities(
                history, game, frozen_cutoff
            ),
            "Updated situational baseline": situational_probabilities(
                training, game, cutoff
            ),
        }

        actual.extend(game["pitch_type"].to_list())

        for name, probabilities in predictors.items():
            collected[name].append(probabilities)
            scores = score(game["pitch_type"], probabilities)

            # Keep each prediction's context available for later diagnostics.
            details = game[
                [
                    "game_date",
                    "game_pk",
                    "balls",
                    "strikes",
                    "stand",
                    "inning",
                    "previous_pitch",
                    "pitch_type",
                ]
            ].reset_index(drop=True).copy()

            details["predictor"] = name
            details["predicted"] = (
                probabilities.idxmax(axis=1).to_numpy()
            )

            for pitch in PITCH_TYPES:
                details[f"prob_{pitch}"] = (
                    probabilities[pitch].to_numpy()
                )

            pitch_results.append(details)

            per_game.append({
                "game_pk": game_id,
                "game_date": cutoff.date().isoformat(),
                "predictor": name,
                "pitches": len(game),
                **scores,
            })

        print(
            f"Completed game {number}/{len(games)}: "
            f"{cutoff.date()} | {len(training):,} training pitches"
        )

    print("\nCombined 2026 simulation results:")
    summary = []

    for name, batches in collected.items():
        # Pool pitches rather than giving short and long games equal weight.
        probabilities = pd.concat(batches, ignore_index=True)
        scores = score(actual, probabilities)

        summary.append({"predictor": name, **scores})

        print(
            f"{name}: accuracy {scores['accuracy']:.1%} | "
            f"log loss {scores['log_loss']:.3f}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(per_game).to_csv(
        DATA_DIR / "season_simulation_by_game.csv",
        index=False,
    )
    pd.DataFrame(summary).to_csv(
        DATA_DIR / "season_simulation_summary.csv",
        index=False,
    )
    pd.concat(pitch_results, ignore_index=True).to_csv(
        DATA_DIR / "season_simulation_predictions.csv",
        index=False,
    )

    print("\nSaved simulation results. App model unchanged.")
    print(
        "2026 is retrospective development evidence, "
        "not an untouched test set."
    )

def inspect_predictions():
    predictions = pd.read_csv(
        DATA_DIR / "season_simulation_predictions.csv"
    )

    predictors = [
        "Updated forest",
        "Updated situational baseline",
    ]
    count_results = []

    for name in predictors:
        rows = predictions[predictions["predictor"] == name]

        print(f"\n{name} — results by actual pitch type:")
        print(
            classification_report(
                rows["pitch_type"],
                rows["predicted"],
                zero_division=0,
            )
        )

        counts = pd.DataFrame({
            "actual": rows["pitch_type"].value_counts(),
            "predicted": rows["predicted"].value_counts(),
        }).fillna(0).astype(int)

        print("Actual versus predicted pitch counts:")
        print(counts.to_string())

        for (balls, strikes), group in rows.groupby(
            ["balls", "strikes"]
        ):
            probabilities = group[
                [f"prob_{pitch}" for pitch in PITCH_TYPES]
            ].copy()
            probabilities.columns = PITCH_TYPES

            # Include sample sizes so rare counts aren't overinterpreted.
            count_results.append({
                "balls": balls,
                "strikes": strikes,
                "predictor": name,
                "pitches": len(group),
                **score(group["pitch_type"], probabilities),
            })

    count_results = pd.DataFrame(count_results)

    print("\nResults by count:")
    print(
        count_results.to_string(
            index=False,
            formatters={
                "accuracy": lambda value: f"{value:.1%}",
                "log_loss": lambda value: f"{value:.3f}",
            },
        )
    )

    forest = predictions[
        predictions["predictor"] == "Updated forest"
    ]
    three_zero = forest[
        (forest["balls"] == 3) & (forest["strikes"] == 0)
    ]

    print(f"\n3–0 diagnostic: {len(three_zero)} pitches")

    if not three_zero.empty:
        actual_frequency = (
            three_zero["pitch_type"] == "FF"
        ).mean()
        estimated_probability = three_zero["prob_FF"].mean()

        print(f"Actual fastball frequency: {actual_frequency:.1%}")
        print(
            "Average forest fastball probability: "
            f"{estimated_probability:.1%}"
        )

    per_game = pd.read_csv(
        DATA_DIR / "season_simulation_by_game.csv"
    )
    per_game = per_game[
        per_game["predictor"].isin(predictors)
    ]

    comparison = per_game.pivot(
        index="game_date",
        columns="predictor",
        values="accuracy",
    )

    print("\nAccuracy by game (%):")
    print((comparison * 100).round(1).to_string())

    count_results.to_csv(
        DATA_DIR / "diagnostics_by_count.csv",
        index=False,
    )