import pandas as pd

from compare_history import FEATURES, PITCH_TYPES
from test_2026 import DATA_DIR, load_test_data, score
from validate_model import build_model


def recency_weights(training, cutoff):
    age_days = (
        cutoff - training["game_date"]
    ).dt.total_seconds() / 86400

    weights = 0.5 ** (age_days / 365)
    return weights / weights.mean()


def fit_forest(training, cutoff):
    model = build_model()
    weights = recency_weights(training, cutoff)

    model.fit(
        training[FEATURES],
        training["pitch_type"],
        classifier__sample_weight=weights.to_numpy(),
    )

    return model


def forest_probabilities(model, game):
    return pd.DataFrame(
        model.predict_proba(game[FEATURES]),
        columns=model.classes_,
    ).reindex(columns=PITCH_TYPES, fill_value=0)


def situational_probabilities(training, game, cutoff):
    training = training.copy()

    # Give the baseline the same recency weighting as the forest.
    training["weight"] = recency_weights(training, cutoff).to_numpy()

    overall = (
        training.groupby("pitch_type")["weight"].sum()
        .reindex(PITCH_TYPES, fill_value=0)
    )
    overall = overall / overall.sum()

    situation_columns = ["balls", "strikes", "stand"]
    lookup = {}

    for situation, group in training.groupby(situation_columns):
        counts = (
            group.groupby("pitch_type")["weight"].sum()
            .reindex(PITCH_TYPES, fill_value=0)
        )

        # Keep the same smoothing strength used in earlier experiments.
        lookup[situation] = (
            (counts + 10 * overall) / (counts.sum() + 10)
        ).to_numpy()

    return pd.DataFrame(
        [
            lookup.get(situation, overall.to_numpy())
            for situation in game[situation_columns].itertuples(
                index=False, name=None
            )
        ],
        columns=PITCH_TYPES,
    )


def main():
    history = pd.read_csv(
        DATA_DIR / "dylan_cease_2021_2025_prepared.csv",
        parse_dates=["game_date"],
    )

    # Five calendar seasons including the prediction year: 2022–2026.
    history = history[
        history["season"].between(2022, 2025)
        & (history["game_date"] < pd.Timestamp("2026-01-01"))
    ].copy()

    current = load_test_data()
    current["game_date"] = pd.to_datetime(current["game_date"])
    current["season"] = 2026

    current = current.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    unknown_types = set(current["pitch_type"]) - set(PITCH_TYPES)
    if unknown_types:
        raise ValueError(f"Unrecognized pitch types: {unknown_types}")

    if current[FEATURES].isna().any().any():
        raise ValueError("Missing prediction inputs.")

    history = history.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    all_data = pd.concat([history, current], ignore_index=True)

    # Fit once before any 2026 games; this model stays frozen.
    frozen = fit_forest(history, pd.Timestamp("2026-01-01"))

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

        # Exclude the entire current date to prevent same-game leakage.
        training = all_data[all_data["game_date"] < cutoff].copy()
        training = training.sort_values(
            ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        )

        updated = fit_forest(training, cutoff)

        predictors = {
            "Frozen forest": forest_probabilities(frozen, game),
            "Updated forest": forest_probabilities(updated, game),
            "Frozen situational baseline": situational_probabilities(
                history, game, pd.Timestamp("2026-01-01")
            ),
            "Updated situational baseline": situational_probabilities(
                training, game, cutoff
            ),
        }

        actual.extend(game["pitch_type"].to_list())

        for name, probabilities in predictors.items():
            collected[name].append(probabilities)
            scores = score(game["pitch_type"], probabilities)
            # Preserve each prediction's situation so aggregate scores can be investigated.
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
            details["predicted"] = probabilities.idxmax(axis=1).to_numpy()

            for pitch in PITCH_TYPES:
                details[f"prob_{pitch}"] = probabilities[pitch].to_numpy()

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

    # Pool individual predictions so each pitch has equal evaluation weight.
    print("\nCombined 2026 simulation results:")
    summary = []

    for name, batches in collected.items():
        probabilities = pd.concat(batches, ignore_index=True)
        scores = score(actual, probabilities)

        print(
            f"{name}: accuracy {scores['accuracy']:.1%} | "
            f"log loss {scores['log_loss']:.3f}"
        )

        summary.append({"predictor": name, **scores})

    pd.DataFrame(per_game).to_csv(
        DATA_DIR / "season_simulation_by_game.csv", index=False
    )
    pd.DataFrame(summary).to_csv(
        DATA_DIR / "season_simulation_summary.csv", index=False
    )

    pd.concat(pitch_results, ignore_index=True).to_csv(
    DATA_DIR / "season_simulation_predictions.csv",
    index=False,
    )

    print("\nSaved simulation results. App model unchanged.")


if __name__ == "__main__":
    main()