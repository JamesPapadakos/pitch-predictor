import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_model import DATA_DIR, FEATURES, TARGET


# Use a fixed pitch vocabulary so probability columns match across rounds.
PITCH_TYPES = ["CH", "FF", "KC", "SI", "SL", "ST"]
SITUATION_COLUMNS = ["balls", "strikes", "stand"]


def build_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["stand", "previous_pitch"],
            )
        ],
        remainder="passthrough",
    )

    # Keep the existing forest settings so only the evaluation window changes.
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def baseline_probabilities(training, validation):
    frequencies = (
        training[TARGET].value_counts(normalize=True)
        .reindex(PITCH_TYPES, fill_value=0)
    )

    frequency_predictions = pd.DataFrame(
        [frequencies.to_numpy()] * len(validation),
        columns=PITCH_TYPES,
    )

    lookup = {}

    for situation, group in training.groupby(SITUATION_COLUMNS):
        counts = (
            group[TARGET].value_counts()
            .reindex(PITCH_TYPES, fill_value=0)
        )

        # Match the smoothing strength used in our earlier comparison.
        lookup[situation] = (
            (counts + 10 * frequencies) / (len(group) + 10)
        ).to_numpy()

    situational_predictions = pd.DataFrame(
        [
            lookup.get(situation, frequencies.to_numpy())
            for situation in validation[SITUATION_COLUMNS].itertuples(
                index=False, name=None
            )
        ],
        columns=PITCH_TYPES,
    )

    return frequency_predictions, situational_predictions


def evaluate(actual, probabilities):
    predicted = probabilities.idxmax(axis=1).to_numpy()

    # Apply the same small floor to all predictors. A pitch type absent
    # from training still receives a large penalty if it appears later.
    probabilities = probabilities.clip(lower=1e-12)
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

    return {
        "accuracy": accuracy_score(actual, predicted),
        "log_loss": log_loss(
            actual,
            probabilities.to_numpy(),
            labels=PITCH_TYPES,
        ),
    }


def main():
    data = pd.read_csv(DATA_DIR / "dylan_cease_2025_prepared.csv")
    data = data.sort_values(["game_date", "game_pk"])
    games = data["game_pk"].drop_duplicates().to_list()

    if len(games) != 32:
        raise ValueError("These evaluation windows expect our 32-game dataset.")

    # Exclude the final seven games before constructing any validation rounds.
    development_games = games[:25]
    development = data[data["game_pk"].isin(development_games)]

    results = []

    for round_number, train_end in enumerate([10, 15, 20], start=1):
        training_games = development_games[:train_end]
        validation_games = development_games[train_end:train_end + 5]

        training = development[
            development["game_pk"].isin(training_games)
        ]
        validation = development[
            development["game_pk"].isin(validation_games)
        ]

        # Refit both preprocessing and the forest using this round's past only.
        model = build_model()
        model.fit(training[FEATURES], training[TARGET])

        forest_probabilities = pd.DataFrame(
            model.predict_proba(validation[FEATURES]),
            columns=model.classes_,
        ).reindex(columns=PITCH_TYPES, fill_value=0)

        frequency, situational = baseline_probabilities(
            training, validation
        )

        predictors = {
            "Frequency baseline": frequency,
            "Smoothed situational": situational,
            "Random forest": forest_probabilities,
        }

        print(
            f"\nRound {round_number}: train games 1–{train_end}, "
            f"validate games {train_end + 1}–{train_end + 5}"
        )

        for name, probabilities in predictors.items():
            scores = evaluate(validation[TARGET], probabilities)

            results.append({
                "round": round_number,
                "predictor": name,
                "pitches": len(validation),
                **scores,
            })

            print(
                f"{name}: accuracy {scores['accuracy']:.1%}, "
                f"log loss {scores['log_loss']:.3f}"
            )

    results = pd.DataFrame(results)

    print("\nCombined validation results:")
    for name, group in results.groupby("predictor", sort=False):
        # Weight by pitches because five-game windows can differ in size.
        weights = group["pitches"] / group["pitches"].sum()
        accuracy = (group["accuracy"] * weights).sum()
        loss = (group["log_loss"] * weights).sum()

        print(f"{name}: accuracy {accuracy:.1%}, log loss {loss:.3f}")

    print("\nFinal seven test games remain unused.")


if __name__ == "__main__":
    main()