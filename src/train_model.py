from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURES = ["balls", "strikes", "stand", "inning", "outs_when_up", "previous_pitch",]
TARGET = "pitch_type"


def main():
    data = pd.read_csv(DATA_DIR / "dylan_cease_2025_prepared.csv")
    data = data.sort_values(["game_date", "game_pk"])

    # Keep whole games together and simulate predicting future games.
    games = data["game_pk"].drop_duplicates().to_list()
    train_end = int(len(games) * 0.6)
    validation_end = int(len(games) * 0.8)

    training_games = games[:train_end]
    validation_games = games[train_end:validation_end]
    test_games = games[validation_end:]

    training = data[data["game_pk"].isin(training_games)]
    validation = data[data["game_pk"].isin(validation_games)]

    X_train = training[FEATURES]
    y_train = training[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_validation)

    situation_columns = ["balls", "strikes", "stand"]

    # Learn the lookup from training games so validation answers stay unseen.
    situation_lookup = (
        training.groupby(situation_columns)[TARGET]
        .agg(lambda pitches: pitches.mode().iloc[0])
        .to_dict()
    )

    # An unseen situation needs a fallback learned from the same training data.
    fallback_pitch = y_train.mode().iloc[0]

    situational_predictions = [
        situation_lookup.get(situation, fallback_pitch)
        for situation in X_validation[situation_columns].itertuples(
            index=False, name=None
        )
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), ["stand", "previous_pitch"])
        ],
        remainder="passthrough",
    )

    # Average across trees to reduce reliance on one tree's particular splits.
    model = Pipeline(
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    print(f"Training: {len(training_games)} games, {len(training):,} pitches")
    print(
        f"Validation: {len(validation_games)} games, "
        f"{len(validation):,} pitches"
    )

    # Reserve these games for a final evaluation after model development.
    print(f"Reserved test games: {len(test_games)}")

    baseline_accuracy = accuracy_score(y_validation, baseline_predictions)
    model_accuracy = accuracy_score(y_validation, predictions)

    print(f"\nBaseline prediction: {baseline.classes_[baseline.class_prior_.argmax()]}")
    print(f"Baseline validation accuracy: {baseline_accuracy:.1%}")
    situational_accuracy = accuracy_score(
        y_validation, situational_predictions
    )
    print(f"Situational baseline validation accuracy: {situational_accuracy:.1%}")

    print(f"Model training accuracy: {model.score(X_train, y_train):.1%}")
    print(f"Model validation accuracy: {model_accuracy:.1%}")

    print("\nValidation results by pitch type:")
    print(classification_report(y_validation, predictions, zero_division=0))

    # Show whether rare pitches are never predicted or are predicted incorrectly.
    prediction_counts = pd.DataFrame({
        "actual": y_validation.value_counts(),
        "predicted": pd.Series(predictions).value_counts(),
    }).fillna(0).astype(int)

    print("\nActual versus predicted pitch counts:")
    print(prediction_counts.to_string())

    # Use the model's class order for every probability comparison.
    classes = model.classes_
    model_probabilities = model.predict_proba(X_validation)

    # Compare against training frequencies, rather than assigning slider 100%.
    frequency_baseline = DummyClassifier(strategy="prior")
    frequency_baseline.fit(X_train, y_train)

    frequency_probabilities = pd.DataFrame(
        frequency_baseline.predict_proba(X_validation),
        columns=frequency_baseline.classes_,
    )[classes].to_numpy()

    training_frequencies = (
        y_train.value_counts(normalize=True)
        .reindex(classes, fill_value=0)
    )

    situation_columns = ["balls", "strikes", "stand"]
    probability_lookup = {}

    for situation, group in training.groupby(situation_columns):
        counts = group[TARGET].value_counts().reindex(classes, fill_value=0)

        # Blend in 10 pitches' worth of overall training frequencies so small
        # groups don't declare a possible pitch impossible after zero examples.
        probabilities = (
            counts + 10 * training_frequencies
        ) / (len(group) + 10)

        probability_lookup[situation] = probabilities.to_numpy()

    # Fall back to overall training frequencies for an unseen situation.
    situational_probabilities = [
        probability_lookup.get(
            situation,
            training_frequencies.to_numpy(),
        )
        for situation in X_validation[situation_columns].itertuples(
            index=False, name=None
        )
    ]

    print("\nValidation log loss (lower is better):")

    probability_predictions = {
        "Training-frequency baseline": frequency_probabilities,
        "Smoothed situational baseline": situational_probabilities,
        "Random forest": model_probabilities,
    }

    # An unseen target class needs explicit handling rather than a misleading score.
    unseen_classes = set(y_validation) - set(classes)

    if unseen_classes:
        print(f"Cannot score yet: unseen pitch types {sorted(unseen_classes)}")
    else:
        for name, probabilities in probability_predictions.items():
            loss = log_loss(y_validation, probabilities, labels=classes)
            print(f"{name}: {loss:.3f}")

    preview = pd.DataFrame(model_probabilities[:5], columns=classes)
    preview.insert(0, "actual", y_validation.iloc[:5].to_numpy())

    print("\nProbabilities for the first five validation pitches:")
    print(preview.round(3).to_string(index=False))


if __name__ == "__main__":
    main()