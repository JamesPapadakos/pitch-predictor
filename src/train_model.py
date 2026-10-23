from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURES = ["balls", "strikes", "stand", "inning", "outs_when_up", "previous_pitch"]
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

    # Limit complexity to reduce memorization of small historical samples.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=30,
                    random_state=42,
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


if __name__ == "__main__":
    main()