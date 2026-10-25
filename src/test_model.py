import pandas as pd
from sklearn.metrics import classification_report

from train_model import DATA_DIR, FEATURES, TARGET
from validate_model import (
    PITCH_TYPES,
    baseline_probabilities,
    build_model,
    evaluate,
)


def main():
    data = pd.read_csv(DATA_DIR / "dylan_cease_2025_prepared.csv")
    data = data.sort_values(["game_date", "game_pk"])
    games = data["game_pk"].drop_duplicates().to_list()

    if len(games) != 32:
        raise ValueError("This final evaluation expects our 32-game dataset.")

    training = data[data["game_pk"].isin(games[:25])]
    test = data[data["game_pk"].isin(games[25:])]

    # Freeze the model choices before looking at the reserved games.
    model = build_model()
    model.fit(training[FEATURES], training[TARGET])

    forest_probabilities = pd.DataFrame(
        model.predict_proba(test[FEATURES]),
        columns=model.classes_,
    ).reindex(columns=PITCH_TYPES, fill_value=0)

    # Each baseline must also learn exclusively from the first 25 games.
    frequency, situational = baseline_probabilities(training, test)

    predictors = {
        "Frequency baseline": frequency,
        "Smoothed situational": situational,
        "Random forest": forest_probabilities,
    }

    print(f"Training: 25 games, {len(training):,} pitches")
    print(f"Final test: 7 games, {len(test):,} pitches")
    print(
        f"Test dates: {test['game_date'].min()} "
        f"to {test['game_date'].max()}"
    )

    print("\nFinal test results:")
    for name, probabilities in predictors.items():
        scores = evaluate(test[TARGET], probabilities)

        print(
            f"{name}: accuracy {scores['accuracy']:.1%}, "
            f"log loss {scores['log_loss']:.3f}"
        )

    predictions = model.predict(test[FEATURES])

    print("\nRandom forest results by pitch type:")
    print(
        classification_report(
            test[TARGET],
            predictions,
            zero_division=0,
        )
    )

    prediction_counts = pd.DataFrame({
        "actual": test[TARGET].value_counts(),
        "predicted": pd.Series(predictions).value_counts(),
    }).fillna(0).astype(int)

    print("\nActual versus predicted pitch counts:")
    print(prediction_counts.to_string())


if __name__ == "__main__":
    main()