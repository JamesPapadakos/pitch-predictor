from pathlib import Path

import pandas as pd
from pybaseball import statcast_pitcher
from sklearn.metrics import accuracy_score, log_loss

from compare_history import FEATURES, PITCH_TYPES
from validate_model import build_model


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_END = "2026-10-27"


def load_test_data():
    file_path = DATA_DIR / f"dylan_cease_2026_through_{TEST_END}.csv"

    if file_path.exists():
        print("Using saved 2026 data.")
        pitches = pd.read_csv(file_path)
    else:
        print("Downloading 2026 data...")

        pitches = statcast_pitcher(
            start_dt="2026-03-01",
            end_dt=TEST_END,
            player_id=656302,
        )

        if pitches.empty:
            raise ValueError("No 2026 data returned.")

        pitches = pitches[pitches["game_type"] == "R"].copy()

        if pitches.empty:
            raise ValueError("No regular-season data returned.")

        pitches.to_csv(file_path, index=False)

    pitches = pitches[pitches["game_type"] == "R"].copy()

    pitches = pitches.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    # Build sequences before dropping missing target labels.
    previous = pitches.groupby(
        ["game_pk", "pitcher", "at_bat_number"]
    )["pitch_type"].shift(1)

    pitches["previous_pitch"] = previous.fillna("UNKNOWN")
    pitches.loc[pitches["pitch_number"] == 1, "previous_pitch"] = "START"

    return pitches.dropna(subset=["pitch_type"]).copy()


def score(actual, probabilities):
    predictions = probabilities.idxmax(axis=1).to_numpy()

    # Match the probability floor used in our history comparison.
    probabilities = probabilities.clip(lower=1e-12)
    probabilities = probabilities.div(
        probabilities.sum(axis=1), axis=0
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
    history = pd.read_csv(
        DATA_DIR / "dylan_cease_2021_2025_prepared.csv",
        parse_dates=["game_date"],
    )

    history = history.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    # Enforce the training boundary independently of the input filename.
    history = history[
        (history["season"] >= 2021)
        & (history["season"] <= 2025)
        & (history["game_date"] < pd.Timestamp("2026-01-01"))
    ].copy()

    test = load_test_data()

    unknown_types = set(test["pitch_type"]) - set(PITCH_TYPES)
    if unknown_types:
        raise ValueError(
            f"New test pitch types require explicit handling: {unknown_types}"
        )

    if test[FEATURES].isna().any().any():
        raise ValueError("Missing test inputs. Inspect before evaluation.")

    print(
        f"\nTest data: {test['game_pk'].nunique()} games, "
        f"{len(test):,} labeled pitches"
    )
    print(
        f"Dates: {test['game_date'].min()} "
        f"to {test['game_date'].max()}"
    )

    experiments = [
        ("2025 only", history[history["season"] == 2025], False),
        ("2021–2025 weighted", history, True),
    ]

    results = []

    baseline_training = history[history["season"] == 2025]

    # Estimate probabilities using past data, never the 2026 answers.
    frequencies = (
        baseline_training["pitch_type"]
        .value_counts(normalize=True)
        .reindex(PITCH_TYPES, fill_value=0)
    )

    frequency_probabilities = pd.DataFrame(
        [frequencies.to_numpy()] * len(test),
        columns=PITCH_TYPES,
    )

    situation_columns = ["balls", "strikes", "stand"]
    lookup = {}

    for situation, group in baseline_training.groupby(situation_columns):
        counts = (
            group["pitch_type"]
            .value_counts()
            .reindex(PITCH_TYPES, fill_value=0)
        )

        # Match the smoothing used during development.
        lookup[situation] = (
            (counts + 10 * frequencies) / (len(group) + 10)
        ).to_numpy()

    situational_probabilities = pd.DataFrame(
        [
            lookup.get(situation, frequencies.to_numpy())
            for situation in test[situation_columns].itertuples(
                index=False, name=None
            )
        ],
        columns=PITCH_TYPES,
    )

    baselines = {
        "2025 frequency baseline": frequency_probabilities,
        "2025 smoothed situational baseline": situational_probabilities,
    }

    print("\nBaselines learned from 2025:")
    print(f"Constant top prediction: {frequencies.idxmax()}")

    for name, probabilities in baselines.items():
        scores = score(test["pitch_type"], probabilities)

        print(
            f"{name}: accuracy {scores['accuracy']:.1%} | "
            f"log loss {scores['log_loss']:.3f}"
        )

        results.append({"model": name, **scores})

    for name, training, weighted in experiments:
        model = build_model()
        fit_options = {}

        if weighted:
            age_days = (
                pd.Timestamp("2026-01-01") - training["game_date"]
            ).dt.total_seconds() / 86400

            weights = 0.5 ** (age_days / 365)
            weights = weights / weights.mean()

            fit_options["classifier__sample_weight"] = weights.to_numpy()

        model.fit(
            training[FEATURES],
            training["pitch_type"],
            **fit_options,
        )

        probabilities = pd.DataFrame(
            model.predict_proba(test[FEATURES]),
            columns=model.classes_,
        ).reindex(columns=PITCH_TYPES, fill_value=0)

        scores = score(test["pitch_type"], probabilities)

        print(
            f"\n{name}: {len(training):,} training pitches\n"
            f"Accuracy: {scores['accuracy']:.1%}\n"
            f"Log loss: {scores['log_loss']:.3f}"
        )

        results.append({"model": name, **scores})

    pd.DataFrame(results).to_csv(
        DATA_DIR / "final_2026_comparison.csv",
        index=False,
    )

    print("\nSaved results to data/final_2026_comparison.csv")
    print("The app's saved model has not been replaced.")


if __name__ == "__main__":
    main()