import pandas as pd
from sklearn.metrics import classification_report

from compare_history import PITCH_TYPES
from test_2026 import DATA_DIR, score


PREDICTORS = [
    "Updated forest",
    "Updated situational baseline",
]


def get_probabilities(rows):
    # Restore the class labels and order expected by our scoring function.
    columns = [f"prob_{pitch}" for pitch in PITCH_TYPES]
    probabilities = rows[columns].copy()
    probabilities.columns = PITCH_TYPES
    return probabilities


def main():
    predictions = pd.read_csv(
        DATA_DIR / "season_simulation_predictions.csv"
    )

    count_results = []

    for name in PREDICTORS:
        rows = predictions[predictions["predictor"] == name]

        print(f"\n{name} — results by actual pitch type:")
        print(
            classification_report(
                rows["pitch_type"],
                rows["predicted"],
                zero_division=0,
            )
        )

        print("Actual versus predicted pitch counts:")
        counts = pd.DataFrame({
            "actual": rows["pitch_type"].value_counts(),
            "predicted": rows["predicted"].value_counts(),
        }).fillna(0).astype(int)

        print(counts.to_string())

        for (balls, strikes), group in rows.groupby(["balls", "strikes"]):
            scores = score(
                group["pitch_type"],
                get_probabilities(group),
            )

            # Show sample size alongside metrics for rare counts.
            count_results.append({
                "balls": balls,
                "strikes": strikes,
                "predictor": name,
                "pitches": len(group),
                **scores,
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
        actual_fastballs = (three_zero["pitch_type"] == "FF").mean()
        predicted_fastball_probability = three_zero["prob_FF"].mean()

        print(f"Actual fastball frequency: {actual_fastballs:.1%}")
        print(
            "Average forest fastball probability: "
            f"{predicted_fastball_probability:.1%}"
        )

    per_game = pd.read_csv(DATA_DIR / "season_simulation_by_game.csv")
    per_game = per_game[per_game["predictor"].isin(PREDICTORS)]

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


if __name__ == "__main__":
    main()