from src.data import load_training_data
from src.model import MODEL_PATH, fit_model, save_model


def main():
    data = load_training_data()
    model = fit_model(data)
    save_model(model, data)

    summary = data.groupby("season").agg(
        pitches=("pitch_type", "size"),
        games=("game_pk", "nunique"),
    )

    print(summary.to_string())
    print(f"\nTraining pitches: {len(data):,}")
    print(
        f"Training dates: "
        f"{data['game_date'].min().date()} to "
        f"{data['game_date'].max().date()}"
    )
    print(f"Saved model: {MODEL_PATH}")
    print("Deployment fit completed; this is not a new accuracy evaluation.")


if __name__ == "__main__":
    main()