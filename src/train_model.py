"""Train the selected model for app use, rather than evaluate it."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES = ["balls", "strikes", "stand", "inning", "outs_when_up", "previous_pitch"]
TARGET = "pitch_type"
AS_OF = pd.Timestamp("2026-10-28")
MODEL_PATH = DATA_DIR.parent / "models" / "dylan_cease_v2.joblib"


def load_training_data():
    history = pd.read_csv(
        DATA_DIR / "dylan_cease_2021_2025_prepared.csv",
        parse_dates=["game_date"],
    )
    current = pd.read_csv(
        DATA_DIR / "dylan_cease_2026_through_2026-10-27.csv",
        parse_dates=["game_date"],
    )
    current = current[current["game_type"] == "R"].copy()
    current = current.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )
    # Shift before removing unlabeled pitches so we never skip a real pitch.
    previous = current.groupby(
        ["game_pk", "pitcher", "at_bat_number"]
    )[TARGET].shift(1)
    current["previous_pitch"] = previous.fillna("UNKNOWN")
    current.loc[current["pitch_number"] == 1, "previous_pitch"] = "START"

    data = pd.concat([history, current], ignore_index=True)
    data = data.dropna(subset=[TARGET]).copy()
    # Include the prediction year in the five-calendar-season limit.
    data = data[
        data["game_date"].dt.year.between(AS_OF.year - 4, AS_OF.year)
        & (data["game_date"] < AS_OF)
    ].copy()
    data["season"] = data["game_date"].dt.year
    keys = ["game_pk", "at_bat_number", "pitch_number"]
    if data.empty or data[FEATURES].isna().any().any():
        raise ValueError("Training data is empty or has missing inputs.")
    if data.duplicated(keys).any():
        raise ValueError("Duplicate pitches found; inspect the input files.")
    if set(data["season"]) != set(range(2022, 2027)):
        raise ValueError("Expected data from each season, 2022 through 2026.")
    if not set(data[TARGET]).issubset({"CH", "FC", "FF", "KC", "SI", "SL", "ST"}):
        raise ValueError("Unrecognized pitch types in training data.")
    return data.sort_values(["game_date", *keys])


def fit_model(data):
    preprocessing = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore"),
          ["stand", "previous_pitch"])],
        remainder="passthrough",
    )
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        )),
    ])
    age_days = (AS_OF - data["game_date"]).dt.total_seconds() / 86400
    weights = 0.5 ** (age_days / 365)
    # Match the weighting used in the evaluated season simulation.
    weights = weights / weights.mean()
    model.fit(data[FEATURES], data[TARGET],
              classifier__sample_weight=weights.to_numpy())
    return model


def main():
    data = load_training_data()
    model = fit_model(data)
    bundle = {
        "model": model,
        "features": FEATURES.copy(),
        "pitcher_id": 656302,
        "training_games": data["game_pk"].drop_duplicates().tolist(),
        "training_start": data["game_date"].min().date().isoformat(),
        "training_end": data["game_date"].max().date().isoformat(),
        "trained_as_of": AS_OF.date().isoformat(),
        "training_pitches": len(data),
        "seasons": sorted(data["season"].unique().tolist()),
        "recency_half_life_days": 365,
        "model_version": "v2",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Finish writing before replacing an existing v2 artifact.
    temporary_path = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary_path)
    temporary_path.replace(MODEL_PATH)
    print(data.groupby("season").agg(
        pitches=(TARGET, "size"), games=("game_pk", "nunique")
    ).to_string())
    print(f"\nTraining pitches: {len(data):,}")
    print(f"Training dates: {bundle['training_start']} to {bundle['training_end']}")
    print(f"Saved model: {MODEL_PATH}")
    print("This is a deployment fit, not a new accuracy evaluation.")


if __name__ == "__main__":
    main()
