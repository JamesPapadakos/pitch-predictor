from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MODELS_DIR = DATA_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "dylan_cease_v2.joblib"

DEFAULT_PITCHER_ID = 656302

FEATURES = [
    "balls",
    "strikes",
    "stand",
    "inning",
    "outs_when_up",
    "previous_pitch",
]

TARGET = "pitch_type"

#All possible pitches 
PITCH_NAMES = {
    "CH": "Changeup",
    "CS": "Slow curve",
    "CU": "Curveball",
    "EP": "Eephus",
    "FA": "Fastball",
    "FC": "Cutter",
    "FF": "Four-seam fastball",
    "FO": "Forkball",
    "FS": "Splitter",
    "FT": "Two-seam fastball",
    "GY": "Gyroball",
    "KC": "Knuckle curve",
    "KN": "Knuckleball",
    "SC": "Screwball",
    "SI": "Sinker",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
}

# Keep a consistent class order for probability comparisons.
PITCH_TYPES = sorted(PITCH_NAMES)

AS_OF = pd.Timestamp("2026-10-28")

def recency_weights(data, cutoff):
    age_days = (
        pd.Timestamp(cutoff) - data["game_date"]
    ).dt.total_seconds() / 86400

    if data.empty or (age_days <= 0).any():
        raise ValueError(
            "Training pitches must precede the prediction cutoff."
        )

    # Match the one-year half-life used during evaluation.
    weights = 0.5 ** (age_days / 365)
    return weights / weights.mean()


def fit_model(data, cutoff=AS_OF):
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

    weights = recency_weights(data, cutoff)

    model.fit(
        data[FEATURES],
        data[TARGET],
        classifier__sample_weight=weights.to_numpy(),
    )

    return model


def pitcher_model_path(pitcher_id, models_dir=MODELS_DIR):
    if pitcher_id <= 0:
        raise ValueError("Pitcher ID must be a positive integer.")

    return models_dir / f"pitcher_{pitcher_id}_v2.joblib"

def save_model(
    model,
    data,
    path=None,
    *,
    pitcher_id=DEFAULT_PITCHER_ID,
    pitcher_name=None,
):
    if (
        data.empty
        or "pitcher" not in data.columns
        or data["pitcher"].isna().any()
        or not data["pitcher"].eq(pitcher_id).all()
    ):
        raise ValueError(
            f"Training data must contain only pitcher {pitcher_id}."
        )

    path = (
        pitcher_model_path(pitcher_id)
        if path is None
        else Path(path)
    )

    bundle = {
        "model": model,
        "features": FEATURES.copy(),
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name or f"Pitcher {pitcher_id}",
        "pitch_types": model.classes_.tolist(),
        "training_games": data["game_pk"].drop_duplicates().tolist(),
        "training_start": data["game_date"].min().date().isoformat(),
        "training_end": data["game_date"].max().date().isoformat(),
        "trained_as_of": AS_OF.date().isoformat(),
        "training_pitches": len(data),
        "seasons": sorted(data["game_date"].dt.year.unique().tolist()),
        "recency_half_life_days": 365,
        "model_version": "v2",
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)