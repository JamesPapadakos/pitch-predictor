from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA_DIR.parent / "models" / "dylan_cease_v2.joblib"

FEATURES = [
    "balls",
    "strikes",
    "stand",
    "inning",
    "outs_when_up",
    "previous_pitch",
]

TARGET = "pitch_type"
PITCH_TYPES = ["CH", "FC", "FF", "KC", "SI", "SL", "ST"]

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


def save_model(model, data, path=MODEL_PATH):
    bundle = {
        "model": model,
        "features": FEATURES.copy(),
        "pitcher_id": 656302,
        "training_games": data["game_pk"].drop_duplicates().tolist(),
        "training_start": data["game_date"].min().date().isoformat(),
        "training_end": data["game_date"].max().date().isoformat(),
        "trained_as_of": AS_OF.date().isoformat(),
        "training_pitches": len(data),
        "seasons": sorted(data["game_date"].dt.year.unique().tolist()),
        "recency_half_life_days": 365,
        "model_version": "v2",
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)