import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.model import FEATURES, pitcher_model_path, save_model


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "game_date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
        "game_pk": [1, 2],
        "pitcher": [656302, 656302],
        "balls": [0, 1],
        "strikes": [0, 2],
        "stand": ["R", "L"],
        "inning": [1, 3],
        "outs_when_up": [0, 1],
        "previous_pitch": ["START", "FF"],
        "pitch_type": ["FF", "SL"],
    })


def test_saved_models_keep_pitchers_separate(training_data, tmp_path):
    model = DummyClassifier(strategy="prior")
    model.fit(training_data[FEATURES], training_data["pitch_type"])

    first_path = pitcher_model_path(656302, tmp_path)
    second_path = pitcher_model_path(123456, tmp_path)

    save_model(
        model,
        training_data,
        path=first_path,
        pitcher_id=656302,
    )
    save_model(
        model,
        training_data.assign(pitcher=123456),
        path=second_path,
        pitcher_id=123456,
    )

    first = joblib.load(first_path)
    second = joblib.load(second_path)

    assert first_path != second_path
    assert first["pitcher_id"] == 656302
    assert second["pitcher_id"] == 123456
    assert first["features"] == FEATURES
    assert first["training_pitches"] == len(training_data)


def test_save_rejects_wrong_pitcher_metadata(training_data, tmp_path):
    model = DummyClassifier(strategy="prior")
    model.fit(training_data[FEATURES], training_data["pitch_type"])
    path = pitcher_model_path(123456, tmp_path)

    with pytest.raises(ValueError, match="only pitcher"):
        save_model(
            model,
            training_data,
            path=path,
            pitcher_id=123456,
        )

    assert not path.exists()