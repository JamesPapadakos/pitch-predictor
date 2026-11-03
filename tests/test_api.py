from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)

SITUATION = {
    "balls": 1,
    "strikes": 2,
    "stand": "L",
    "inning": 6,
    "outs_when_up": 1,
    "previous_pitch": "FF",
}

PITCHERS = [
    {
        "pitcher_id": 656302,
        "name": "Dylan Cease",
        "pitch_types": ["FF", "SL"],
        "training_start": "2022-04-09",
        "training_end": "2026-09-18",
    },
    {
        "pitcher_id": 554430,
        "name": "Zack Wheeler",
        "pitch_types": ["FF", "SI", "FS"],
        "training_start": "2022-04-12",
        "training_end": "2026-09-27",
    },
]


@pytest.fixture(autouse=True)
def available_pitchers():
    # Keep API tests independent of locally trained model files.
    with patch("src.api.list_available_pitchers", return_value=PITCHERS):
        yield


def test_list_pitchers():
    response = client.get("/pitchers")

    assert response.status_code == 200
    assert response.json() == PITCHERS


@pytest.mark.parametrize(
    ("pitcher_id", "name"),
    [
        (656302, "Dylan Cease"),
        (554430, "Zack Wheeler"),
    ],
)


def test_predict_selected_pitcher(pitcher_id, name):
    with patch(
        "src.api.predict_pitch_probabilities",
        return_value=[("FF", 0.6), ("SL", 0.4)],
    ) as predict:
        response = client.post(
            "/predict",
            json={**SITUATION, "pitcher_id": pitcher_id},
        )

    assert response.status_code == 200
    assert response.json() == {
        "pitcher_id": pitcher_id,
        "pitcher": name,
        "probabilities": [
            {"pitch_type": "FF", "probability": 0.6},
            {"pitch_type": "SL", "probability": 0.4},
        ],
    }
    predict.assert_called_once_with(SITUATION, pitcher_id=pitcher_id)


def test_predict_defaults_to_cease():
    with patch(
        "src.api.predict_pitch_probabilities",
        return_value=[("FF", 1.0)],
    ) as predict:
        response = client.post("/predict", json=SITUATION)

    assert response.status_code == 200
    assert response.json()["pitcher_id"] == 656302
    predict.assert_called_once_with(SITUATION, pitcher_id=656302)


def test_unknown_pitcher():
    with patch("src.api.predict_pitch_probabilities") as predict:
        response = client.post(
            "/predict",
            json={**SITUATION, "pitcher_id": 999999},
        )

    assert response.status_code == 404
    predict.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"balls": 4},
        {"strikes": 3},
        {"pitcher_id": -1},
        {"previous_pitch": "INVALID"},
    ],
)
def test_invalid_inputs(changes):
    with patch("src.api.predict_pitch_probabilities") as predict:
        response = client.post(
            "/predict",
            json={**SITUATION, **changes},
        )

    assert response.status_code == 422
    predict.assert_not_called()


def test_splitter_is_accepted():
    with patch(
        "src.api.predict_pitch_probabilities",
        return_value=[("FS", 1.0)],
    ) as predict:
        response = client.post(
            "/predict",
            json={
                **SITUATION,
                "pitcher_id": 554430,
                "previous_pitch": "FS",
            },
        )

    assert response.status_code == 200
    predict.assert_called_once_with(
        {**SITUATION, "previous_pitch": "FS"},
        pitcher_id=554430,
    )


def test_missing_model():
    with patch(
        "src.api.predict_pitch_probabilities",
        side_effect=FileNotFoundError("Missing model"),
    ):
        response = client.post("/predict", json=SITUATION)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The selected pitcher's saved model is unavailable."
    )