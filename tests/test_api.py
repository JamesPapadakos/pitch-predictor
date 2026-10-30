from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def situation():
    return {
        "balls": 1,
        "strikes": 2,
        "stand": "L",
        "inning": 6,
        "outs_when_up": 1,
        "previous_pitch": "FF",
    }


def test_prediction_returns_expected_response(client, situation):
    # Isolate the API contract from training and model-file availability.
    with patch(
        "src.api.predict_pitch_probabilities",
        return_value=[("FF", 0.6), ("SL", 0.4)],
    ) as predict:
        response = client.post("/predict", json=situation)

    assert response.status_code == 200
    assert response.json() == {
        "pitcher": "Dylan Cease",
        "probabilities": [
            {"pitch_type": "FF", "probability": 0.6},
            {"pitch_type": "SL", "probability": 0.4},
        ],
    }
    predict.assert_called_once_with(situation)


def test_invalid_count_is_rejected(client, situation):
    situation["balls"] = 4

    with patch("src.api.predict_pitch_probabilities") as predict:
        response = client.post("/predict", json=situation)

    assert response.status_code == 422
    predict.assert_not_called()


def test_missing_model_returns_service_unavailable(client, situation):
    with patch(
        "src.api.predict_pitch_probabilities",
        side_effect=FileNotFoundError("Missing model"),
    ):
        response = client.post("/predict", json=situation)

    assert response.status_code == 503
    assert "python -m scripts.train" in response.json()["detail"]