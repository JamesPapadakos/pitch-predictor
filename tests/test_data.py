import pandas as pd
import pytest

from src.data import prepare_pitches
from src.model import recency_weights


@pytest.fixture
def raw_pitches():
    base = {
        "game_date": "2026-05-01",
        "game_pk": 123,
        "pitcher": 656302,
        "game_type": "R",
        "balls": 0,
        "strikes": 0,
        "stand": "R",
        "inning": 1,
        "outs_when_up": 0,
    }

    return pd.DataFrame([
        {
            **base,
            "at_bat_number": 1,
            "pitch_number": 1,
            "pitch_type": "FF",
        },
        {
            **base,
            "at_bat_number": 1,
            "pitch_number": 2,
            "pitch_type": None,
        },
        {
            **base,
            "at_bat_number": 1,
            "pitch_number": 3,
            "pitch_type": "SL",
        },
        {
            **base,
            "at_bat_number": 2,
            "pitch_number": 1,
            "pitch_type": "KC",
        },
    ])


def test_previous_pitch_preserves_unknowns_and_restarts(raw_pitches):
    prepared = prepare_pitches(raw_pitches)

    # The unlabeled second pitch must not make the third skip back to FF.
    assert prepared["previous_pitch"].tolist() == [
        "START",
        "UNKNOWN",
        "START",
    ]
    assert prepared["pitch_type"].tolist() == ["FF", "SL", "KC"]


def test_duplicate_pitches_are_rejected(raw_pitches):
    duplicated = pd.concat(
        [raw_pitches, raw_pitches.iloc[:1]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_pitches(duplicated)


def test_recency_weights_have_one_year_half_life():
    cutoff = pd.Timestamp("2026-10-28")
    data = pd.DataFrame({
        "game_date": [
            cutoff - pd.Timedelta(days=1),
            cutoff - pd.Timedelta(days=366),
        ]
    })

    weights = recency_weights(data, cutoff)

    assert weights.iloc[0] / weights.iloc[1] == pytest.approx(2.0)
    assert weights.mean() == pytest.approx(1.0)


def test_recency_weights_reject_prediction_date():
    cutoff = pd.Timestamp("2026-10-28")
    data = pd.DataFrame({"game_date": [cutoff]})

    with pytest.raises(ValueError, match="precede"):
        recency_weights(data, cutoff)