import pandas as pd
import pytest
import io
import json

from src.data import (
    load_training_data,
    pitcher_data_dir,
    prepare_pitches,
    get_pitcher_name,
)
from src.model import recency_weights, AS_OF, DEFAULT_PITCHER_ID


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

def test_prepare_pitches_accepts_another_pitcher(raw_pitches):
    # Test configuration without downloading data.
    other_pitcher = raw_pitches.assign(pitcher=123456)

    prepared = prepare_pitches(
        other_pitcher,
        pitcher_id=123456,
    )

    assert prepared["pitch_type"].tolist() == ["FF", "SL", "KC"]
    assert prepared["previous_pitch"].tolist() == [
        "START",
        "UNKNOWN",
        "START",
    ]


def test_prepare_pitches_rejects_the_wrong_pitcher(raw_pitches):
    with pytest.raises(ValueError, match="Expected only pitches"):
        prepare_pitches(raw_pitches, pitcher_id=123456)

def test_loader_accepts_a_pitcher_with_one_season(raw_pitches, tmp_path):
    pitcher_id = 123456
    prepared = prepare_pitches(
        raw_pitches.assign(pitcher=pitcher_id),
        pitcher_id=pitcher_id,
    )
    prepared["pitcher"] = pitcher_id

    folder = pitcher_data_dir(pitcher_id, tmp_path)
    folder.mkdir(parents=True)

    path = folder / f"prepared_{AS_OF.year - 4}_{AS_OF.year}.csv"
    prepared.to_csv(path, index=False)

    loaded = load_training_data(
        data_dir=tmp_path,
        pitcher_id=pitcher_id,
    )

    assert len(loaded) == 3
    assert loaded["pitcher"].eq(pitcher_id).all()
    assert loaded["season"].nunique() == 1


def test_loader_rejects_wrong_pitcher_contents(raw_pitches, tmp_path):
    prepared = prepare_pitches(raw_pitches)
    prepared["pitcher"] = 123456

    folder = pitcher_data_dir(DEFAULT_PITCHER_ID, tmp_path)
    folder.mkdir(parents=True)

    path = folder / f"prepared_{AS_OF.year - 4}_{AS_OF.year}.csv"
    prepared.to_csv(path, index=False)

    with pytest.raises(ValueError, match="only pitcher"):
        load_training_data(data_dir=tmp_path)


def test_loader_reports_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts.data"):
        load_training_data(
            data_dir=tmp_path,
            pitcher_id=123456,
        )

@pytest.mark.parametrize("pitch_type", ["CU", "FS", "SV"])
def test_prepare_pitches_accepts_additional_types(
    raw_pitches,
    pitch_type,
):
    raw = raw_pitches.assign(pitch_type=pitch_type)

    prepared = prepare_pitches(raw)

    assert len(prepared) == len(raw)
    assert prepared["pitch_type"].eq(pitch_type).all()


def test_prepare_pitches_rejects_unrecognized_types(raw_pitches):
    raw = raw_pitches.assign(pitch_type="NOT_A_PITCH")

    with pytest.raises(ValueError, match="Unrecognized pitch types"):
        prepare_pitches(raw)

def test_pitcher_name_is_fetched_then_cached(tmp_path, monkeypatch):
    payload = {
        "people": [
            {"id": 123456, "fullName": "Example Pitcher"}
        ]
    }
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.StringIO(json.dumps(payload))

    monkeypatch.setattr("src.data.urlopen", fake_urlopen)

    first = get_pitcher_name(123456, data_dir=tmp_path)
    second = get_pitcher_name(123456, data_dir=tmp_path)

    assert first == second == "Example Pitcher"
    assert len(calls) == 1


def test_pitcher_name_rejects_wrong_player(tmp_path, monkeypatch):
    payload = {
        "people": [
            {"id": 999999, "fullName": "Wrong Player"}
        ]
    }

    monkeypatch.setattr(
        "src.data.urlopen",
        lambda url, timeout: io.StringIO(json.dumps(payload)),
    )

    with pytest.raises(ValueError, match="Invalid player metadata"):
        get_pitcher_name(123456, data_dir=tmp_path)

    assert not (tmp_path / "pitchers" / "123456" / "player.json").exists()