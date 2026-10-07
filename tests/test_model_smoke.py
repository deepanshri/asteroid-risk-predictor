"""Smoke test: the saved model loads and predicts a valid probability for a sample row."""
import joblib
import pandas as pd
import pytest

from src.config import MODEL_PATH, RAW_FEATURES

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")


def test_saved_model_predicts_sample_row():
    bundle = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{"absolute_magnitude": 20.0, "relative_velocity": 60000.0,
                         "miss_distance": 3e7}])[RAW_FEATURES]
    proba = bundle["pipeline"].predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0
    assert 0.0 < bundle["threshold"] < 1.0


def test_bigger_asteroid_is_scored_higher_than_tiny_one():
    pipe = joblib.load(MODEL_PATH)["pipeline"]
    rows = pd.DataFrame({"absolute_magnitude": [19.0, 30.0],
                         "relative_velocity": [50000.0, 50000.0],
                         "miss_distance": [3e7, 3e7]})[RAW_FEATURES]
    big, tiny = pipe.predict_proba(rows)[:, 1]
    assert big > tiny
