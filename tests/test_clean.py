"""Unit tests for src/clean.py and src/features.py."""
import numpy as np
import pandas as pd
import pytest

from src.clean import class_balance, clean, parse_target
from src.features import add_features


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["(A)", "(A)", "(B)", "(C)"],
        "est_diameter_min": [0.1, 0.1, 0.2, 0.3],
        "est_diameter_max": [0.2, 0.2, 0.4, 0.6],
        "relative_velocity": [10000.0, 10000.0, -5.0, 30000.0],
        "miss_distance": [1e6, 1e6, 2e6, 3e6],
        "orbiting_body": ["Earth"] * 4,
        "absolute_magnitude": [20.0, 20.0, 25.0, 18.0],
        "hazardous": [True, True, False, "False"],
    })


def test_parse_target_variants():
    out = parse_target(pd.Series([True, False, "Y", "n", 1, 0]))
    assert out.tolist() == [1, 0, 1, 0, 1, 0]


def test_parse_target_rejects_garbage():
    with pytest.raises(ValueError):
        parse_target(pd.Series(["maybe"]))


def test_clean_drops_duplicates_and_constants():
    out = clean(sample_df())
    assert len(out) == 3
    assert "orbiting_body" not in out.columns
    assert out["hazardous"].isin([0, 1]).all()


def test_clean_keeps_rows_with_impossible_values_as_nan():
    out = clean(sample_df())
    assert len(out) == 3                      # rows are not dropped
    assert out["relative_velocity"].isna().sum() == 1


def test_class_balance_sums_to_one():
    bal = class_balance(clean(sample_df()))
    assert bal["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_add_features_values():
    raw = pd.DataFrame({"absolute_magnitude": [21.0, 25.0],
                        "relative_velocity": [36000.0, 36000.0],
                        "miss_distance": [384400.0, 3844000.0]})
    f = add_features(raw)
    assert f["velocity_kms"].tolist() == [10.0, 10.0]
    assert f["size_flag"].tolist() == [1.0, 0.0]
    assert f["within_moon_orbit"].tolist() == [0.0, 0.0]
    assert f["miss_distance_ld"].iloc[1] == pytest.approx(10.0)
    assert np.isfinite(f.to_numpy()).all()
