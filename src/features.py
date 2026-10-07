"""Feature engineering. Everything is a pure function of the 3 raw inputs, so it can
live inside the sklearn Pipeline (via FunctionTransformer) and run unchanged in the app.

Raw inputs
  absolute_magnitude (H)  brightness at standard distance; lower = bigger asteroid
  relative_velocity       speed relative to Earth at close approach, km/h
  miss_distance           distance at closest approach, km

Engineered features
  velocity_kms        relative velocity in km/s (easier units)
  log_velocity        log10 of velocity; tames the right-skewed distribution
  miss_distance_ld    miss distance in lunar distances (1 LD = 384,400 km)
  log_miss_distance   log10 of the miss distance in LD
  within_moon_orbit   1 if the flyby was closer than the Moon (< 1 LD)
  within_10_ld        1 if the flyby was closer than 10 lunar distances
  size_flag           1 if H <= 22 (roughly > 140 m; HALF of the official PHA definition)
  log_energy_proxy    log10(D^3 * v^2): rough kinetic-energy proxy, since mass ~ D^3
                      and D ~ 10^(-H/5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LUNAR_DISTANCE_KM = 384_400.0
H_PHA_LIMIT = 22.0


def diameter_km_from_h(h, albedo: float = 0.15):
    """Estimate diameter (km) from absolute magnitude H and an assumed albedo."""
    return 1329 / np.sqrt(albedo) * 10 ** (-np.asarray(h, dtype=float) / 5)


def select_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Keep only the listed feature columns (used to define each experiment)."""
    return X[cols]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all engineered features for the raw input columns."""
    h = X["absolute_magnitude"].astype(float)
    v = X["relative_velocity"].astype(float)
    miss_ld = X["miss_distance"].astype(float) / LUNAR_DISTANCE_KM
    out = pd.DataFrame(index=X.index)
    out["absolute_magnitude"] = h
    out["size_flag"] = (h <= H_PHA_LIMIT).astype(float).where(h.notna())
    out["velocity_kms"] = v / 3600
    out["log_velocity"] = np.log10(v)
    out["miss_distance_ld"] = miss_ld
    out["log_miss_distance"] = np.log10(miss_ld)
    out["within_moon_orbit"] = (miss_ld < 1).astype(float).where(miss_ld.notna())
    out["within_10_ld"] = (miss_ld < 10).astype(float).where(miss_ld.notna())
    out["log_energy_proxy"] = np.log10(diameter_km_from_h(h) ** 3 * (v / 3600) ** 2)
    return out
