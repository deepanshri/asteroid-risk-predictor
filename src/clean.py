"""Load and clean the raw NEO close-approach table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV, GROUP, RAW_CSV, TARGET

NUMERIC_COLS = [
    "est_diameter_min", "est_diameter_max", "relative_velocity",
    "miss_distance", "absolute_magnitude",
]


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV from disk."""
    return pd.read_csv(path)


def parse_target(series: pd.Series) -> pd.Series:
    """Convert a hazardous flag (True/False, 'Y'/'N', 1/0) to int 1/0."""
    mapping = {"true": 1, "t": 1, "y": 1, "yes": 1, "1": 1,
               "false": 0, "f": 0, "n": 0, "no": 0, "0": 0}
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError("Unrecognised values in the hazardous column")
    return out.astype(int)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with a single unique value (they carry no information)."""
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse types, encode the target, drop duplicates and impossible values.

    Missing values are NOT dropped or filled here: imputation (median + missing
    indicators) happens inside the sklearn Pipeline so it is fit on training data only.
    """
    df = df.copy()
    df[TARGET] = parse_target(df[TARGET])
    for col in NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = drop_constant_columns(df)
    # Physically impossible values become NaN (the pipeline imputer handles them)
    for col in ("miss_distance", "relative_velocity"):
        df.loc[df[col] <= 0, col] = np.nan
    # Group key: every close approach of the same asteroid shares one name
    df[GROUP] = df["name"].str.strip()
    return df.reset_index(drop=True)


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return the count and share of each class."""
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame({"count": counts, "share": (counts / counts.sum()).round(4)})


def diameter_h_formula_error(df: pd.DataFrame) -> float:
    """Max abs. log10 gap between est_diameter_min and the H-based formula.

    D = 1329 / sqrt(albedo) * 10**(-H/5) km. The dataset's diameters follow this
    (albedo 0.25 for _min), so they carry no information beyond H and are not used
    as features.
    """
    d_formula = 1329 / np.sqrt(0.25) * 10 ** (-df["absolute_magnitude"] / 5)
    return float(np.abs(np.log10(df["est_diameter_min"] / d_formula)).max())


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    print(f"Rows: raw={len(raw):,}  clean={len(cleaned):,}")
    print(f"Unique asteroids: {cleaned[GROUP].nunique():,} (each row is one close approach)")
    print(f"Missing values: {int(cleaned[NUMERIC_COLS].isna().sum().sum())}")
    print("Dropped constant columns:", sorted(set(raw.columns) - set(cleaned.columns)))
    print("Class balance:\n", class_balance(cleaned))
    print(f"Diameter vs H-formula, max log10 error: {diameter_h_formula_error(cleaned):.4f}")
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    print("Saved", CLEAN_CSV)
