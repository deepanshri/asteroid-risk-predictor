"""Locate the raw dataset. The project uses the Kaggle "NASA - Nearest Earth Objects" CSV.

Usage:  python -m src.fetch_data
The file is not downloaded automatically (Kaggle needs a login); if it is missing
this script prints exactly what to do.
"""
import sys

from src.config import RAW_CSV

INSTRUCTIONS = f"""
Dataset not found at: {RAW_CSV}

1. Open https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
2. Download the dataset (free Kaggle account) and unzip it.
3. Copy the CSV (neo.csv or neo_v2.csv) to:  {RAW_CSV}
4. Run this command again.
"""


def check_raw_data() -> bool:
    """Return True if the raw CSV exists, otherwise print download instructions."""
    if RAW_CSV.exists():
        print(f"OK: found {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
        return True
    print(INSTRUCTIONS)
    return False


if __name__ == "__main__":
    sys.exit(0 if check_raw_data() else 1)
