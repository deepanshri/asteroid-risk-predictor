"""Central configuration: paths, constants and feature groups used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "neo.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "neo_clean.csv"
MODEL_PATH = ROOT / "models" / "pha_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42             # one seed for splits, CV and models -> reproducible results
TEST_SIZE = 0.20
N_FOLDS = 5
TARGET = "hazardous"
GROUP = "asteroid"    # one asteroid appears many times (one row per close approach)
TARGET_RECALL = 0.90  # threshold is tuned so >= 90% of hazardous asteroids are caught

# Raw model inputs (what a user must supply in the app)
RAW_FEATURES = ["absolute_magnitude", "relative_velocity", "miss_distance"]

# Which engineered features each experiment may see (see src/features.py).
SIZE_FEATURES = ["absolute_magnitude", "size_flag", "log_energy_proxy"]
FLYBY_FEATURES = [
    "velocity_kms", "log_velocity", "miss_distance_ld", "log_miss_distance",
    "within_moon_orbit", "within_10_ld",
]
EXPERIMENTS = {
    # MAIN model: H (brightness/size) + how the asteroid flew by
    "A_full": SIZE_FEATURES + FLYBY_FEATURES,
    # Ablation 1: remove H -> only flyby kinematics are left
    "B_flyby_only": FLYBY_FEATURES,
    # Ablation 2: only H -> shows how much of the answer is "just size"
    "C_H_only": ["absolute_magnitude", "size_flag"],
}
MAIN_EXPERIMENT = "A_full"
