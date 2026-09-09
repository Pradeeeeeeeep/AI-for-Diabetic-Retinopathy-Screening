"""
config.py — RetinaScan AI global configuration.
Loads from .env and provides typed settings for all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "reports"
WEIGHTS_DIR = BASE_DIR / "backend" / "models" / "weights"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model weight paths ────────────────────────────────────────────────────────
MODEL_GRADER_PATH = os.getenv(
    "MODEL_GRADER_PATH",
    str(WEIGHTS_DIR / "dr_grader_india.pth")
)
MODEL_VESSEL_PATH = os.getenv(
    "MODEL_VESSEL_PATH",
    str(WEIGHTS_DIR / "vessel_unet.pth")
)
MODEL_SVM_PATH = os.getenv(
    "MODEL_SVM_PATH",
    str(WEIGHTS_DIR / "svm_ma_classifier.pkl")
)
TEMPERATURE_PATH = os.getenv(
    "TEMPERATURE_PATH",
    str(WEIGHTS_DIR / "temperature.json")
)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./retinascan.db"   # SQLite fallback for local dev
)

# ── Storage ───────────────────────────────────────────────────────────────────
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
STORAGE_PATH = os.getenv("STORAGE_PATH", str(OUTPUT_DIR))

# ── Inference ─────────────────────────────────────────────────────────────────
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DEVICE = os.getenv("DEVICE", "cuda:0" if USE_GPU else "cpu")
SIMULATION_MODE = not Path(MODEL_GRADER_PATH).exists()

# ── Quality assessment thresholds ─────────────────────────────────────────────
QUALITY_THRESHOLDS = {
    "sharpness_min": 100.0,
    "brightness_min": 40,
    "brightness_max": 220,
    "accept_threshold": 0.70,
    "borderline_threshold": 0.40,
}

# ── Grading ───────────────────────────────────────────────────────────────────
REFERRAL_THRESHOLD = 2        # ICDR Level 2+ = refer
CONFIDENCE_LOW_FLAG = 0.65    # Below this → flag for human review
TEMPERATURE_SCALING = 1.5     # Calibration temperature

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
