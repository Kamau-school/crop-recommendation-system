"""Shared settings used by every stage of the project (preprocessing, training, app)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "Crop_recommendation.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

RANDOM_STATE = 42   # fixed seed so every run gives the same split and results
TEST_SIZE = 0.20

# Artifacts saved by preprocess.py and reused by train.py and the Flask app
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
