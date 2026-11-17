"""Shared paths and settings for the whole project.

Data lives outside the repo (external drive). Override the location with the
WATKINS_ROOT environment variable, e.g. when running inside WSL2:
    export WATKINS_ROOT="/mnt/d/UNI&WORK/Watkins"
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- data (external drive, REPSOL's "Data/") ----------
DATA_ROOT = Path(os.environ.get("WATKINS_ROOT", r"D:\UNI&WORK\Watkins"))
RAW_DIR = DATA_ROOT / "raw"                  # original wavs: raw/<species>/<year>/<file>.wav (never edited)
ANNOT_DIR = DATA_ROOT / "Annotations"        # clean annotations + splits
SPEC_DIR = DATA_ROOT / "Spectrograms"        # Spectrograms/<category>/<file>.wav.norm.pt
OLD_DIR = DATA_ROOT / "old"                  # 2025 metadata/splits + index.csv (reference only)

MAIN_SPLIT_DIR = ANNOT_DIR                   # 44 species, clip-level split
GROUPED_SPLIT_DIR = ANNOT_DIR / "grouped_34"  # 34 species, split by recording

# ---------- outputs (in repo) ----------
MODELS_OUTPUT = PROJECT_ROOT / "Models_output"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"

# ---------- cleaning ----------
MIN_DURATION_S = 0.2       # drop clips shorter than this
LOW_SR_HZ = 4000           # flag (keep) clips below this sample rate
HIGH_SR_HZ = 96000         # flag (keep) clips above this sample rate
MIN_RECORDINGS_GROUPED = 3 # species needs >= 3 recordings for the grouped split

# ---------- spectrograms (same as 2025, but fixed FFT size) ----------
TARGET_SR = 44000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TARGET_WIDTH = 400                                   # frames (~4.65 s at 44 kHz / hop 512)
LOAD_SECONDS = TARGET_WIDTH * HOP_LENGTH / TARGET_SR + 0.5  # only the first ~5.2 s is needed

SEED = 42
