from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Final

import numpy as np

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
MODEL_DIR: Final[Path] = OUTPUT_DIR / "models"
RESULTS_DIR: Final[Path] = OUTPUT_DIR / "results"
SESSION_STORE_DIR: Final[Path] = OUTPUT_DIR / "session_store"
FIGURES_DIR: Final[Path] = OUTPUT_DIR / "figures"
CACHE_DIR: Final[Path] = OUTPUT_DIR / "cache"

for _d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, SESSION_STORE_DIR, FIGURES_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Local dataset discovery roots (priority order) ─────────────
LOCAL_DATASET_ROOTS: Final[list[Path]] = [
    PROJECT_ROOT / "eeg-motor-movementimagery-dataset-1.0.0" / "files",
    PROJECT_ROOT / "files",
    DATA_DIR,
]

# Recording parameters
SFREQ: Final[int] = 160

# Preprocessing
BANDPASS_LOW: Final[float] = 0.5
BANDPASS_HIGH: Final[float] = 40.0
NOTCH_FREQS: Final[list[float]] = [50.0, 60.0]
ICA_N_COMPONENTS: Final[float] = 0.95
ICA_METHOD: Final[str] = "fastica"
ICA_MAX_ITER: Final[int] = 500
REFERENCE: Final[str] = "average"

# Epoch extraction (passive mode)
EPOCH_TMIN: Final[float] = -0.2
EPOCH_TMAX: Final[float] = 0.8
BASELINE: Final[tuple[float | None, float | None]] = (None, 0)

# Active task defaults
ACTIVE_COMMANDS: Final[tuple[str, str, str]] = (
    "REST",
    "IMAGINE_WALKING",
    "IMAGINE_HAND",
)
ACTIVE_BASELINE_WINDOW: Final[tuple[float, float]] = (-1.0, 0.0)
ACTIVE_IMAGERY_WINDOW: Final[tuple[float, float]] = (0.0, 4.0)
ACTIVE_CUE_DURATION_S: Final[float] = 2.0
ACTIVE_IMAGERY_DURATION_S: Final[float] = 4.0
ACTIVE_REST_DURATION_S: Final[float] = 3.0
ACTIVE_N_TRIALS: Final[int] = 20
ACTIVE_RANDOMIZED_ORDER: Final[bool] = True
ACTIVE_INTER_TRIAL_INTERVAL_S: Final[float] = 1.0
ACTIVE_REQUIRED_EVENT_MARKERS: Final[dict[str, str]] = {
    "IMAGINE_WALKING": "CMD_WALK",
    "IMAGINE_HAND": "CMD_HAND",
}
ACTIVE_COMPATIBILITY_EVENT_MARKERS: Final[dict[str, str]] = {
    "REST": "T0",
    "IMAGINE_HAND": "T1",
    "IMAGINE_WALKING": "T2",
}

# Feature extraction
P300_WINDOW: Final[tuple[float, float]] = (0.250, 0.500)
FREQ_BANDS: Final[dict[str, tuple[float, float]]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
WAVELET_FREQS_MIN: Final[float] = 1.0
WAVELET_FREQS_MAX: Final[float] = 40.0
WAVELET_N_FREQS: Final[int] = 30
WAVELET_N_CYCLES: Final[float] = 7.0

# Model training
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
RANDOM_STATE: Final[int] = 42
RF_N_ESTIMATORS: Final[int] = 300
RF_MAX_DEPTH: Final[int] = 12
CNN_EPOCHS: Final[int] = 50
CNN_BATCH_SIZE: Final[int] = 32
CNN_LEARNING_RATE: Final[float] = 1e-3

# FRS scoring
FRS_WEIGHTS: Final[dict[str, float]] = {
    "erp_amplitude": 0.30,
    "band_power_shift": 0.25,
    "model_probability": 0.30,
    "connectivity_index": 0.15,
}

# CFI interpretation thresholds
CFI_LOW_MAX: Final[float] = 39.999
CFI_MODERATE_MAX: Final[float] = 69.999
CFI_MIN_USABLE_TRIALS: Final[int] = 10

# PhysioNet download defaults – all 109 subjects for full dataset
PHYSIONET_SUBJECTS: Final[list[int]] = list(range(1, 110))
PHYSIONET_RUNS: Final[list[int]] = [3, 7, 11]

# ── LIS / Stimulus-evoked response configuration ─────────────
LIS_DATASET_ROOT: Final[Path] = PROJECT_ROOT / "data" / "external" / "lis_zenodo" / "Raw_Files"

# Stimulus-locked epoch windows
STIMULUS_BASELINE_WINDOW: Final[tuple[float, float]] = (-0.2, 0.0)
STIMULUS_RESPONSE_WINDOW: Final[tuple[float, float]] = (0.0, 0.8)
STIMULUS_EXTENDED_WINDOW: Final[tuple[float, float]] = (0.0, 2.0)

# SERI thresholds
SERI_LOW_MAX: Final[float] = 39.999
SERI_MODERATE_MAX: Final[float] = 69.999
SERI_MIN_USABLE_TRIALS: Final[int] = 5

# Parallelism — auto-detect available cores, leave 1 free for OS
import multiprocessing as _mp
DEFAULT_N_JOBS: Final[int] = max(1, _mp.cpu_count() - 1)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
