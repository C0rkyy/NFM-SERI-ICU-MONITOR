"""
NFM — EDF Data Loader
======================
Responsible for:
  • Reading EDF files into MNE Raw objects
  • Extracting & mapping stimulus event markers
  • Building a subject-level metadata catalogue
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import mne

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SFREQ

logger = logging.getLogger(__name__)

# PhysioNet EEGBCI event mapping
# In the raw annotations: T0=rest, T1=left-fist, T2=right-fist
EVENT_ID_MAP = {"T0": 1, "T1": 2, "T2": 3}


def load_edf(filepath: Path, resample: float = SFREQ) -> mne.io.Raw:
    """
    Load a single EDF file and return an MNE Raw object.

    Parameters
    ----------
    filepath : Path
        Full path to the .edf file.
    resample : float
        Target sampling frequency (Hz).

    Returns
    -------
    mne.io.Raw
    """
    logger.info("Loading %s", filepath.name)
    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    
    # Standardise channel names to 10-20 system
    mne.datasets.eegbci.standardize(raw)
    
    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")
    
    # Resample
    if raw.info["sfreq"] != resample:
        raw.resample(resample, verbose=False)
        logger.info("  Resampled → %d Hz", resample)
    
    return raw


def extract_events(raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extract event markers from Raw annotations.

    Returns
    -------
    events : ndarray, shape (n_events, 3)
    event_id : dict
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    logger.info("  Found %d events with ids %s", len(events), event_id)
    return events, event_id


def load_subject(
    edf_paths: List[Path],
    subject_id: int,
) -> Dict[str, Any]:
    """
    Load all runs for a single subject and concatenate.

    Returns
    -------
    dict with keys:
        raw        : concatenated Raw
        events     : ndarray
        event_id   : dict
        metadata   : dict (subject_id, n_channels, sfreq, duration_s)
    """
    raws = []
    for fp in sorted(edf_paths):
        raws.append(load_edf(fp))

    raw = mne.concatenate_raws(raws)
    events, event_id = extract_events(raw)

    metadata = {
        "subject_id": subject_id,
        "n_channels": len(raw.ch_names),
        "sfreq": raw.info["sfreq"],
        "duration_s": raw.times[-1],
        "n_events": len(events),
        "files": [str(p.name) for p in edf_paths],
    }
    logger.info("Subject S%03d → %d ch, %.1f s, %d events",
                subject_id, metadata["n_channels"],
                metadata["duration_s"], metadata["n_events"])

    return {"raw": raw, "events": events, "event_id": event_id,
            "metadata": metadata}


def build_metadata_table(subjects: Dict[int, Dict]) -> pd.DataFrame:
    """
    Create a tidy DataFrame of per-subject metadata.
    """
    rows = [s["metadata"] for s in subjects.values()]
    return pd.DataFrame(rows).set_index("subject_id")


# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from downloader import download_physionet
    files = download_physionet(subjects=[1], runs=[3])
    data = load_subject(files[1], subject_id=1)
    print(data["metadata"])
