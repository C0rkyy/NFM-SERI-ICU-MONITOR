"""
NFM — Connectivity Feature Extraction
========================================
Computes pairwise inter-channel coherence and derives
a scalar *Global Connectivity Index* (GCI) per epoch.

Coherence is estimated in the alpha (8–13 Hz) and beta (13–30 Hz)
bands using Welch's method, then averaged to a single matrix.

GCI = mean of the upper triangle of the coherence matrix
      (excluding the diagonal).
"""

import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.signal import coherence as _coherence
import mne

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FREQ_BANDS

logger = logging.getLogger(__name__)


def _pairwise_coherence(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float] = (8.0, 30.0),
) -> np.ndarray:
    """
    Compute pairwise magnitude-squared coherence for one epoch.

    Parameters
    ----------
    data : (n_channels, n_times)
    sfreq : sampling frequency
    band : frequency range over which to average coherence

    Returns
    -------
    coh_matrix : (n_channels, n_channels) symmetric, diagonal = 1
    """
    n_ch = data.shape[0]
    nperseg = min(data.shape[1], int(sfreq))
    coh_matrix = np.eye(n_ch)

    # Developer note:
    # O(n^2) pair loops are intentional here; n_ch is small in our current setup,
    # and this keeps the math transparent for clinical debugging.
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs, Cxy = _coherence(
                data[i], data[j],
                fs=sfreq, nperseg=nperseg,
            )
            idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
            mean_coh = Cxy[idx].mean() if len(idx) > 0 else 0.0
            coh_matrix[i, j] = mean_coh
            coh_matrix[j, i] = mean_coh

    return coh_matrix


def global_connectivity_index(coh_matrix: np.ndarray) -> float:
    """
    Scalar summary = mean of upper triangle (excluding self-coherence on diagonal).
    """
    n = coh_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(coh_matrix[iu].mean())


def extract_connectivity_features(
    epochs: mne.Epochs,
    band: Tuple[float, float] = (8.0, 30.0),
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute connectivity features for every epoch.

    Parameters
    ----------
    epochs : mne.Epochs
    band : frequency band for coherence estimation.

    Returns
    -------
    df : DataFrame with columns: epoch_idx, gci
    coh_matrices : ndarray (n_epochs, n_ch, n_ch)
    """
    data = epochs.get_data()  # (n_ep, n_ch, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs = data.shape[0]
    n_ch = data.shape[1]

    logger.info("Computing pairwise coherence for %d epochs "
                "(%d channels, band %.0f–%.0f Hz) …",
                n_epochs, n_ch, band[0], band[1])

    coh_matrices = np.zeros((n_epochs, n_ch, n_ch))
    rows: List[Dict] = []

    for i_ep in range(n_epochs):
        # One matrix per epoch lets us inspect temporal stability later.
        coh = _pairwise_coherence(data[i_ep], sfreq, band)
        coh_matrices[i_ep] = coh
        gci = global_connectivity_index(coh)
        rows.append({"epoch_idx": i_ep, "gci": gci})

    df = pd.DataFrame(rows)
    logger.info("Extracted connectivity features → %s", df.shape)
    return df, coh_matrices


def compute_average_coherence_matrix(
    coh_matrices: np.ndarray,
) -> np.ndarray:
    """Return the trial-averaged coherence matrix for visualisation."""
    return coh_matrices.mean(axis=0)
