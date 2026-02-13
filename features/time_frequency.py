"""
NFM — Time-Frequency Feature Extraction
==========================================
Computes trial-level time-frequency representations using
complex Morlet wavelets via MNE's ``Epochs.compute_tfr()`` API.

Wavelet sizing constraint
-------------------------
For an epoch of duration T seconds sampled at fs Hz, a Morlet
wavelet at frequency f with n_cycles cycles has temporal extent
≈ n_cycles / f seconds.  To avoid the "wavelet longer than signal"
error we:
  1. Set the minimum frequency so that even the widest wavelet
     fits inside the epoch.
  2. Use frequency-adaptive cycles: n_cycles = freqs / 2,
     clamped to [2, 7], which gives shorter wavelets at low
     frequencies and sharper resolution at high frequencies.

Primary output:
    • Average induced power change relative to baseline
    • Full TFR array for visualisation
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import mne

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WAVELET_FREQS_MAX,
    WAVELET_N_FREQS,
)

logger = logging.getLogger(__name__)


def _safe_freqs_and_cycles(
    epochs: mne.Epochs,
    fmin_requested: float = 2.0,
    fmax: float = WAVELET_FREQS_MAX,
    n_freqs: int = WAVELET_N_FREQS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build frequency and n_cycles vectors that are guaranteed to
    produce wavelets shorter than the epoch.

    Returns
    -------
    freqs    : (n_freqs,)
    n_cycles : (n_freqs,)  — frequency-adaptive
    """
    n_times = len(epochs.times)
    sfreq = epochs.info["sfreq"]

    # Adaptive cycles: freqs / 2, clamped to [2, 7]
    freqs = np.linspace(fmin_requested, fmax, n_freqs)
    n_cycles = np.clip(freqs / 2.0, 2.0, 7.0)

    # Safety check: wavelet_length_samples ≈ n_cycles / freq * sfreq
    # MNE internally uses ≈ 5 * sigma * sfreq with sigma = n_cycles / (2π·freq)
    # Conservative bound: 2 * n_cycles * sfreq / freq
    max_wavelet_samples = 2.0 * n_cycles / freqs * sfreq

    # Remove any freq where the wavelet would exceed epoch length
    keep = max_wavelet_samples < n_times
    if not keep.all():
        n_dropped = (~keep).sum()
        logger.warning("Dropping %d low frequencies whose wavelets "
                       "exceed epoch length (%d samples).", n_dropped, n_times)
        freqs = freqs[keep]
        n_cycles = n_cycles[keep]

    if len(freqs) == 0:
        raise RuntimeError(
            f"No wavelet fits inside {n_times} samples at {sfreq} Hz. "
            f"Increase epoch duration or raise fmin."
        )

    return freqs, n_cycles


def compute_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray | None = None,
    n_cycles: np.ndarray | float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute time-frequency representation via Morlet wavelets.

    Uses the modern ``Epochs.compute_tfr()`` API (MNE ≥ 1.6).

    Returns
    -------
    tfr_data : ndarray (n_epochs, n_channels, n_freqs, n_times)
    freqs    : ndarray (n_freqs,)
    times    : ndarray (n_times,)
    """
    if freqs is None or n_cycles is None:
        freqs, n_cycles = _safe_freqs_and_cycles(epochs)

    logger.info("Computing Morlet TFR (%d freqs, %.1f–%.1f Hz) …",
                len(freqs), freqs[0], freqs[-1])

    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False,
        verbose=False,
    )

    tfr_data = tfr.get_data()   # (n_epochs, n_ch, n_freqs, n_times)
    times = epochs.times

    return tfr_data, freqs, times


def extract_tf_features(
    epochs: mne.Epochs,
    average_channels: bool = True,
) -> pd.DataFrame:
    """
    Extract average induced power change per epoch.

    Baseline normalisation:
        relative change = (power − baseline_power) / baseline_power
    where baseline = mean power in the pre-stimulus window (times < 0).

    Returns
    -------
    DataFrame: epoch_idx, avg_induced_power_change
    """
    tfr_data, freqs, times = compute_tfr(epochs)
    # tfr_data: (n_epochs, n_ch, n_freqs, n_times)

    baseline_mask = times < 0
    stim_mask = times >= 0

    bl_power = tfr_data[..., baseline_mask].mean(axis=-1, keepdims=True)  # keep for broadcasting
    bl_power = np.clip(bl_power, 1e-15, None)  # avoid division by zero

    # Relative change in the stimulus window
    stim_power = tfr_data[..., stim_mask].mean(axis=-1)  # (n_ep, n_ch, n_freq)
    rel_change = (stim_power - bl_power.squeeze(-1)) / bl_power.squeeze(-1)

    rows = []
    for i_ep in range(tfr_data.shape[0]):
        val = rel_change[i_ep]  # (n_ch, n_freq)
        if average_channels:
            rows.append({
                "epoch_idx": i_ep,
                "avg_induced_power_change": float(val.mean()),
            })
        else:
            for i_ch, ch in enumerate(epochs.ch_names):
                rows.append({
                    "epoch_idx": i_ep,
                    "channel": ch,
                    "avg_induced_power_change": float(val[i_ch].mean()),
                })

    df = pd.DataFrame(rows)
    logger.info("Extracted TF features → %s", df.shape)
    return df


def get_average_tfr(
    epochs: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the trial-averaged TFR for visualisation.

    Returns
    -------
    avg_power : (n_ch, n_freqs, n_times)
    freqs, times
    """
    tfr_data, freqs, times = compute_tfr(epochs)
    return tfr_data.mean(axis=0), freqs, times
