"""
NFM — ERP Feature Extraction
==============================
Computes classic event-related potential features per epoch:

• P300 peak amplitude   (within 250–500 ms post-stimulus)
• P300 peak latency     (time of peak in ms)
• Area under the curve  (trapezoidal integral of the ERP in the window)

All metrics are computed per channel; the caller decides whether to
average across channels or keep them separate.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import mne

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import P300_WINDOW

logger = logging.getLogger(__name__)


def _find_p300_peaks(
    epoch_data: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    window: tuple = P300_WINDOW,
) -> Dict[str, np.ndarray]:
    """
    For a single epoch (n_channels × n_times), find P300 features.

    Returns dict with arrays of length n_channels:
        peak_amplitude, peak_latency_ms, auc
    """
    t_start, t_end = window
    mask = (times >= t_start) & (times <= t_end)
    windowed = epoch_data[:, mask]
    windowed_times = times[mask]

    # Developer note:
    # we deliberately use max positive deflection here (classic P300 definition).
    # If you switch paradigms, this is one of the first places to revisit.
    peak_idx = np.argmax(windowed, axis=1)
    peak_amplitude = windowed[np.arange(len(peak_idx)), peak_idx]

    # Keep latency in ms so clinicians can read it directly.
    peak_latency_ms = windowed_times[peak_idx] * 1000.0

    # AUC is a robustness feature when peak timing jitters trial-to-trial.
    dt = 1.0 / sfreq
    auc = np.trapz(windowed, dx=dt, axis=1)

    return {
        "peak_amplitude": peak_amplitude,
        "peak_latency_ms": peak_latency_ms,
        "auc": auc,
    }


def extract_erp_features(
    epochs: mne.Epochs,
    window: tuple = P300_WINDOW,
    average_channels: bool = True,
) -> pd.DataFrame:
    """
    Extract ERP features for every epoch.

    Parameters
    ----------
    epochs : mne.Epochs
    window : (float, float)
        Time window in seconds for P300 detection.
    average_channels : bool
        If True, return channel-averaged scalar per epoch.
        If False, return one row per (epoch, channel).

    Returns
    -------
    pd.DataFrame with columns:
        epoch_idx, [channel], p300_amplitude, p300_latency_ms, p300_auc
    """
    # Developer note:
    # we keep the extraction loop explicit for readability/debugging.
    # Vectorization is possible, but this is easier to maintain.
    data = epochs.get_data()        # (n_epochs, n_ch, n_times)
    times = epochs.times
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    rows: List[Dict] = []

    for i_ep in range(data.shape[0]):
        feats = _find_p300_peaks(data[i_ep], times, sfreq, window)

        if average_channels:
            # Fast path used by most pipelines: one ERP row per epoch.
            rows.append({
                "epoch_idx": i_ep,
                "p300_amplitude": float(np.mean(feats["peak_amplitude"])),
                "p300_latency_ms": float(np.mean(feats["peak_latency_ms"])),
                "p300_auc": float(np.mean(feats["auc"])),
            })
        else:
            # Detailed path for channel-level analysis/ablation experiments.
            for i_ch, ch in enumerate(ch_names):
                rows.append({
                    "epoch_idx": i_ep,
                    "channel": ch,
                    "p300_amplitude": float(feats["peak_amplitude"][i_ch]),
                    "p300_latency_ms": float(feats["peak_latency_ms"][i_ch]),
                    "p300_auc": float(feats["auc"][i_ch]),
                })

    df = pd.DataFrame(rows)
    logger.info("Extracted ERP features → %s", df.shape)
    return df


def compute_grand_average_erp(epochs: mne.Epochs) -> np.ndarray:
    """
    Compute the grand-average ERP waveform across all epochs.

    Returns
    -------
    ndarray (n_channels, n_times)
    """
    return epochs.get_data().mean(axis=0)
