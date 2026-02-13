"""
NFM — Spectral Feature Extraction
====================================
Computes power spectral density in canonical frequency bands and
derives relative band-power changes between baseline and stimulus
segments.

Bands
-----
Delta  0.5 – 4  Hz
Theta  4   – 8  Hz
Alpha  8   – 13 Hz
Beta   13  – 30 Hz
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import mne

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FREQ_BANDS

logger = logging.getLogger(__name__)


def _band_power(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Sum PSD values within a frequency band (inclusive).

    Parameters
    ----------
    psd   : (..., n_freqs)
    freqs : (n_freqs,)
    band  : (low, high) in Hz

    Returns
    -------
    ndarray same shape as psd with last axis collapsed.
    """
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def compute_psd(
    epochs: mne.Epochs,
    fmin: float = 0.5,
    fmax: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD for each epoch using Welch's method.

    Returns
    -------
    psd   : ndarray (n_epochs, n_channels, n_freqs)
    freqs : ndarray (n_freqs,)
    """
    spectrum = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd = spectrum.get_data()       # (n_epochs, n_ch, n_freqs)
    freqs = spectrum.freqs
    return psd, freqs


def extract_spectral_features(
    epochs: mne.Epochs,
    baseline_data: np.ndarray | None = None,
    average_channels: bool = True,
) -> pd.DataFrame:
    """
    Extract absolute & relative band power per epoch.

    If *baseline_data* is provided (n_epochs_bl, n_ch, n_times),
    compute relative change:  (stim_power − baseline_power) / baseline_power.
    Otherwise, report absolute power only.

    Returns
    -------
    DataFrame with columns per band: {band}_power, [{band}_relative_change]
    """
    psd, freqs = compute_psd(epochs)          # (n_ep, n_ch, n_freq)

    # Compute absolute band power
    band_powers: Dict[str, np.ndarray] = {}
    for band_name, (lo, hi) in FREQ_BANDS.items():
        bp = _band_power(psd, freqs, (lo, hi))  # (n_ep, n_ch)
        band_powers[band_name] = bp

    # Baseline band power (average across baseline epochs if given)
    baseline_bp: Dict[str, np.ndarray] | None = None
    if baseline_data is not None:
        # Build PSD from raw numpy baseline segments
        sfreq = epochs.info["sfreq"]
        from scipy.signal import welch as _welch
        nperseg = min(baseline_data.shape[-1], int(sfreq * 2))
        f_bl, psd_bl = _welch(baseline_data, fs=sfreq, nperseg=nperseg, axis=-1)
        psd_bl_mean = psd_bl.mean(axis=0)  # average across baseline epochs → (n_ch, n_freq)
        baseline_bp = {}
        for band_name, (lo, hi) in FREQ_BANDS.items():
            baseline_bp[band_name] = _band_power(
                psd_bl_mean[np.newaxis, ...], f_bl, (lo, hi)
            ).squeeze(0)  # (n_ch,)

    rows = []
    n_epochs = psd.shape[0]
    ch_names = epochs.ch_names

    for i_ep in range(n_epochs):
        row: Dict = {"epoch_idx": i_ep}
        for band_name in FREQ_BANDS:
            bp_epoch = band_powers[band_name][i_ep]  # (n_ch,)
            if average_channels:
                row[f"{band_name}_power"] = float(bp_epoch.mean())
            else:
                for i_ch, ch in enumerate(ch_names):
                    row[f"{band_name}_power_{ch}"] = float(bp_epoch[i_ch])

            if baseline_bp is not None:
                bl = baseline_bp[band_name]  # (n_ch,)
                rel = (bp_epoch - bl) / (bl + 1e-12)
                if average_channels:
                    row[f"{band_name}_rel_change"] = float(rel.mean())
                else:
                    for i_ch, ch in enumerate(ch_names):
                        row[f"{band_name}_rel_change_{ch}"] = float(rel[i_ch])
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Extracted spectral features → %s", df.shape)
    return df


def compute_band_power_summary(epochs: mne.Epochs) -> Dict[str, float]:
    """
    Quick summary: mean absolute power per band across all epochs & channels.
    """
    psd, freqs = compute_psd(epochs)
    summary = {}
    for band_name, (lo, hi) in FREQ_BANDS.items():
        bp = _band_power(psd, freqs, (lo, hi))
        summary[band_name] = float(bp.mean())
    return summary
