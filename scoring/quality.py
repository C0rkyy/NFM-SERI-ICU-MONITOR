from __future__ import annotations

from typing import Any

import mne
import numpy as np
from scipy.signal import welch


def estimate_line_noise_residual(
    raw: mne.io.Raw,
    notch_freqs: tuple[float, float] = (50.0, 60.0),
    bandwidth_hz: float = 1.0,
) -> float:
    """Estimate residual line-noise ratio after filtering (lower is better)."""
    data = raw.get_data().mean(axis=0)
    sfreq = float(raw.info["sfreq"])
    nperseg = min(len(data), int(max(sfreq, 64)))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)

    broad_mask = (freqs >= 1.0) & (freqs <= 40.0)
    broad_power = float(np.trapz(psd[broad_mask], freqs[broad_mask])) if broad_mask.any() else 1.0

    residual_power = 0.0
    for notch in notch_freqs:
        band_mask = (freqs >= notch - bandwidth_hz) & (freqs <= notch + bandwidth_hz)
        if band_mask.any():
            residual_power += float(np.trapz(psd[band_mask], freqs[band_mask]))

    ratio = residual_power / (broad_power + 1e-12)
    return float(np.clip(ratio, 0.0, 1.0))


def estimate_snr_proxy(
    epochs: mne.Epochs,
    baseline_window: tuple[float, float],
    imagery_window: tuple[float, float],
) -> float:
    """Compute a simple imagery-vs-baseline RMS ratio as SNR proxy."""
    data = epochs.get_data()
    times = epochs.times

    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    imagery_mask = (times >= imagery_window[0]) & (times <= imagery_window[1])

    baseline = data[:, :, baseline_mask]
    imagery = data[:, :, imagery_mask]

    baseline_rms = float(np.sqrt(np.mean(np.square(baseline))))
    imagery_rms = float(np.sqrt(np.mean(np.square(imagery))))
    return imagery_rms / (baseline_rms + 1e-12)


def compute_quality_score(
    epoch_drop_rate: float,
    bad_channel_fraction: float,
    line_noise_residual: float,
    snr_proxy: float,
) -> dict[str, Any]:
    """Compute quality score (0-100) from data quality indicators."""
    drop_component = (1.0 - np.clip(epoch_drop_rate, 0.0, 1.0)) * 100.0
    bad_channel_component = (1.0 - np.clip(bad_channel_fraction, 0.0, 1.0)) * 100.0
    line_noise_component = (1.0 - np.clip(line_noise_residual, 0.0, 1.0)) * 100.0

    snr_norm = np.clip((snr_proxy - 0.8) / 1.2, 0.0, 1.0)
    snr_component = snr_norm * 100.0

    quality_score = (
        0.35 * drop_component
        + 0.25 * bad_channel_component
        + 0.20 * line_noise_component
        + 0.20 * snr_component
    )

    return {
        "QualityScore": float(np.clip(quality_score, 0.0, 100.0)),
        "components": {
            "epoch_component": float(drop_component),
            "bad_channel_component": float(bad_channel_component),
            "line_noise_component": float(line_noise_component),
            "snr_component": float(snr_component),
        },
    }
