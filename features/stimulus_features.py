"""
NFM — Stimulus-Evoked Response Features
==========================================
Replace MI-centric features with stimulus-centric features:

- ERP amplitudes/latencies (N100, P200, P300 where feasible)
- Baseline vs post-stimulus power change (delta/theta/alpha/beta)
- Trial-to-trial consistency / response stability
- Evoked SNR proxy
- Optional connectivity change in response window

Output:
  trial-level tidy dataframe
  session-level summary dataframe with strongest markers
"""
from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

logger = logging.getLogger(__name__)

# ── ERP component windows (seconds) ─────────────────────────
N100_WINDOW: tuple[float, float] = (0.080, 0.150)
P200_WINDOW: tuple[float, float] = (0.150, 0.280)
P300_WINDOW: tuple[float, float] = (0.250, 0.500)

# ── Frequency bands ─────────────────────────────────────────
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Default time windows
DEFAULT_BASELINE: tuple[float, float] = (-0.2, 0.0)
DEFAULT_RESPONSE: tuple[float, float] = (0.0, 0.8)


def _find_component(
    erp_waveform: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float],
    polarity: str = "negative",
) -> dict[str, float]:
    """Find ERP component amplitude and latency in a time window.

    Parameters
    ----------
    erp_waveform : 1D array, averaged across channels/trials as needed
    times : time axis in seconds
    window : (start, end) seconds
    polarity : 'negative' (N100) or 'positive' (P200, P300)

    Returns
    -------
    dict with amplitude, latency
    """
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        return {"amplitude": 0.0, "latency": 0.0}

    segment = erp_waveform[mask]
    segment_times = times[mask]

    # Developer note:
    # component polarity is explicit so we can reuse this helper for N/P components
    # without duplicating window logic.
    if polarity == "negative":
        idx = int(np.argmin(segment))
    else:
        idx = int(np.argmax(segment))

    return {
        "amplitude": float(segment[idx]),
        "latency": float(segment_times[idx]),
    }


def _bandpower(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
) -> float:
    """Compute average band power using Welch's method.

    data : 1D or 2D (channels x samples).
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]

    nperseg = min(data.shape[-1], int(sfreq))
    if nperseg < 4:
        return 0.0

    # Keep per-channel powers before averaging; makes it easier to debug outlier channels.
    powers: list[float] = []
    for ch_data in data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if band_mask.any():
            powers.append(float(np.trapz(psd[band_mask], freqs[band_mask])))
        else:
            powers.append(0.0)

    return float(np.mean(powers)) if powers else 0.0


def extract_stimulus_response_features(
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE,
    response_window: tuple[float, float] = DEFAULT_RESPONSE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract stimulus-evoked response features at trial level.

    Returns
    -------
    trial_df : trial-level tidy DataFrame
    summary_df : session-level summary (one row)
    summary_details : dict with strongest markers, consistency, etc.
    """
    # Developer note:
    # this function is intentionally trial-centric. We keep one row per epoch so
    # downstream quality, calibration, and plotting stay transparent.
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    n_epochs, n_channels, _ = data.shape

    # Precompute masks once; this keeps the per-trial loop lightweight.
    bl_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    resp_mask = (times >= response_window[0]) & (times <= response_window[1])

    trial_rows: list[dict[str, Any]] = []

    for i in range(n_epochs):
        epoch_data = data[i]  # (n_channels, n_times)

        # We aggregate channels here for robust global ERP estimates in noisy ICU-like data.
        avg_waveform = np.mean(epoch_data, axis=0)

        # ── ERP components ────────────────────────────────────
        n100 = _find_component(avg_waveform, times, N100_WINDOW, polarity="negative")
        p200 = _find_component(avg_waveform, times, P200_WINDOW, polarity="positive")
        p300 = _find_component(avg_waveform, times, P300_WINDOW, polarity="positive")

        # ── Baseline vs response power change per band ────────
        baseline_data = epoch_data[:, bl_mask]
        response_data = epoch_data[:, resp_mask]

        band_features: dict[str, float] = {}
        for band_name, band_range in FREQ_BANDS.items():
            bl_power = _bandpower(baseline_data, sfreq, band_range)
            resp_power = _bandpower(response_data, sfreq, band_range)

            # Relative power change (%) is easier to compare across sessions than raw power.
            if bl_power > 1e-15:
                pct_change = ((resp_power - bl_power) / bl_power) * 100.0
            else:
                pct_change = 0.0

            band_features[f"{band_name}_bl_power"] = bl_power
            band_features[f"{band_name}_resp_power"] = resp_power
            band_features[f"{band_name}_pct_change"] = float(np.clip(pct_change, -500.0, 500.0))

        # ── Evoked SNR proxy ──────────────────────────────────
        bl_rms = float(np.sqrt(np.mean(np.square(baseline_data)))) if bl_mask.any() else 1e-12
        resp_rms = float(np.sqrt(np.mean(np.square(response_data)))) if resp_mask.any() else 0.0
        evoked_snr = resp_rms / (bl_rms + 1e-12)

        # ── Peak-to-peak amplitude in response window ─────────
        resp_waveform = avg_waveform[resp_mask]
        peak_to_peak = float(np.ptp(resp_waveform)) if len(resp_waveform) > 0 else 0.0

        # ── Build trial row ───────────────────────────────────
        row: dict[str, Any] = {
            "epoch_idx": i,
            "n100_amplitude": n100["amplitude"],
            "n100_latency": n100["latency"],
            "p200_amplitude": p200["amplitude"],
            "p200_latency": p200["latency"],
            "p300_amplitude": p300["amplitude"],
            "p300_latency": p300["latency"],
            "evoked_snr": evoked_snr,
            "peak_to_peak": peak_to_peak,
            **band_features,
        }

        # Merge metadata if available
        if i < len(metadata):
            for col in metadata.columns:
                if col != "epoch_idx":
                    row[col] = metadata.iloc[i][col]

        trial_rows.append(row)

    trial_df = pd.DataFrame(trial_rows)

    # Trial consistency acts as a stability signal for confidence scoring.
    resp_amplitudes = trial_df["p300_amplitude"].values if "p300_amplitude" in trial_df.columns else np.array([])
    consistency = _compute_response_consistency(resp_amplitudes)

    # ── Session-level summary ─────────────────────────────────
    summary_details = _compute_summary_details(trial_df, consistency)
    summary_df = pd.DataFrame([{
        "n_trials": n_epochs,
        "mean_n100_amp": float(trial_df["n100_amplitude"].mean()) if len(trial_df) else 0.0,
        "mean_p200_amp": float(trial_df["p200_amplitude"].mean()) if len(trial_df) else 0.0,
        "mean_p300_amp": float(trial_df["p300_amplitude"].mean()) if len(trial_df) else 0.0,
        "mean_evoked_snr": float(trial_df["evoked_snr"].mean()) if len(trial_df) else 0.0,
        "response_consistency": consistency,
        "mean_alpha_change": float(trial_df["alpha_pct_change"].mean()) if "alpha_pct_change" in trial_df.columns else 0.0,
        "mean_beta_change": float(trial_df["beta_pct_change"].mean()) if "beta_pct_change" in trial_df.columns else 0.0,
    }])

    return trial_df, summary_df, summary_details


def _compute_response_consistency(amplitudes: np.ndarray) -> float:
    """Compute trial-to-trial response consistency in [0, 1].

    Uses coefficient of variation (lower CV = more consistent).
    """
    if len(amplitudes) < 2:
        return 0.0

    amplitudes = amplitudes[~np.isnan(amplitudes)]
    if len(amplitudes) < 2:
        return 0.0

    mean_amp = float(np.mean(np.abs(amplitudes)))
    if mean_amp < 1e-12:
        return 0.0

    cv = float(np.std(amplitudes)) / (mean_amp + 1e-12)
    # Map CV to [0, 1] consistency: lower CV → higher consistency
    consistency = float(np.clip(np.exp(-2.0 * cv), 0.0, 1.0))
    return consistency


def _compute_summary_details(
    trial_df: pd.DataFrame,
    consistency: float,
) -> dict[str, Any]:
    """Compute session-level summary with strongest markers."""
    strongest_markers: list[str] = []

    if trial_df.empty:
        return {
            "strongest_markers": [],
            "trial_consistency": consistency,
            "channels_available": [],
        }

    # Rank features by effect size (absolute mean / std or just absolute mean)
    feature_scores: list[tuple[str, float]] = []

    for col in ["n100_amplitude", "p200_amplitude", "p300_amplitude"]:
        if col in trial_df.columns:
            vals = trial_df[col].dropna()
            if len(vals) > 0:
                score = float(np.abs(vals.mean())) / (float(vals.std()) + 1e-12)
                feature_scores.append((col, score))

    for band in FREQ_BANDS:
        col = f"{band}_pct_change"
        if col in trial_df.columns:
            vals = trial_df[col].dropna()
            if len(vals) > 0:
                score = float(np.abs(vals.mean())) / (float(vals.std()) + 1e-12)
                feature_scores.append((col, score))

    if "evoked_snr" in trial_df.columns:
        vals = trial_df["evoked_snr"].dropna()
        if len(vals) > 0:
            score = float(vals.mean())
            feature_scores.append(("evoked_snr", score))

    # Sort by score descending, take top 5
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    strongest_markers = [f"{name} (effect={score:.2f})" for name, score in feature_scores[:5]]

    return {
        "strongest_markers": strongest_markers,
        "trial_consistency": consistency,
        "channels_available": [],
    }


def compute_grand_average_erp(epochs: mne.Epochs) -> np.ndarray:
    """Compute grand average ERP waveform (averaged across trials and channels)."""
    data = epochs.get_data()
    # Average across trials, then across channels
    avg = data.mean(axis=0).mean(axis=0)
    return avg
