"""
NFM — Stimulus-Locked Epoching
=================================
Epoch extraction for stimulus-evoked response analysis.

- baseline window default: [-0.2, 0.0] s
- response window default: [0.0, 0.8] s
- optional extended window for spectral modulation: [0.0, 2.0] s

Returns epoch objects, metadata per trial, and artifact/drop statistics.
"""
from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Developer note:
# these defaults mirror common ERP practice and keep enough pre-stimulus context
# for baseline correction without overextending noisy tails.
DEFAULT_BASELINE_WINDOW: tuple[float, float] = (-0.2, 0.0)
DEFAULT_RESPONSE_WINDOW: tuple[float, float] = (0.0, 0.8)
DEFAULT_EXTENDED_WINDOW: tuple[float, float] = (0.0, 2.0)

# Default reject threshold assumes MNE data in volts.
DEFAULT_REJECT: dict[str, float] = {"eeg": 150e-6}


def extract_stimulus_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    reject: dict[str, float] | None = None,
    flat: dict[str, float] | None = None,
) -> tuple[mne.Epochs, pd.DataFrame, dict[str, Any]]:
    """Extract stimulus-locked epochs for evoked response analysis.

    Parameters
    ----------
    raw : preprocessed MNE Raw object
    events : (n_events, 3) stimulus event array
    event_id : event description → code mapping
    baseline_window : (tmin, tmax) for baseline correction, default (-0.2, 0.0)
    response_window : (tmin, tmax) for response period, default (0.0, 0.8)
    reject : artifact rejection thresholds
    flat : flatline rejection thresholds

    Returns
    -------
    epochs : mne.Epochs object
    metadata : DataFrame with per-trial info
    stats : dict with epoch_drop_rate, bad_channel_fraction, etc.
    """
    if events.shape[0] == 0:
        raise ValueError("No stimulus events provided for epoching.")

    tmin = float(baseline_window[0])
    tmax = float(response_window[1])
    baseline = (float(baseline_window[0]), float(baseline_window[1]))

    n_requested = len(events)

    # Guard against unit mismatches. Some exports arrive in microvolts, not volts.
    if reject is None:
        # Check if data is in volts (typical for BrainVision)
        data_sample = raw.get_data(start=0, stop=min(100, raw.n_times))
        max_amp = np.max(np.abs(data_sample))
        if max_amp > 1.0:
            # Data likely in microvolts, adjust threshold
            reject = {"eeg": 150.0}
        else:
            reject = DEFAULT_REJECT

    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            flat=flat,
            preload=True,
            verbose=False,
            on_missing="warn",
        )
    except Exception:
        # Fallback: no rejection
        logger.warning("Epoch extraction with rejection failed; retrying without rejection")
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=None,
            flat=None,
            preload=True,
            verbose=False,
            on_missing="warn",
        )

    # Build lightweight metadata table so trial-level features can be joined safely.
    code_to_desc = {v: k for k, v in event_id.items()}
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(epochs.events):
        code = int(event[2])
        desc = code_to_desc.get(code, f"code_{code}")
        rows.append({
            "epoch_idx": idx,
            "event_code": code,
            "description": desc,
            "sample": int(event[0]),
            "time_s": float(event[0]) / raw.info["sfreq"],
        })
    metadata = pd.DataFrame(rows)

    # Keep these stats explicit; they feed quality scoring and clinical confidence.
    bad_ch = len(raw.info.get("bads", []))
    total_ch = max(len(raw.ch_names), 1)
    n_kept = len(epochs)
    drop_rate = 1.0 - (n_kept / max(n_requested, 1))

    stats: dict[str, Any] = {
        "n_events_requested": n_requested,
        "n_epochs_kept": n_kept,
        "epoch_drop_rate": float(np.clip(drop_rate, 0.0, 1.0)),
        "bad_channel_fraction": float(np.clip(bad_ch / total_ch, 0.0, 1.0)),
        "sfreq": float(raw.info["sfreq"]),
        "tmin": tmin,
        "tmax": tmax,
        "baseline_window": list(baseline_window),
        "response_window": list(response_window),
    }

    logger.info(
        "Stimulus epochs: kept=%d/%d (drop=%.1f%%), channels=%d (bad=%d)",
        n_kept, n_requested, drop_rate * 100, total_ch, bad_ch,
    )

    return epochs, metadata, stats


def extract_extended_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    extended_window: tuple[float, float] = DEFAULT_EXTENDED_WINDOW,
) -> mne.Epochs | None:
    """Extract wider epochs for spectral/time-frequency analysis.

    Uses extended_window (default 0.0 to 2.0s) for frequency decomposition.
    Returns None if extraction fails.
    """
    tmin = float(baseline_window[0])
    tmax = float(extended_window[1])
    baseline = (float(baseline_window[0]), float(baseline_window[1]))

    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=None,
            preload=True,
            verbose=False,
            on_missing="warn",
        )
        return epochs
    except Exception as exc:
        logger.warning("Extended epoch extraction failed: %s", exc)
        return None


def separate_baseline_response(
    epochs: mne.Epochs,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each epoch into baseline and post-stimulus response segments.

    Returns
    -------
    baseline_data : (n_epochs, n_channels, n_baseline_samples)
    response_data : (n_epochs, n_channels, n_response_samples)
    """
    times = epochs.times
    data = epochs.get_data()

    bl_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    resp_mask = (times >= response_window[0]) & (times <= response_window[1])

    baseline_data = data[:, :, bl_mask]
    response_data = data[:, :, resp_mask]

    return baseline_data, response_data
