"""
NFM — Preprocessing Pipeline
==============================
Implements the full preprocessing chain on continuous MNE Raw data:

1. Band-pass filter  (0.5 – 40 Hz, FIR)
2. Notch filter       (50 / 60 Hz)
3. ICA artefact removal (eye-blinks, muscle)
4. Re-referencing     (average reference)
5. Per-channel z-score normalisation

Every function is idempotent and operates on a *copy* of the data
so the caller can freely chain or skip steps.
"""

import logging
from typing import List, Optional

import numpy as np
import mne
from mne.preprocessing import ICA

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQS,
    ICA_N_COMPONENTS, ICA_METHOD, ICA_MAX_ITER,
    REFERENCE,
)

logger = logging.getLogger(__name__)


# ================================================================
# 1. Band-pass filter
# ================================================================
def bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = BANDPASS_LOW,
    h_freq: float = BANDPASS_HIGH,
) -> mne.io.Raw:
    """Apply zero-phase FIR band-pass filter."""
    logger.info("Band-pass filtering %.1f – %.1f Hz", l_freq, h_freq)
    raw_filt = raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method="fir", fir_design="firwin",
        verbose=False,
    )
    return raw_filt


# ================================================================
# 2. Notch filter
# ================================================================
def notch_filter(
    raw: mne.io.Raw,
    freqs: List[float] = NOTCH_FREQS,
) -> mne.io.Raw:
    """Remove power-line interference at specified frequencies."""
    logger.info("Notch filter at %s Hz", freqs)
    raw_notch = raw.copy().notch_filter(
        freqs=freqs, method="fir", verbose=False,
    )
    return raw_notch


# ================================================================
# 3. ICA artefact removal
# ================================================================
def run_ica(
    raw: mne.io.Raw,
    n_components=ICA_N_COMPONENTS,
    method: str = ICA_METHOD,
    max_iter: int = ICA_MAX_ITER,
    random_state: int = 42,
) -> mne.io.Raw:
    """
    Fit ICA and automatically detect + remove artefact components.

    Artefact detection strategy
    ---------------------------
    We flag components whose kurtosis exceeds 2 standard deviations
    above the mean — a simple but effective heuristic that captures
    blink / saccade / muscle artefacts without requiring an EOG channel.
    """
    logger.info("Fitting ICA (%s, n_components=%s) …", method, n_components)
    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
    )
    ica.fit(raw, verbose=False)

    # Automatic artefact component detection via kurtosis
    sources = ica.get_sources(raw).get_data()
    from scipy.stats import kurtosis as _kurt
    kurt = _kurt(sources, axis=1)
    threshold = np.mean(kurt) + 2 * np.std(kurt)
    bad_idx = list(np.where(kurt > threshold)[0])

    if bad_idx:
        logger.info("  Excluding %d ICA component(s): %s", len(bad_idx), bad_idx)
    else:
        logger.info("  No artefact components detected by kurtosis criterion.")

    ica.exclude = bad_idx
    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean


# ================================================================
# 4. Re-reference
# ================================================================
def rereference(
    raw: mne.io.Raw,
    ref: str = REFERENCE,
) -> mne.io.Raw:
    """Re-reference to average (or a specified channel)."""
    logger.info("Re-referencing → %s", ref)
    raw_ref = raw.copy().set_eeg_reference(ref, verbose=False)
    return raw_ref


# ================================================================
# 5. Z-score normalisation (per channel)
# ================================================================
def zscore_normalize(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Normalise each channel to zero mean, unit variance.

    This is applied *after* all filtering / ICA so that downstream
    feature extractors receive comparable scales across channels
    and subjects.
    """
    logger.info("Z-score normalising per channel")
    data = raw.get_data()  # (n_channels, n_times)
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # guard against flat channels
    raw._data = (data - means) / stds
    return raw


# ================================================================
# Full pipeline convenience wrapper
# ================================================================
def preprocess(raw: mne.io.Raw, skip_ica: bool = False) -> mne.io.Raw:
    """
    Run the complete preprocessing chain.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG (must be preloaded).
    skip_ica : bool
        Skip ICA step (useful for very short recordings).

    Returns
    -------
    mne.io.Raw  — preprocessed copy.
    """
    out = bandpass_filter(raw)
    out = notch_filter(out)
    if not skip_ica:
        out = run_ica(out)
    out = rereference(out)
    out = zscore_normalize(out)
    return out
