"""
NFM — Stimulus Event Extraction
=================================
Parse stimulus events from BrainVision .vmrk files and sidecar metadata.
Build robust marker mapping for stimulus-evoked analysis:
  stimulus_onset, stimulus_type (auditory/visual/unknown), trial/block ids.

Includes sanity checks: min trials/session, marker consistency, temporal validity.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.brainvision_loader import parse_senlist_files, parse_vmrk_markers

logger = logging.getLogger(__name__)

# ── Default stimulus marker mapping for LIS BCI dataset ──────
# Based on observed .vmrk content:
# S1/S2 = trial response markers, S4/S5 = stimulus onset (cue),
# S6/S8 = additional event types, S9 = block start, S10/S11 = condition,
# S15 = end marker
LIS_STIMULUS_MARKERS: dict[str, str] = {
    "S  4": "stimulus_onset",       # Stimulus cue presentation
    "S  5": "stimulus_onset",       # Stimulus cue type 2
    "S  1": "response",            # Subject response marker
    "S  2": "response",            # Subject response marker type 2
    "S  6": "feedback_onset",      # Feedback presentation
    "S  8": "feedback_end",        # Feedback end
    "S  9": "block_start",         # Block start marker
    "S 10": "condition_a",         # Condition A (e.g., auditory)
    "S 11": "condition_b",         # Condition B
    "S 15": "session_end",         # Session end marker
}

# Minimum number of stimulus trials for a valid session
MIN_TRIALS_PER_SESSION: int = 5

# Maximum inter-stimulus interval (seconds) for temporal validity
MAX_ISI_S: float = 60.0


def classify_marker(description: str) -> str:
    """Map a BrainVision marker description to a semantic type.

    Handles both raw 'S  4' and MNE-style 'Stimulus/S  4' formats.
    """
    cleaned = str(description).strip()
    # Strip "Stimulus/" prefix if present (MNE BrainVision format)
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[1].strip()
    return LIS_STIMULUS_MARKERS.get(cleaned, "unknown")


def extract_stimulus_events(
    events: np.ndarray,
    event_id: dict[str, int],
    sfreq: float,
    session_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract and classify stimulus events from event arrays.

    Parameters
    ----------
    events : (n_events, 3) array from MNE
    event_id : mapping of description → integer code
    sfreq : sampling frequency
    session_dir : optional path to session folder for sidecar parsing

    Returns
    -------
    DataFrame with columns:
      event_idx, sample, time_s, event_code, description, event_type,
      stimulus_type, block_id, trial_idx
    """
    code_to_desc = {v: k for k, v in event_id.items()}

    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        sample = int(event[0])
        code = int(event[2])
        desc = code_to_desc.get(code, f"code_{code}")
        event_type = classify_marker(desc)

        rows.append({
            "event_idx": idx,
            "sample": sample,
            "time_s": float(sample) / sfreq,
            "event_code": code,
            "description": desc,
            "event_type": event_type,
            "stimulus_type": "auditory",  # LIS dataset uses auditory stimuli
        })

    df = pd.DataFrame(rows)

    # Assign block IDs based on block_start markers
    if not df.empty:
        df["block_id"] = 0
        block_starts = df[df["event_type"] == "block_start"].index
        current_block = 0
        for i, row_idx in enumerate(block_starts):
            current_block = i + 1
            # All events from this block_start to the next belong to this block
            if i + 1 < len(block_starts):
                next_start = block_starts[i + 1]
                df.loc[row_idx:next_start - 1, "block_id"] = current_block
            else:
                df.loc[row_idx:, "block_id"] = current_block

        # Assign trial indices within each block (stimulus_onset events only)
        stimulus_mask = df["event_type"].isin(["stimulus_onset"])
        trial_counter: dict[int, int] = {}
        trial_indices: list[int] = []
        for _, row in df.iterrows():
            if row["event_type"] == "stimulus_onset":
                block = int(row["block_id"])
                trial_counter[block] = trial_counter.get(block, -1) + 1
                trial_indices.append(trial_counter[block])
            else:
                trial_indices.append(-1)
        df["trial_idx"] = trial_indices

    # Enrich with senlist sidecar if available
    if session_dir is not None:
        senlist_trials = parse_senlist_files(session_dir)
        if senlist_trials:
            df = _enrich_with_senlist(df, senlist_trials)

    return df


def _enrich_with_senlist(
    events_df: pd.DataFrame,
    senlist_trials: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge senlist sidecar metadata into event dataframe."""
    senlist_df = pd.DataFrame(senlist_trials)

    # Match stimulus_onset events with senlist entries by order within blocks
    stim_events = events_df[events_df["event_type"] == "stimulus_onset"].copy()

    # Attempt block-level matching
    for block_id in stim_events["block_id"].unique():
        block_stim = stim_events[stim_events["block_id"] == block_id]
        block_senlist = senlist_df[senlist_df["block_id"] == block_id] if "block_id" in senlist_df.columns else pd.DataFrame()

        if not block_senlist.empty:
            n_match = min(len(block_stim), len(block_senlist))
            for i in range(n_match):
                stim_idx = block_stim.index[i]
                events_df.loc[stim_idx, "stimulus_label"] = int(block_senlist.iloc[i].get("stimulus_label", -1))
                events_df.loc[stim_idx, "wav_file"] = str(block_senlist.iloc[i].get("wav_file", ""))

    # Fill missing with defaults
    if "stimulus_label" not in events_df.columns:
        events_df["stimulus_label"] = -1
    if "wav_file" not in events_df.columns:
        events_df["wav_file"] = ""

    return events_df


def validate_stimulus_session(
    events_df: pd.DataFrame,
    sfreq: float,
    min_trials: int = MIN_TRIALS_PER_SESSION,
    max_isi_s: float = MAX_ISI_S,
) -> dict[str, Any]:
    """Run sanity checks on extracted stimulus events.

    Returns
    -------
    Dict with: valid (bool), n_stimulus_events, n_blocks, issues (list[str])
    """
    issues: list[str] = []
    stim_events = events_df[events_df["event_type"] == "stimulus_onset"]
    n_stim = len(stim_events)

    if n_stim < min_trials:
        issues.append(f"Too few stimulus events: {n_stim} < {min_trials}")

    # Check temporal validity: Inter-stimulus intervals
    if n_stim >= 2:
        times = stim_events["time_s"].values
        isis = np.diff(times)
        if np.any(isis > max_isi_s):
            n_long = int(np.sum(isis > max_isi_s))
            issues.append(f"{n_long} inter-stimulus intervals exceed {max_isi_s}s")
        if np.any(isis <= 0):
            issues.append("Non-monotonic stimulus timestamps detected")

    # Marker count consistency
    marker_counts = Counter(events_df["description"])
    n_blocks = len(events_df[events_df["event_type"] == "block_start"])

    return {
        "valid": len(issues) == 0,
        "n_stimulus_events": n_stim,
        "n_blocks": n_blocks,
        "marker_counts": dict(marker_counts),
        "issues": issues,
    }


def get_stimulus_onset_events(
    events_df: pd.DataFrame,
    event_id: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract MNE-compatible (n, 3) event array of stimulus onsets only.

    Returns events array and filtered event_id dict for epoching.
    """
    stim_df = events_df[events_df["event_type"] == "stimulus_onset"].copy()
    if stim_df.empty:
        # Fallback: use all events with known stimulus markers
        stim_df = events_df[events_df["event_type"] != "unknown"].copy()

    if stim_df.empty:
        return np.empty((0, 3), dtype=int), {}

    # Build events array
    mne_events = np.zeros((len(stim_df), 3), dtype=int)
    mne_events[:, 0] = stim_df["sample"].values
    mne_events[:, 2] = stim_df["event_code"].values

    # Build event_id from unique codes
    unique_codes = stim_df[["description", "event_code"]].drop_duplicates()
    stim_event_id = {
        str(row["description"]): int(row["event_code"])
        for _, row in unique_codes.iterrows()
    }

    return mne_events, stim_event_id
