from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from config import ACTIVE_BASELINE_WINDOW, ACTIVE_IMAGERY_WINDOW, BASELINE, EPOCH_TMAX, EPOCH_TMIN

logger = logging.getLogger(__name__)

EEGBCI_COMMAND_MARKER_MAP: dict[str, str] = {
    "REST": "T0",
    "IMAGINE_HAND": "T1",
    "IMAGINE_WALKING": "T2",
}


def extract_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    reject: Optional[Dict[str, float]] = None,
) -> mne.Epochs:
    """Extract passive stimulus-locked epochs with baseline correction."""
    stim_event_id = {k: v for k, v in event_id.items() if k != "T0"}
    if not stim_event_id:
        stim_event_id = event_id

    logger.info("Epoching [%.3f, %.3f] s for passive flow", tmin, tmax)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=stim_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False,
    )
    return epochs


def separate_baseline_stimulus(epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
    """Split each epoch into baseline and post-stimulus segments."""
    times = epochs.times
    zero_idx = int(np.argmin(np.abs(times - 0.0)))
    data = epochs.get_data()
    baseline_data = data[:, :, :zero_idx]
    stimulus_data = data[:, :, zero_idx:]
    return baseline_data, stimulus_data


def get_baseline_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = EPOCH_TMIN,
    tmax: float = 0.0,
) -> mne.Epochs:
    """Extract passive baseline epochs."""
    rest_event_id = {k: v for k, v in event_id.items() if k == "T0"}
    if not rest_event_id:
        rest_event_id = event_id

    return mne.Epochs(
        raw,
        events,
        event_id=rest_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )


def resolve_command_event_id(
    event_id: dict[str, int],
    marker_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """Resolve command labels to event codes using explicit or fallback marker maps."""
    mapping = marker_map or EEGBCI_COMMAND_MARKER_MAP
    resolved: dict[str, int] = {}
    for command_label, marker in mapping.items():
        if marker in event_id:
            resolved[command_label] = int(event_id[marker])

    if not resolved:
        logger.warning("No command markers found with map=%s", mapping)
    return resolved


def _drop_reason_counts(drop_log: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for reasons in drop_log:
        for reason in reasons:
            if reason:
                counts[reason] += 1
    return dict(counts)


def extract_command_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    baseline_window: tuple[float, float] = ACTIVE_BASELINE_WINDOW,
    imagery_window: tuple[float, float] = ACTIVE_IMAGERY_WINDOW,
    marker_map: dict[str, str] | None = None,
    reject: dict[str, float] | None = None,
) -> tuple[mne.Epochs, pd.DataFrame, dict[str, Any]]:
    """Extract command-locked epochs and artifact/quality metadata."""
    resolved_event_id = resolve_command_event_id(event_id=event_id, marker_map=marker_map)
    if not resolved_event_id:
        raise ValueError("No command events found for active mode.")

    tmin = float(baseline_window[0])
    tmax = float(imagery_window[1])
    baseline = (float(baseline_window[0]), float(baseline_window[1]))

    selected_codes = set(resolved_event_id.values())
    n_requested = int(sum(1 for ev in events if int(ev[2]) in selected_codes))

    epochs = mne.Epochs(
        raw,
        events,
        event_id=resolved_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False,
    )

    code_to_label = {code: label for label, code in resolved_event_id.items()}
    rows: list[dict[str, Any]] = []
    for epoch_idx, event in enumerate(epochs.events):
        event_code = int(event[2])
        command_label = code_to_label.get(event_code, "UNKNOWN")
        rows.append(
            {
                "epoch_idx": epoch_idx,
                "event_code": event_code,
                "command_label": command_label,
                "is_imagery": int(command_label != "REST"),
                "task_label": command_label.replace("IMAGINE_", "") if command_label != "REST" else "REST",
            }
        )

    metadata = pd.DataFrame(rows)

    bad_ch = len(raw.info.get("bads", []))
    total_ch = max(len(raw.ch_names), 1)
    n_kept = int(len(epochs))
    drop_rate = 1.0 - (n_kept / n_requested) if n_requested > 0 else 1.0

    stats: dict[str, Any] = {
        "n_events_requested": n_requested,
        "n_epochs_kept": n_kept,
        "epoch_drop_rate": float(np.clip(drop_rate, 0.0, 1.0)),
        "bad_channel_fraction": float(np.clip(bad_ch / total_ch, 0.0, 1.0)),
        "drop_reason_counts": _drop_reason_counts(epochs.drop_log),
        "available_commands": sorted(metadata["command_label"].unique().tolist()) if len(metadata) else [],
        "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
        "imagery_window": [float(imagery_window[0]), float(imagery_window[1])],
    }

    logger.info(
        "Command epochs kept=%d requested=%d drop_rate=%.3f",
        n_kept,
        n_requested,
        stats["epoch_drop_rate"],
    )
    return epochs, metadata, stats
