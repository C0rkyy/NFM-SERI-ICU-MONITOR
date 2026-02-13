"""
NFM — BrainVision LIS Data Loader
====================================
Robust ingestion of BrainVision (.vhdr/.vmrk/.eeg) files from
the LIS Zenodo dataset with hierarchical session structure:
  Subject (Pxx) / Visit (Vxx) / Day (Dxx) / SessionType (Training|Feedback)

Produces a normalised session registry for stimulus-evoked analysis.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from config import RESULTS_DIR

logger = logging.getLogger(__name__)

# ── Default LIS dataset root ──────────────────────────────────
_DEFAULT_LIS_ROOT = Path(__file__).resolve().parent / "external" / "lis_zenodo" / "Raw_Files"

# ── BrainVision marker pattern ────────────────────────────────
_MARKER_RE = re.compile(
    r"Mk\d+=(?P<type>\w[\w ]*),(?P<desc>[\w ]+),(?P<pos>\d+),(?P<size>\d+),(?P<ch>\d+)"
)

# Channel mapping: LIS custom names → approximate 10-20 equivalents
LIS_CHANNEL_RENAME: dict[str, str] = {
    "R1": "C4",   # right hemisphere channel 1 → C4 approximation
    "R2": "P4",   # right hemisphere channel 2 → P4 approximation
    "L1": "C3",   # left hemisphere channel 1 → C3 approximation
    "L2": "P3",   # left hemisphere channel 2 → P3 approximation
    "Cz": "Cz",
    "C1": "FC1",
    "C2": "FC2",
}

# EOG channel names to identify and optionally exclude
EOG_CHANNELS: set[str] = {"EOGD", "EOGL", "EOGR"}


def discover_brainvision_sessions(
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Walk the LIS directory tree and discover all .vhdr files with metadata.

    Returns a list of dicts, each describing one recording session.
    """
    root = Path(root) if root else _DEFAULT_LIS_ROOT
    if not root.exists():
        logger.warning("LIS dataset root not found: %s", root)
        return []

    sessions: list[dict[str, Any]] = []
    vhdr_files = sorted(root.rglob("*.vhdr"))
    logger.info("Discovered %d .vhdr files under %s", len(vhdr_files), root)

    for vhdr_path in vhdr_files:
        try:
            session_info = _parse_session_path(vhdr_path, root)
            sessions.append(session_info)
        except Exception as exc:
            logger.warning("Skipping %s: %s", vhdr_path, exc)

    return sessions


def _parse_session_path(vhdr_path: Path, root: Path) -> dict[str, Any]:
    """Extract hierarchical metadata from file path structure."""
    rel_parts = vhdr_path.relative_to(root).parts
    # Expected: Pxx / Vxx / Dxx / SessionType / filename.vhdr
    subject_id = "unknown"
    visit_id = "unknown"
    day_id = "unknown"
    session_type = "unknown"

    for part in rel_parts:
        upper = part.upper()
        if re.match(r"^P\d+$", upper):
            subject_id = upper
        elif re.match(r"^V\d+$", upper):
            visit_id = upper
        elif re.match(r"^D\d+$", upper):
            day_id = upper
        elif part.lower() in ("training", "feedback", "speller"):
            session_type = part.capitalize()

    # Verify companion files exist
    eeg_path = vhdr_path.with_suffix(".eeg")
    vmrk_path = vhdr_path.with_suffix(".vmrk")

    if not eeg_path.exists():
        raise FileNotFoundError(f"Missing .eeg companion: {eeg_path}")
    if not vmrk_path.exists():
        raise FileNotFoundError(f"Missing .vmrk companion: {vmrk_path}")

    return {
        "dataset_name": "lis_zenodo",
        "subject_id": subject_id,
        "visit_id": visit_id,
        "day_id": day_id,
        "session_type": session_type,
        "file_path": str(vhdr_path),
        "vmrk_path": str(vmrk_path),
        "eeg_path": str(eeg_path),
    }


def load_brainvision_raw(
    vhdr_path: Path | str,
    exclude_eog: bool = True,
    rename_channels: bool = True,
) -> mne.io.Raw:
    """Load a BrainVision .vhdr file into an MNE Raw object.

    Parameters
    ----------
    vhdr_path : path to .vhdr file
    exclude_eog : drop EOG channels (EOGD, EOGL, EOGR).
    rename_channels : map LIS channel names to standard 10-20 names.

    Returns
    -------
    mne.io.Raw  (preloaded)
    """
    vhdr_path = Path(vhdr_path)
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Drop EOG channels if requested
    if exclude_eog:
        eog_present = [ch for ch in raw.ch_names if ch in EOG_CHANNELS]
        if eog_present:
            raw.drop_channels(eog_present)
            logger.debug("Dropped EOG channels: %s", eog_present)

    # Rename LIS channels to standard 10-20
    if rename_channels:
        rename_map = {k: v for k, v in LIS_CHANNEL_RENAME.items() if k in raw.ch_names}
        if rename_map:
            raw.rename_channels(rename_map)

    # Set channel types for remaining channels
    eeg_channels = [ch for ch in raw.ch_names if ch not in EOG_CHANNELS]
    if eeg_channels:
        raw.set_channel_types({ch: "eeg" for ch in eeg_channels})

    return raw


def extract_brainvision_events(
    raw: mne.io.Raw,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract stimulus events from a BrainVision raw object.

    Returns
    -------
    events : (n_events, 3) array
    event_id : dict mapping description strings to integer codes
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Filter to keep only stimulus markers (S xx patterns)
    # MNE BrainVision annotations come as "Stimulus/S  1", "Stimulus/S  4", etc.
    stim_event_id: dict[str, int] = {}
    for desc, code in event_id.items():
        cleaned = str(desc).strip()
        # Strip "Stimulus/" prefix if present
        core = cleaned
        if "/" in core:
            core = core.split("/", 1)[1].strip()
        if re.match(r"^S\s*\d+$", core):
            stim_event_id[cleaned] = code

    if not stim_event_id:
        # Fallback: use all events
        logger.warning("No 'S xx' markers found; using all %d events", len(event_id))
        stim_event_id = event_id

    # Filter events array to only stimulus codes
    stim_codes = set(stim_event_id.values())
    mask = np.isin(events[:, 2], list(stim_codes))
    stim_events = events[mask]

    return stim_events, stim_event_id


def parse_vmrk_markers(vmrk_path: Path | str) -> pd.DataFrame:
    """Parse a .vmrk file directly for detailed marker information.

    Returns a DataFrame with columns:
      marker_num, marker_type, description, position, size, channel
    """
    vmrk_path = Path(vmrk_path)
    rows: list[dict[str, Any]] = []

    in_marker_section = False
    for line in vmrk_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("[Marker Infos]"):
            in_marker_section = True
            continue
        if line.startswith("[") and in_marker_section:
            break
        if not in_marker_section or not line or line.startswith(";"):
            continue

        # Parse Mkxx=Type,Description,Position,Size,Channel,...
        match = re.match(r"Mk(\d+)=(.+)", line)
        if not match:
            continue

        marker_num = int(match.group(1))
        parts = match.group(2).split(",")
        if len(parts) < 5:
            continue

        rows.append({
            "marker_num": marker_num,
            "marker_type": parts[0].strip(),
            "description": parts[1].strip(),
            "position": int(parts[2]) if parts[2].strip().isdigit() else 0,
            "size": int(parts[3]) if parts[3].strip().isdigit() else 0,
            "channel": int(parts[4]) if parts[4].strip().isdigit() else 0,
        })

    return pd.DataFrame(rows)


def parse_senlist_files(session_dir: Path) -> list[dict[str, Any]]:
    """Parse Block_x_senlist.txt sidecar files for stimulus metadata.

    Each senlist file has alternating lines: wav_filename, label (0 or 1).
    Returns list of trial dicts with block_id, trial_idx, wav_file, label.
    """
    trials: list[dict[str, Any]] = []
    senlist_files = sorted(session_dir.glob("Block_*_senlist.txt"))

    for sf in senlist_files:
        # Extract block number
        block_match = re.search(r"Block_(\d+)", sf.name)
        block_id = int(block_match.group(1)) if block_match else 0

        lines = sf.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        trial_idx = 0
        i = 0
        while i < len(lines) - 1:
            wav_file = lines[i].strip()
            label_str = lines[i + 1].strip()
            try:
                label = int(label_str)
            except ValueError:
                label = -1
            trials.append({
                "block_id": block_id,
                "trial_idx": trial_idx,
                "wav_file": wav_file,
                "stimulus_label": label,
                "stimulus_type": "auditory",  # wav files = auditory
            })
            trial_idx += 1
            i += 2

    return trials


def build_stimulus_dataset_registry(
    root: Path | str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build a normalised session registry from the LIS datasets.

    Columns: dataset_name, subject_id, visit_id, day_id, session_type,
             file_path, sfreq, n_channels, duration_s, marker_summary.

    Saves to: outputs/results/stimulus_dataset_registry.csv
    """
    sessions = discover_brainvision_sessions(root)
    if not sessions:
        logger.warning("No BrainVision sessions found.")
        return pd.DataFrame()

    registry_rows: list[dict[str, Any]] = []

    for session_info in sessions:
        vhdr_path = Path(session_info["file_path"])
        try:
            raw = load_brainvision_raw(vhdr_path)
            events, event_id = extract_brainvision_events(raw)

            # Build marker summary
            marker_counts = Counter(events[:, 2])
            marker_summary_parts = []
            code_to_desc = {v: k for k, v in event_id.items()}
            for code, count in sorted(marker_counts.items()):
                desc = code_to_desc.get(code, f"code_{code}")
                marker_summary_parts.append(f"{desc}:{count}")
            marker_summary = "|".join(marker_summary_parts)

            registry_rows.append({
                "dataset_name": session_info["dataset_name"],
                "subject_id": session_info["subject_id"],
                "visit_id": session_info["visit_id"],
                "day_id": session_info["day_id"],
                "session_type": session_info["session_type"],
                "file_path": str(vhdr_path),
                "sfreq": float(raw.info["sfreq"]),
                "n_channels": len(raw.ch_names),
                "duration_s": float(raw.times[-1]),
                "marker_summary": marker_summary,
                "n_events": len(events),
            })

            raw.close()
            del raw

        except Exception as exc:
            logger.warning("Failed to load %s: %s", vhdr_path, exc)
            registry_rows.append({
                "dataset_name": session_info["dataset_name"],
                "subject_id": session_info["subject_id"],
                "visit_id": session_info["visit_id"],
                "day_id": session_info["day_id"],
                "session_type": session_info["session_type"],
                "file_path": str(vhdr_path),
                "sfreq": 0.0,
                "n_channels": 0,
                "duration_s": 0.0,
                "marker_summary": f"ERROR: {exc}",
                "n_events": 0,
            })

    df = pd.DataFrame(registry_rows)

    if save and not df.empty:
        save_path = RESULTS_DIR / "stimulus_dataset_registry.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved registry: %s (%d sessions)", save_path, len(df))

    return df
