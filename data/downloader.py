"""
NFM — EDF Data Loader / Downloader
====================================
Responsible for:
  • Discovering EDF files from multiple local dataset roots
  • Downloading from PhysioNet via MNE when needed
  • Deduplicating by canonical path and subject/run key
  • Logging discovery summary

Reference
---------
Goldberger et al., PhysioBank, PhysioToolkit, and PhysioNet (2000).
Schalk et al., BCI2000 (2004).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import mne

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, LOCAL_DATASET_ROOTS, PHYSIONET_RUNS, PHYSIONET_SUBJECTS

logger = logging.getLogger(__name__)
logger.info("MNE version: %s", mne.__version__)

# Regex to parse S<subject>R<run>.edf filenames
_SUBJECT_RUN_RE = re.compile(r"S(\d{3})R(\d{2})\.edf$", re.IGNORECASE)


# ================================================================
# Public API
# ================================================================


def load_eeg_data(
    data_dir: Optional[Path] = None,
    dataset: str = "physionet",
    subjects: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
    dataset_roots: Optional[List[Path]] = None,
) -> Dict[int, List[Path]]:
    """
    Unified entry-point for obtaining EDF file paths.

    Parameters
    ----------
    data_dir : Path, optional
        Root directory for local EDF files (used when dataset="local").
    dataset : {"physionet", "local"}
        "physionet" → download via MNE;  "local" → scan local roots.
    subjects : list of int, optional
        1-based subject IDs to filter (None = all found).
    runs : list of int, optional
        Run numbers to filter (None = all found).
    dataset_roots : list of Path, optional
        Extra roots to search for local EDF files.

    Returns
    -------
    dict  {subject_id: [Path, …]}
    """
    if dataset == "physionet":
        return download_physionet(subjects=subjects, runs=runs)
    elif dataset == "local":
        extra_roots = list(dataset_roots or [])
        if data_dir is not None:
            extra_roots.append(data_dir)
        return discover_local_edf_files(
            extra_roots=extra_roots,
            subjects=subjects,
            runs=runs,
        )
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset!r}. Use 'physionet' or 'local'."
        )


def discover_local_edf_files(
    extra_roots: Optional[List[Path]] = None,
    subjects: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
) -> Dict[int, List[Path]]:
    """
    Discover EDF files from all configured local roots with deduplication.

    Searches LOCAL_DATASET_ROOTS (in priority order) plus any extra_roots.
    Deduplicates files by canonical absolute path AND by (subject, run) key
    so that even if the same subject/run exists in multiple directories,
    only the first (highest-priority) copy is used.

    Parameters
    ----------
    extra_roots : list of Path, optional
        Additional directories to search.
    subjects : list of int, optional
        Filter to these subject IDs only (1-based). None = all.
    runs : list of int, optional
        Filter to these run numbers only. None = all.

    Returns
    -------
    dict {subject_id: [Path, …]} sorted by subject and filename.
    """
    search_roots: list[Path] = list(LOCAL_DATASET_ROOTS)
    if extra_roots:
        for r in extra_roots:
            if r not in search_roots:
                search_roots.append(r)

    # Also add MNE home cache as fallback
    home_mne = Path.home() / "mne_data"
    if home_mne.exists() and home_mne not in search_roots:
        search_roots.append(home_mne)

    seen_canonical: set[str] = set()
    seen_subject_run: set[tuple[int, int]] = set()
    result: Dict[int, List[Path]] = {}
    subject_filter = set(subjects) if subjects else None
    run_filter = set(runs) if runs else None

    total_found = 0
    total_dedup = 0
    malformed_names: list[str] = []

    for root in search_roots:
        if not root.exists():
            logger.debug("Dataset root does not exist: %s", root)
            continue

        edf_files = sorted(root.rglob("*.edf"))
        logger.info("Scanning %s -> %d EDF file(s) found", root, len(edf_files))

        for fp in edf_files:
            total_found += 1
            # Canonical path dedup
            canon = str(fp.resolve())
            if canon in seen_canonical:
                total_dedup += 1
                continue
            seen_canonical.add(canon)

            # Parse subject/run from filename
            match = _SUBJECT_RUN_RE.search(fp.name)
            if not match:
                # Warn loudly for .edf that look like subject data but fail regex
                if "S" in fp.name.upper() and "R" in fp.name.upper():
                    malformed_names.append(fp.name)
                    logger.warning("Malformed EDF filename (skipped): %s", fp.name)
                else:
                    logger.debug("Skipping non-subject EDF file: %s", fp.name)
                continue

            subj_id = int(match.group(1))
            run_id = int(match.group(2))

            # Filter
            if subject_filter and subj_id not in subject_filter:
                continue
            if run_filter and run_id not in run_filter:
                continue

            # Subject/run dedup (keep first = highest priority root)
            key = (subj_id, run_id)
            if key in seen_subject_run:
                total_dedup += 1
                continue
            seen_subject_run.add(key)

            result.setdefault(subj_id, []).append(fp)

    # Sort file lists for each subject
    for subj_id in result:
        result[subj_id] = sorted(result[subj_id], key=lambda p: p.name)

    # Log summary
    n_subjects = len(result)
    n_files = sum(len(v) for v in result.values())
    runs_per_subject = [len(v) for v in result.values()]

    logger.info("=" * 60)
    logger.info("LOCAL EDF DISCOVERY SUMMARY")
    logger.info("  Roots searched: %d", len(search_roots))
    logger.info("  Total EDF files scanned: %d", total_found)
    logger.info("  Duplicates removed: %d", total_dedup)
    logger.info("  Malformed filenames: %d", len(malformed_names))
    logger.info("  Unique subjects: %d", n_subjects)
    logger.info("  Unique EDF files: %d", n_files)
    if runs_per_subject:
        logger.info(
            "  Runs per subject: min=%d, max=%d, median=%.0f",
            min(runs_per_subject),
            max(runs_per_subject),
            sorted(runs_per_subject)[len(runs_per_subject) // 2],
        )
        # Histogram of runs per subject
        from collections import Counter
        hist = Counter(runs_per_subject)
        for k in sorted(hist):
            logger.info("    %d run(s): %d subject(s)", k, hist[k])
    if malformed_names:
        logger.warning("  Malformed EDF filenames skipped: %s", malformed_names[:10])
    logger.info("=" * 60)

    return result


def download_physionet(
    subjects: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
) -> Dict[int, List[Path]]:
    """
    Download EDF files from PhysioNet EEG Motor Movement / Imagery Dataset.

    Parameters
    ----------
    subjects : list of int, optional
        1-based subject IDs (default: config.PHYSIONET_SUBJECTS).
    runs : list of int, optional
        Run numbers to fetch (default: config.PHYSIONET_RUNS).

    Returns
    -------
    dict  {subject_id: [Path, ...]}
    """
    subjects = subjects or PHYSIONET_SUBJECTS
    runs = runs or PHYSIONET_RUNS

    if not isinstance(subjects, (list, tuple)):
        subjects = [subjects]
    subjects = [int(s) for s in subjects]

    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    runs = [int(r) for r in runs]

    for s in subjects:
        if not (1 <= s <= 109):
            raise ValueError(f"Subject ID must be 1-109, got {s}")

    logger.info("Requesting %d subject(s) x %d run(s)", len(subjects), len(runs))

    downloaded: Dict[int, List[Path]] = {}

    for subj in subjects:
        logger.info("Downloading subject S%03d …", subj)
        try:
            raw_fnames = mne.datasets.eegbci.load_data(
                subj,
                runs,
                path=str(DATA_DIR),
                update_path=False,
            )
            paths = [Path(f) for f in raw_fnames]
            downloaded[subj] = paths

            for p in paths:
                logger.debug("  ok %s", p.name)
            logger.info("  -> %d file(s) for S%03d", len(paths), subj)

        except Exception as exc:
            logger.error("  FAILED for S%03d: %s", subj, exc)
            continue

    logger.info(
        "Download complete: %d / %d subjects succeeded.",
        len(downloaded),
        len(subjects),
    )
    return downloaded


# Backward-compatible alias
def get_local_edf_files(
    data_dir: Optional[Path] = None,
) -> Dict[int, List[Path]]:
    """Backward-compatible wrapper for discover_local_edf_files."""
    extra = [data_dir] if data_dir else []
    return discover_local_edf_files(extra_roots=extra)


# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"MNE version: {mne.__version__}")
    result = discover_local_edf_files()
    for sid, paths in sorted(result.items())[:3]:
        for p in paths:
            print(f"  S{sid:03d}: {p.name}")
