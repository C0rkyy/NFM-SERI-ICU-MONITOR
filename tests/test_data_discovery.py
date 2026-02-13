"""Tests for local EDF dataset discovery and deduplication."""
from __future__ import annotations

import tempfile
from pathlib import Path

from data.downloader import discover_local_edf_files


def test_discover_returns_dict() -> None:
    """discover_local_edf_files always returns a dict even if no files found."""
    result = discover_local_edf_files(extra_roots=[Path(tempfile.gettempdir())])
    assert isinstance(result, dict)


def test_discover_from_uploaded_path() -> None:
    """Verify that the uploaded dataset root is included in search."""
    from config import LOCAL_DATASET_ROOTS, PROJECT_ROOT

    # Check that the project-level 'files' directory is included
    expected_root = PROJECT_ROOT / "files"
    assert expected_root in LOCAL_DATASET_ROOTS

    # Check specific dataset root
    dataset_root = PROJECT_ROOT / "eeg-motor-movementimagery-dataset-1.0.0" / "files"
    assert dataset_root in LOCAL_DATASET_ROOTS


def test_discover_deduplication() -> None:
    """If the same EDF file exists in multiple roots, only one copy is kept."""
    from config import LOCAL_DATASET_ROOTS

    # Just verify the function runs without error on current roots
    result = discover_local_edf_files()
    # Check deduplication: each (subject, run) should be unique
    seen: set[str] = set()
    for subj_id, paths in result.items():
        for p in paths:
            key = p.name  # e.g., S001R03.edf
            assert key not in seen, f"Duplicate file found: {key}"
            seen.add(key)


def test_subject_run_parsing() -> None:
    """Subject and run IDs are correctly parsed from discovered files."""
    result = discover_local_edf_files()
    for subj_id, paths in result.items():
        assert 1 <= subj_id <= 109, f"Invalid subject ID: {subj_id}"
        for p in paths:
            assert p.name.startswith(f"S{subj_id:03d}R"), f"Filename mismatch: {p.name}"


def test_subject_filter() -> None:
    """--subjects filter restricts discovery."""
    result_all = discover_local_edf_files()
    if not result_all:
        return  # No files locally
    first_subj = sorted(result_all.keys())[0]
    result_filtered = discover_local_edf_files(subjects=[first_subj])
    assert len(result_filtered) <= 1
    if result_filtered:
        assert first_subj in result_filtered
