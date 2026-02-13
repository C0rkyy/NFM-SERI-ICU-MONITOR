from __future__ import annotations

import json
from pathlib import Path

from dashboard.command_following import load_active_sessions
from dashboard.stimulus_response import load_stimulus_sessions


def _write_session(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stimulus_loader_filters_to_seri_sessions(tmp_path: Path) -> None:
    _write_session(
        tmp_path / "session_s001_active.json",
        {
            "session_id": "active_1",
            "mode": "active",
            "patient": {"subject_id": 1, "patient_id": "S001"},
            "scores": {"CFI": 66.0, "Confidence": 75.0, "Quality": 80.0},
            "feature_summary": {"trial_consistency": 0.7, "strongest_markers": []},
        },
    )
    _write_session(
        tmp_path / "session_s001_stimulus.json",
        {
            "session_id": "stimulus_1",
            "mode": "stimulus",
            "patient": {"subject_id": 1, "patient_id": "S001"},
            "scores": {"SERI": 58.0, "Confidence": 73.0, "Quality": 82.0},
            "feature_summary": {"trial_consistency": 0.6, "strongest_markers": []},
        },
    )

    df = load_stimulus_sessions(tmp_path)
    assert len(df) == 1
    assert set(df["session_id"].tolist()) == {"stimulus_1"}
    assert float(df.iloc[0]["SERI"]) == 58.0


def test_command_loader_filters_to_cfi_sessions(tmp_path: Path) -> None:
    _write_session(
        tmp_path / "session_s002_stimulus.json",
        {
            "session_id": "stimulus_2",
            "mode": "stimulus",
            "patient": {"subject_id": 2, "patient_id": "S002"},
            "scores": {"SERI": 61.0, "Confidence": 71.0, "Quality": 85.0},
            "feature_summary": {"trial_consistency": 0.8, "strongest_markers": []},
        },
    )
    _write_session(
        tmp_path / "session_s002_active.json",
        {
            "session_id": "active_2",
            "mode": "active",
            "patient": {"subject_id": 2, "patient_id": "S002"},
            "scores": {"CFI": 52.0, "Confidence": 69.0, "Quality": 78.0},
            "feature_summary": {"trial_consistency": 0.5, "strongest_markers": []},
        },
    )

    df = load_active_sessions(tmp_path)
    assert len(df) == 1
    assert set(df["session_id"].tolist()) == {"active_2"}
    assert float(df.iloc[0]["CFI"]) == 52.0

