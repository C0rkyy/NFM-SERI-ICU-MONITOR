"""Tests for prospective workflow simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.prospective import simulate_prospective_run


def _make_sessions(n: int = 15) -> pd.DataFrame:
    """Create synthetic sessions for prospective simulation."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "session_id": [f"s{i:03d}" for i in range(n)],
        "subject_id": list(range(n)),
        "n_trials": rng.randint(20, 80, n),
        "QualityScore": rng.uniform(40, 100, n),
        "response_probability": rng.beta(2, 5, n),
        "SERI": rng.uniform(20, 80, n),
    })


class TestSimulateProspectiveRun:
    def test_basic_output_shape(self):
        sessions = _make_sessions(15)
        result = simulate_prospective_run(sessions)
        assert "metrics" in result
        assert "ops_log" in result
        assert "invalid_sessions" in result

    def test_metrics_keys(self):
        sessions = _make_sessions(15)
        result = simulate_prospective_run(sessions)
        metrics = result["metrics"]
        assert "n_sessions" in metrics
        assert "n_valid" in metrics
        assert "n_invalid" in metrics
        assert metrics["n_sessions"] == 15
        assert metrics["n_valid"] + metrics["n_invalid"] == metrics["n_sessions"]

    def test_ops_log_is_dataframe(self):
        sessions = _make_sessions(10)
        result = simulate_prospective_run(sessions)
        ops_log = result["ops_log"]
        assert isinstance(ops_log, pd.DataFrame)
        assert len(ops_log) == 10

    def test_invalid_sessions_is_dataframe(self):
        sessions = _make_sessions(20)
        result = simulate_prospective_run(sessions)
        invalid = result["invalid_sessions"]
        assert isinstance(invalid, pd.DataFrame)
        # All invalid session IDs must come from the input
        all_ids = set(sessions["session_id"])
        if len(invalid) > 0 and "session_id" in invalid.columns:
            for sid in invalid["session_id"]:
                assert sid in all_ids

    def test_min_trials_filter(self):
        """Sessions with low trial count should be flagged as invalid."""
        sessions = _make_sessions(10)
        # We use groupby so n_trials = len(grp), not the column. 
        # Instead, create sessions that have only 1 row each (min_trials=10 will catch them).
        result = simulate_prospective_run(sessions, min_trials=200)
        # Each session has 1 row → n_trials=1, all below 200
        assert result["metrics"]["n_invalid"] == 10

    def test_all_invalid_by_quality(self):
        """When all sessions fail quality, metrics should reflect that."""
        sessions = _make_sessions(5)
        sessions["QualityScore"] = 0.0  # all below threshold
        result = simulate_prospective_run(sessions, min_quality=50.0)
        assert result["metrics"]["n_valid"] == 0
        assert result["metrics"]["n_invalid"] == 5

    def test_valid_column_in_log(self):
        sessions = _make_sessions(10)
        result = simulate_prospective_run(sessions)
        log = result["ops_log"]
        assert "valid" in log.columns
