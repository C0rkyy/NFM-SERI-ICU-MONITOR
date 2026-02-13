"""Tests for model calibration metrics and confidence quality."""
from __future__ import annotations

import numpy as np
import pytest

from models.command_following import _compute_ece


def test_ece_perfect_calibration() -> None:
    """A perfectly calibrated model should have ECE close to 0."""
    rng = np.random.default_rng(42)
    n = 1000
    y_prob = rng.uniform(0, 1, n)
    # For perfect calibration, y_true ~ Bernoulli(y_prob)
    y_true = (rng.random(n) < y_prob).astype(int)
    ece = _compute_ece(y_true, y_prob)
    assert ece < 0.10, f"ECE {ece} too high for near-perfect calibration"


def test_ece_worst_case() -> None:
    """A maximally miscalibrated model has high ECE."""
    y_true = np.zeros(100, dtype=int)
    y_prob = np.ones(100)  # predicts 100% but truth is 0%
    ece = _compute_ece(y_true, y_prob)
    assert ece > 0.5, f"ECE {ece} too low for worst-case calibration"


def test_ece_bounds() -> None:
    """ECE must always be in [0, 1]."""
    for _ in range(20):
        rng = np.random.default_rng(_)
        n = 50
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)
        ece = _compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0, f"ECE {ece} out of [0,1]"


def test_ece_empty_input() -> None:
    """Empty arrays should return 0."""
    ece = _compute_ece(np.array([]), np.array([]))
    assert ece == 0.0


def test_metrics_json_contract() -> None:
    """The saved metrics JSON should contain calibration fields."""
    import json
    from pathlib import Path

    metrics_path = Path("outputs/models/command_following_metrics.json")
    if not metrics_path.exists():
        pytest.skip("No metrics file yet — run pipeline first")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_keys = {
        "brier_score",
        "ece",
        "across_subject_auc",
        "train_auc",
        "calibration_method",
        "n_training_samples",
        "n_features",
        "n_subjects",
        "model_type",
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"

    # Brier should be in [0, 0.25] for a reasonable model
    assert 0.0 <= metrics["brier_score"] <= 0.25
    # ECE should be reasonable
    assert 0.0 <= metrics["ece"] <= 0.5


def test_dashboard_results_csv_contract() -> None:
    """results_command_following.csv should have all dashboard-required columns."""
    import pandas as pd
    from pathlib import Path

    csv_path = Path("outputs/results_command_following.csv")
    if not csv_path.exists():
        pytest.skip("No results CSV yet — run pipeline first")

    df = pd.read_csv(csv_path)
    required_cols = {
        "subject_id",
        "CFI",
        "Confidence",
        "QualityScore",
        "Evidence",
        "ConfidenceLevel",
        "QualityLevel",
        "mean_imagery_probability",
        "consistency",
        "ci_lower",
        "ci_upper",
        "n_trials",
        "sharpness",
        "usable_trial_score",
        "reliability_score",
        "ci_tightness",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing result columns: {missing}"

    # All confidence values in [0, 100]
    assert df["Confidence"].between(0, 100).all()
    assert df["CFI"].between(0, 100).all()
    assert df["QualityScore"].between(0, 100).all()
