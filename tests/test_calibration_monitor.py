"""Tests for calibration monitoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.calibration_monitor import (
    build_calibration_report,
    compute_brier,
    compute_ece,
    compute_mce,
    reliability_curve,
)


class TestReliabilityCurve:
    def test_output_shape(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4, 0.85, 0.95])
        result = reliability_curve(y_true, y_prob, n_bins=5)
        frac_pos = result["bin_true_fractions"]
        mean_pred = result["bin_predicted_means"]
        counts = result["bin_counts"]
        assert len(frac_pos) == len(mean_pred)
        assert len(frac_pos) == len(counts)
        assert all(0 <= f <= 1 for f in frac_pos)

    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
        result = reliability_curve(y_true, y_prob, n_bins=5)
        frac_pos = result["bin_true_fractions"]
        mean_pred = result["bin_predicted_means"]
        # For perfectly calibrated, fraction of positives should be close to mean predicted
        for fp, mp in zip(frac_pos, mean_pred):
            assert abs(fp - mp) < 0.5  # generous tolerance for 10 points


class TestECE:
    def test_perfect_calibration_low_ece(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9, 1.0])
        ece = compute_ece(y_true, y_prob)
        assert 0 <= ece <= 1.0
        assert ece < 0.3  # reasonably calibrated

    def test_terrible_calibration_high_ece(self):
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.9, 1.0, 1.0, 0.9, 0.9])
        ece = compute_ece(y_true, y_prob)
        assert ece > 0.5  # very miscalibrated

    def test_ece_bounds(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        ece = compute_ece(y_true, y_prob)
        assert 0 <= ece <= 1.0


class TestMCE:
    def test_mce_gte_ece(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        ece = compute_ece(y_true, y_prob)
        mce = compute_mce(y_true, y_prob)
        assert mce >= ece - 1e-9  # MCE >= ECE always


class TestBrier:
    def test_brier_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert compute_brier(y_true, y_prob) == pytest.approx(0.0)

    def test_brier_worst(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert compute_brier(y_true, y_prob) == pytest.approx(1.0)

    def test_brier_bounds(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        brier = compute_brier(y_true, y_prob)
        assert 0 <= brier <= 1.0


class TestBuildCalibrationReport:
    def test_report_structure(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.rand(50)
        df = pd.DataFrame({"reference_label": y_true, "response_probability": y_prob})
        report = build_calibration_report(df)
        overall = report["overall"]
        assert "ece" in overall
        assert "mce" in overall
        assert "brier" in overall
        assert "n_samples" in overall
        assert overall["n_samples"] == 50
        assert 0 <= overall["ece"] <= 1
        assert 0 <= overall["brier"] <= 1

    def test_report_with_few_samples(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        df = pd.DataFrame({"reference_label": y_true, "response_probability": y_prob})
        report = build_calibration_report(df)
        assert "error" in report  # too few labelled samples
