"""Tests for drift monitoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    _ks_stat,
    _psi,
    build_drift_report,
    compute_calibration_drift,
    compute_feature_drift,
    compute_prediction_drift,
)
from schemas.clinical_validation import AlertLevel


class TestPSI:
    def test_identical_distributions(self):
        ref = np.linspace(0, 1, 100)
        cur = np.linspace(0, 1, 100)
        psi = _psi(ref, cur)
        assert psi < 0.05  # very low drift

    def test_shifted_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(3, 1, 500)  # big shift
        psi = _psi(ref, cur)
        assert psi > 0.25  # red alert territory

    def test_psi_non_negative(self):
        np.random.seed(42)
        ref = np.random.rand(100)
        cur = np.random.rand(100)
        psi = _psi(ref, cur)
        assert psi >= 0


class TestKSStat:
    def test_identical(self):
        ref = np.random.RandomState(42).rand(100)
        ks = _ks_stat(ref, ref)
        assert ks < 0.05

    def test_shifted(self):
        ref = np.random.RandomState(42).normal(0, 1, 200)
        cur = np.random.RandomState(42).normal(5, 1, 200)
        ks = _ks_stat(ref, cur)
        assert ks > 0.5


class TestFeatureDrift:
    def _make_features_df(self, n, seed, shift=0.0):
        rng = np.random.RandomState(seed)
        return pd.DataFrame({
            "alpha_power": rng.normal(10 + shift, 2, n),
            "beta_power": rng.normal(5 + shift, 1, n),
        })

    def test_no_drift(self):
        ref = self._make_features_df(500, 42)
        cur = self._make_features_df(500, 43)
        report = compute_feature_drift(ref, cur)
        assert isinstance(report, dict)
        for feat, psi_val in report.items():
            assert psi_val < 0.30  # same distribution — low drift

    def test_high_drift(self):
        ref = self._make_features_df(200, 42, shift=0)
        cur = self._make_features_df(200, 43, shift=10)
        report = compute_feature_drift(ref, cur)
        drifted = [v for v in report.values() if v > 0.25]
        assert len(drifted) > 0  # at least one feature drifted


class TestPredictionDrift:
    def test_stable_predictions(self):
        np.random.seed(42)
        ref = np.random.beta(2, 5, 200)
        cur = np.random.beta(2, 5, 200)
        report = compute_prediction_drift(ref, cur)
        assert "psi" in report
        assert "ks_stat" in report

    def test_shifted_predictions(self):
        np.random.seed(42)
        ref = np.random.beta(2, 5, 200)
        cur = np.random.beta(5, 2, 200)  # reversed — major shift
        report = compute_prediction_drift(ref, cur)
        assert report["psi"] > 0.10


class TestCalibrationDrift:
    def test_stable_calibration(self):
        drift = compute_calibration_drift(baseline_ece=0.15, current_ece=0.16)
        assert drift == pytest.approx(0.01, abs=1e-9)

    def test_delta_direction(self):
        drift = compute_calibration_drift(baseline_ece=0.01, current_ece=0.10)
        assert drift >= 0


class TestBuildDriftReport:
    def test_report_structure(self):
        np.random.seed(42)
        ref_df = pd.DataFrame({
            "alpha_power": np.random.normal(10, 2, 200),
            "beta_power": np.random.normal(5, 1, 200),
        })
        cur_df = pd.DataFrame({
            "alpha_power": np.random.normal(10, 2, 200),
            "beta_power": np.random.normal(5, 1, 200),
        })
        ref_preds = np.random.beta(2, 5, 200)
        cur_preds = np.random.beta(2, 5, 200)

        report = build_drift_report(
            baseline_df=ref_df,
            current_df=cur_df,
            baseline_probs=ref_preds,
            current_probs=cur_preds,
            baseline_ece=0.10,
            current_ece=0.12,
        )
        assert hasattr(report, "alert_level")
        assert report.alert_level in (AlertLevel.GREEN, AlertLevel.YELLOW, AlertLevel.RED)
        assert report.prediction_shift_metrics is not None
        assert report.feature_shift_metrics is not None

    def test_red_alert_on_major_shift(self):
        np.random.seed(42)
        ref_df = pd.DataFrame({"f1": np.random.normal(0, 1, 200)})
        cur_df = pd.DataFrame({"f1": np.random.normal(10, 1, 200)})
        ref_preds = np.random.beta(2, 8, 200)
        cur_preds = np.random.beta(8, 2, 200)

        report = build_drift_report(
            baseline_df=ref_df,
            current_df=cur_df,
            baseline_probs=ref_preds,
            current_probs=cur_preds,
            baseline_ece=0.01,
            current_ece=0.50,
        )
        assert report.alert_level == AlertLevel.RED
