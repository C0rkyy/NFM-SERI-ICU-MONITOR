from __future__ import annotations

import numpy as np
import pytest

from scoring.cfi import (
    cfi_evidence_label,
    clinical_summary_text,
    compute_cfi,
    compute_trial_consistency,
    confidence_label,
    quality_label,
)


def test_cfi_outputs_within_bounds() -> None:
    probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    result = compute_cfi(probs, quality_score=75.0)

    assert 0.0 <= result["CFI"] <= 100.0
    assert 0.0 <= result["Confidence"] <= 100.0
    assert 0.0 <= result["QualityScore"] <= 100.0


def test_low_quality_penalizes_cfi() -> None:
    probs = np.array([0.8, 0.82, 0.78, 0.81, 0.79], dtype=float)
    high_quality = compute_cfi(probs, quality_score=90.0)
    low_quality = compute_cfi(probs, quality_score=20.0)

    assert low_quality["CFI"] < high_quality["CFI"]


def test_confidence_bounds_strict() -> None:
    """Confidence must always be in [0, 100] regardless of inputs."""
    for qs in [0.0, 25.0, 50.0, 75.0, 100.0]:
        for p in [
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.5] * 100),
            np.array([0.9] * 50),
        ]:
            result = compute_cfi(p, quality_score=qs)
            assert 0.0 <= result["Confidence"] <= 100.0, (
                f"Confidence={result['Confidence']} out of [0,100] "
                f"with quality={qs}, probs={p[:3]}"
            )
            assert 0.0 <= result["CFI"] <= 100.0


def test_quality_bounds_strict() -> None:
    """QualityScore output is always clamped to [0, 100]."""
    for qs_in in [-10.0, 0.0, 50.0, 100.0, 150.0]:
        result = compute_cfi(np.array([0.7, 0.8]), quality_score=qs_in)
        assert 0.0 <= result["QualityScore"] <= 100.0


def test_empty_probabilities() -> None:
    result = compute_cfi(np.array([]), quality_score=80.0)
    assert result["CFI"] == 0.0
    assert result["Confidence"] == 0.0
    assert result["evidence_label"] == "Low"


def test_nan_probabilities_filtered() -> None:
    probs = np.array([0.8, np.nan, 0.7, np.nan, 0.9])
    result = compute_cfi(probs, quality_score=80.0)
    assert result["CFI"] > 0
    assert result["mean_imagery_probability"] > 0


def test_consistency_monotonic() -> None:
    """More consistent probabilities → higher consistency."""
    consistent = np.array([0.8, 0.81, 0.79, 0.80, 0.82])
    inconsistent = np.array([0.2, 0.9, 0.1, 0.8, 0.3])
    c1 = compute_trial_consistency(consistent)
    c2 = compute_trial_consistency(inconsistent)
    assert c1 > c2


def test_brier_ece_affect_confidence() -> None:
    """Better calibration metrics should yield higher confidence."""
    probs = np.array([0.85, 0.82, 0.88, 0.81, 0.87] * 4)
    good_cal = compute_cfi(probs, quality_score=90, model_reliability=0.85, brier_score=0.05, ece=0.02)
    bad_cal = compute_cfi(probs, quality_score=90, model_reliability=0.85, brier_score=0.30, ece=0.20)
    assert good_cal["Confidence"] > bad_cal["Confidence"]


def test_bootstrap_ci_present() -> None:
    """Bootstrap CI fields should be present in the result."""
    probs = np.array([0.7, 0.8, 0.6, 0.9, 0.75])
    result = compute_cfi(probs, quality_score=80.0)
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["ci_lower"] <= result["ci_upper"]


def test_evidence_labels() -> None:
    assert cfi_evidence_label(10.0) == "Low"
    assert cfi_evidence_label(50.0) == "Moderate"
    assert cfi_evidence_label(80.0) == "High"


def test_confidence_labels() -> None:
    assert confidence_label(30.0) == "Low"
    assert confidence_label(55.0) == "Moderate"
    assert confidence_label(75.0) == "High"


def test_quality_labels() -> None:
    assert quality_label(30.0) == "Limited"
    assert quality_label(60.0) == "Acceptable"
    assert quality_label(85.0) == "Good"


def test_clinical_summary_contains_disclaimer() -> None:
    summary = clinical_summary_text(50.0, 60.0, 70.0)
    assert "decision-support only" in summary.lower()
    assert "not a diagnosis" in summary.lower()
