"""
Tests for SERI (Stimulus-Evoked Response Index) scoring.

Validates:
- SERI output always in [0, 100]
- Confidence always in [0, 100]
- Quality always in [0, 100]
- Low quality penalizes SERI
- Empty / NaN probabilities handled gracefully
- Evidence labels match thresholds
- Consistency monotonicity
"""

from __future__ import annotations

import numpy as np
import pytest

from scoring.seri import (
    compute_seri,
    seri_clinical_summary,
    seri_confidence_label,
    seri_evidence_label,
    seri_quality_label,
)


def test_seri_outputs_within_bounds() -> None:
    probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    result = compute_seri(probs, quality_score=75.0)

    assert 0.0 <= result["SERI"] <= 100.0
    assert 0.0 <= result["Confidence"] <= 100.0
    assert 0.0 <= result["QualityScore"] <= 100.0


def test_low_quality_penalizes_seri() -> None:
    probs = np.array([0.8, 0.82, 0.78, 0.81, 0.79], dtype=float)
    high_quality = compute_seri(probs, quality_score=90.0)
    low_quality = compute_seri(probs, quality_score=20.0)

    assert low_quality["SERI"] < high_quality["SERI"]


def test_confidence_bounds_strict() -> None:
    """Confidence must always be in [0, 100] regardless of inputs."""
    for qs in [0.0, 25.0, 50.0, 75.0, 100.0]:
        for p in [
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.5] * 100),
            np.array([0.9] * 50),
        ]:
            result = compute_seri(p, quality_score=qs)
            assert 0.0 <= result["Confidence"] <= 100.0, (
                f"Confidence={result['Confidence']} out of [0,100] "
                f"with quality={qs}, probs={p[:3]}"
            )
            assert 0.0 <= result["SERI"] <= 100.0


def test_quality_bounds_strict() -> None:
    """QualityScore output is always clamped to [0, 100]."""
    for qs_in in [-10.0, 0.0, 50.0, 100.0, 150.0]:
        result = compute_seri(np.array([0.7, 0.8]), quality_score=qs_in)
        assert 0.0 <= result["QualityScore"] <= 100.0


def test_empty_probabilities() -> None:
    result = compute_seri(np.array([]), quality_score=80.0)
    assert result["SERI"] == 0.0
    assert result["Confidence"] == 0.0
    assert result["evidence_label"] == "Low"


def test_nan_probabilities_filtered() -> None:
    probs = np.array([0.8, np.nan, 0.7, np.nan, 0.9])
    result = compute_seri(probs, quality_score=80.0)
    assert result["SERI"] > 0
    assert result["mean_response_probability"] > 0


def test_evidence_label_thresholds() -> None:
    assert seri_evidence_label(0.0) == "Low"
    assert seri_evidence_label(39.9) == "Low"
    assert seri_evidence_label(40.0) == "Moderate"
    assert seri_evidence_label(69.9) == "Moderate"
    assert seri_evidence_label(70.0) == "High"
    assert seri_evidence_label(100.0) == "High"


def test_confidence_label_values() -> None:
    assert seri_confidence_label(0.0) in {"Low", "Moderate", "High"}
    assert seri_confidence_label(50.0) in {"Low", "Moderate", "High"}
    assert seri_confidence_label(100.0) in {"Low", "Moderate", "High"}


def test_quality_label_values() -> None:
    assert seri_quality_label(0.0) in {"Limited", "Acceptable", "Good"}
    assert seri_quality_label(50.0) in {"Limited", "Acceptable", "Good"}
    assert seri_quality_label(100.0) in {"Limited", "Acceptable", "Good"}


def test_clinical_summary_returns_string() -> None:
    result = compute_seri(np.array([0.6, 0.7, 0.65]), quality_score=70.0)
    summary = seri_clinical_summary(result["SERI"], result["Confidence"], result["QualityScore"])
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_high_snr_boosts_seri() -> None:
    probs = np.array([0.7, 0.75, 0.72, 0.68, 0.71], dtype=float)
    low_snr = compute_seri(probs, quality_score=70.0, evoked_snr=0.5)
    high_snr = compute_seri(probs, quality_score=70.0, evoked_snr=5.0)
    assert high_snr["SERI"] >= low_snr["SERI"]


def test_high_consistency_boosts_seri() -> None:
    probs = np.array([0.7, 0.75, 0.72, 0.68, 0.71], dtype=float)
    low_cons = compute_seri(probs, quality_score=70.0, response_consistency=0.1)
    high_cons = compute_seri(probs, quality_score=70.0, response_consistency=0.9)
    assert high_cons["SERI"] >= low_cons["SERI"]


def test_seri_result_keys() -> None:
    result = compute_seri(np.array([0.5, 0.6, 0.7]), quality_score=60.0)
    expected_keys = {
        "SERI", "Confidence", "QualityScore",
        "evidence_label", "confidence_label", "quality_label",
        "clinical_summary", "mean_response_probability", "consistency",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_extreme_probabilities() -> None:
    """All-zero and all-one probs should not crash."""
    r0 = compute_seri(np.zeros(10), quality_score=80.0)
    r1 = compute_seri(np.ones(10), quality_score=80.0)
    assert 0.0 <= r0["SERI"] <= 100.0
    assert 0.0 <= r1["SERI"] <= 100.0
    assert r1["SERI"] > r0["SERI"]


def test_single_trial() -> None:
    """Single trial should still produce valid output."""
    result = compute_seri(np.array([0.8]), quality_score=70.0)
    assert 0.0 <= result["SERI"] <= 100.0
    assert result["Confidence"] >= 0.0
