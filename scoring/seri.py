"""
NFM — Stimulus-Evoked Response Index (SERI)
=============================================
Replaces CFI emphasis with a stimulus-evoked response indicator.

SERI ∈ [0, 100] computed from:
  calibrated response probability + consistency + quality penalty.

Output triplet: SERI, Confidence, QualityScore

Evidence bands:
  Low < 40, Moderate 40–69, High >= 70
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ── Thresholds ────────────────────────────────────────────────
SERI_LOW_MAX: float = 39.999
SERI_MODERATE_MAX: float = 69.999
SERI_MIN_USABLE_TRIALS: int = 5


def seri_evidence_label(seri: float) -> str:
    """Map SERI to evidence band label."""
    if seri <= SERI_LOW_MAX:
        return "Low"
    if seri <= SERI_MODERATE_MAX:
        return "Moderate"
    return "High"


def seri_confidence_label(confidence: float) -> str:
    """Map confidence to qualitative label."""
    if confidence < 45:
        return "Low"
    if confidence < 70:
        return "Moderate"
    return "High"


def seri_quality_label(quality: float) -> str:
    """Map quality score to qualitative label."""
    if quality < 50:
        return "Limited"
    if quality < 75:
        return "Acceptable"
    return "Good"


def seri_clinical_summary(seri: float, confidence: float, quality: float) -> str:
    """Generate clinical summary text for SERI result."""
    evidence = seri_evidence_label(seri)
    conf = seri_confidence_label(confidence)
    qual = seri_quality_label(quality)

    return (
        f"Evidence of stimulus-evoked cortical modulation is {evidence}. "
        f"Confidence is {conf} and data quality is {qual}. "
        "This output is a stimulus-evoked response indicator reflecting "
        "functional cortical responsiveness to external stimuli, "
        "for decision-support only and not a diagnosis."
    )


def compute_response_consistency(
    response_probabilities: np.ndarray,
) -> float:
    """Compute consistency of stimulus response probabilities in [0, 1].

    Uses class-conditional standard deviation to handle bimodal distributions.
    """
    if response_probabilities.size == 0:
        return 0.0

    probs = np.asarray(response_probabilities, dtype=float)
    probs = probs[~np.isnan(probs)]

    if probs.size == 0:
        return 0.0

    # Developer note:
    # probabilities can be bimodal (clear responders + clear non-responders).
    # class-conditional spread is more informative than one global std.
    hi = probs[probs > 0.5]
    lo = probs[probs <= 0.5]

    stds: list[float] = []
    if hi.size > 1:
        stds.append(float(np.std(hi)))
    if lo.size > 1:
        stds.append(float(np.std(lo)))

    cond_std = float(np.mean(stds)) if stds else float(np.std(probs))
    return float(np.clip(np.exp(-3.0 * cond_std), 0.0, 1.0))


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 3:
        return {"ci_lower": 0.0, "ci_upper": 1.0, "ci_width": 1.0}

    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
    }


def compute_seri(
    response_probabilities: np.ndarray,
    quality_score: float,
    evoked_snr: float = 1.0,
    response_consistency: float | None = None,
    model_reliability: float | None = None,
    min_usable_trials: int = SERI_MIN_USABLE_TRIALS,
) -> dict[str, Any]:
    """Compute Stimulus-Evoked Response Index (SERI).

    Parameters
    ----------
    response_probabilities : per-trial probability of stimulus response
    quality_score : data quality score [0, 100]
    evoked_snr : mean evoked SNR ratio
    response_consistency : pre-computed consistency [0, 1] or None (auto-compute)
    model_reliability : model calibration reliability [0, 1]
    min_usable_trials : minimum trials for adequate confidence

    Returns
    -------
    Dictionary with: SERI, Confidence, QualityScore, consistency,
                     evidence_label, confidence_label, quality_label,
                     clinical_summary, and component scores.
    """
    probs = np.asarray(response_probabilities, dtype=float)
    probs = probs[~np.isnan(probs)]
    quality = float(np.clip(quality_score, 0.0, 100.0))

    if probs.size == 0:
        return {
            "SERI": 0.0,
            "Confidence": 0.0,
            "QualityScore": quality,
            "consistency": 0.0,
            "mean_response_probability": 0.0,
            "evoked_snr": evoked_snr,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "ci_width": 1.0,
            "evidence_label": "Low",
            "confidence_label": seri_confidence_label(0.0),
            "quality_label": seri_quality_label(quality),
            "clinical_summary": seri_clinical_summary(0.0, 0.0, quality),
        }

    mean_prob = float(np.clip(probs.mean(), 0.0, 1.0))

    if response_consistency is not None:
        consistency = float(np.clip(response_consistency, 0.0, 1.0))
    else:
        consistency = compute_response_consistency(probs)

    boot = _bootstrap_ci(probs)

    # SERI block:
    # combine signal strength + stability + SNR + recording quality.
    base_signal = mean_prob * 100.0

    # Consistency bonus
    consistency_score = consistency * 100.0

    # SNR contribution (capped, normalized)
    snr_norm = float(np.clip((evoked_snr - 0.8) / 1.2, 0.0, 1.0))
    snr_score = snr_norm * 100.0

    # Weighted combination
    raw_seri = (
        0.50 * base_signal
        + 0.20 * consistency_score
        + 0.15 * snr_score
        + 0.15 * quality
    )

    # Quality gate:
    # even if model is confident, low-quality recordings should not score high.
    quality_penalty = 0.5 + 0.5 * (quality / 100.0)
    seri = float(np.clip(raw_seri * quality_penalty, 0.0, 100.0))

    # Confidence block:
    # estimate how much trust we should place in this SERI number.
    # Sharpness = per-trial decisiveness.
    sharpness = float(np.mean(np.abs(2.0 * probs - 1.0))) * 100.0

    # Consistency
    consistency_pct = consistency * 100.0

    # Trial count adequacy
    trial_ratio = probs.size / max(float(min_usable_trials), 1.0)
    usable_trial_score = float(np.clip(trial_ratio, 0.0, 1.5)) / 1.5 * 100.0

    # Quality
    quality_score_pct = quality

    # Model reliability
    base_reliability = float(
        np.clip(model_reliability if model_reliability is not None else 0.75, 0.0, 1.0)
    ) * 100.0

    # CI tightness
    ci_tightness = float(np.clip(1.0 - boot["ci_width"], 0.0, 1.0)) * 100.0

    confidence = (
        0.15 * sharpness
        + 0.20 * consistency_pct
        + 0.15 * usable_trial_score
        + 0.15 * quality_score_pct
        + 0.20 * base_reliability
        + 0.15 * ci_tightness
    )

    # Final gates for hard failure modes (poor quality, wide CI, too few trials).
    if quality < 40:
        confidence *= 0.65
    if boot["ci_width"] > 0.5:
        confidence *= 0.80
    if probs.size < min_usable_trials:
        confidence *= max(0.5, probs.size / max(float(min_usable_trials), 1.0))

    confidence = float(np.clip(confidence, 0.0, 100.0))

    return {
        "SERI": seri,
        "Confidence": confidence,
        "QualityScore": quality,
        "consistency": consistency,
        "mean_response_probability": mean_prob,
        "evoked_snr": evoked_snr,
        "ci_lower": boot["ci_lower"],
        "ci_upper": boot["ci_upper"],
        "ci_width": boot["ci_width"],
        "sharpness": sharpness,
        "usable_trial_score": usable_trial_score,
        "reliability_score": base_reliability,
        "ci_tightness": ci_tightness,
        "evidence_label": seri_evidence_label(seri),
        "confidence_label": seri_confidence_label(confidence),
        "quality_label": seri_quality_label(quality),
        "clinical_summary": seri_clinical_summary(seri, confidence, quality),
    }
