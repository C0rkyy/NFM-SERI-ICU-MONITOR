from __future__ import annotations

from typing import Any

import numpy as np

from config import CFI_LOW_MAX, CFI_MIN_USABLE_TRIALS, CFI_MODERATE_MAX


def compute_trial_consistency(probabilities: np.ndarray) -> float:
    """Return consistency in [0,1] based on **class-conditional** dispersion.

    For a balanced dataset with both imagery (p > 0.5) and rest (p ≤ 0.5) trials,
    global std is misleadingly high because the distribution is bimodal by design.
    Instead, we measure variability **within** each predicted class separately,
    then average. This rewards models that consistently assign decisive
    probabilities within each class.
    """
    if probabilities.size == 0:
        return 0.0

    hi = probabilities[probabilities > 0.5]
    lo = probabilities[probabilities <= 0.5]

    stds: list[float] = []
    if hi.size > 1:
        stds.append(float(np.std(hi)))
    if lo.size > 1:
        stds.append(float(np.std(lo)))

    if not stds:
        # Fallback to global std if single class
        cond_std = float(np.std(probabilities))
    else:
        cond_std = float(np.mean(stds))

    return float(np.clip(np.exp(-3.0 * cond_std), 0.0, 1.0))


def cfi_evidence_label(cfi: float) -> str:
    if cfi <= CFI_LOW_MAX:
        return "Low"
    if cfi <= CFI_MODERATE_MAX:
        return "Moderate"
    return "High"


def confidence_label(confidence: float) -> str:
    if confidence < 45:
        return "Low"
    if confidence < 70:
        return "Moderate"
    return "High"


def quality_label(quality: float) -> str:
    if quality < 50:
        return "Limited"
    if quality < 75:
        return "Acceptable"
    return "Good"


def clinical_summary_text(cfi: float, confidence: float, quality: float) -> str:
    evidence = cfi_evidence_label(cfi)
    conf = confidence_label(confidence)
    qual = quality_label(quality)

    return (
        f"Evidence level is {evidence}. Confidence is {conf} and data quality is {qual}. "
        "This output is a command-following indicator reflecting functional responsiveness "
        "to mental commands, for decision-support only and not a diagnosis."
    )


def _bootstrap_ci(
    probabilities: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for mean imagery probability."""
    rng = np.random.default_rng(seed)
    n = len(probabilities)
    if n < 3:
        return {"ci_lower": 0.0, "ci_upper": 1.0, "ci_width": 1.0}

    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(probabilities, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
    }


def compute_cfi(
    imagery_probabilities: np.ndarray,
    quality_score: float,
    min_usable_trials: int = CFI_MIN_USABLE_TRIALS,
    model_reliability: float | None = None,
    brier_score: float | None = None,
    ece: float | None = None,
) -> dict[str, Any]:
    """Compute command-following index, confidence, quality, and interpretation labels.

    Enhanced with:
    - Bootstrap CI-based uncertainty
    - Calibration-aware confidence (brier, ECE)
    - Probability sharpness quantification
    - Trial count adequacy scoring
    - Quality and uncertainty gating
    """
    probabilities = np.asarray(imagery_probabilities, dtype=float)
    probabilities = probabilities[~np.isnan(probabilities)]

    quality = float(np.clip(quality_score, 0.0, 100.0))

    if probabilities.size == 0:
        return {
            "CFI": 0.0,
            "Confidence": 0.0,
            "QualityScore": quality,
            "consistency": 0.0,
            "mean_imagery_probability": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "ci_width": 1.0,
            "evidence_label": "Low",
            "confidence_label": confidence_label(0.0),
            "quality_label": quality_label(quality),
            "clinical_summary": clinical_summary_text(0.0, 0.0, quality),
        }

    mean_prob = float(np.clip(probabilities.mean(), 0.0, 1.0))
    consistency = compute_trial_consistency(probabilities)
    boot = _bootstrap_ci(probabilities)

    # ── CFI computation ────────────────────────────────────────
    base_signal = mean_prob * 100.0
    consistency_score = consistency * 100.0

    raw_cfi = 0.60 * base_signal + 0.25 * consistency_score + 0.15 * quality
    quality_penalty = 0.5 + 0.5 * (quality / 100.0)
    cfi = float(np.clip(raw_cfi * quality_penalty, 0.0, 100.0))

    # ── Confidence computation (improved, calibration-aware) ──
    # Component 1: Probability sharpness [0-100]
    # Per-trial decisiveness: how far each individual prediction is from chance
    # (not mean deviation, which is ~0 in balanced datasets by design)
    sharpness = float(np.mean(np.abs(2.0 * probabilities - 1.0))) * 100.0

    # Component 2: Consistency [0-100]
    consistency_pct = consistency * 100.0

    # Component 3: Usable trial count adequacy [0-100]
    # Sigmoid-shaped: reaches 100 once we have ≥ min_usable_trials
    trial_ratio = probabilities.size / max(float(min_usable_trials), 1.0)
    usable_trial_score = float(np.clip(trial_ratio, 0.0, 1.5)) / 1.5 * 100.0

    # Component 4: Quality [0-100]
    quality_score_pct = quality

    # Component 5: Model reliability from CV/calibration [0-100]
    base_reliability = float(
        np.clip(model_reliability if model_reliability is not None else 0.75, 0.0, 1.0)
    )
    # Adjust reliability with calibration diagnostics if available
    if brier_score is not None:
        # Lower brier = better calibration; brier ~0 is perfect, ~0.25 is chance
        brier_bonus = float(np.clip(1.0 - 4.0 * brier_score, 0.0, 1.0))
        base_reliability = 0.6 * base_reliability + 0.4 * brier_bonus
    if ece is not None:
        # Lower ECE = better calibration
        ece_bonus = float(np.clip(1.0 - 5.0 * ece, 0.0, 1.0))
        base_reliability = 0.8 * base_reliability + 0.2 * ece_bonus
    reliability_score = base_reliability * 100.0

    # Component 6: Bootstrap CI tightness [0-100]
    ci_tightness = float(np.clip(1.0 - boot["ci_width"], 0.0, 1.0)) * 100.0

    # Weighted combination
    confidence = (
        0.15 * sharpness
        + 0.20 * consistency_pct
        + 0.15 * usable_trial_score
        + 0.15 * quality_score_pct
        + 0.20 * reliability_score
        + 0.15 * ci_tightness
    )

    # Gating: cap confidence if data quality is poor or uncertainty is high
    if quality < 40:
        confidence *= 0.65
    if boot["ci_width"] > 0.5:
        confidence *= 0.80
    if probabilities.size < min_usable_trials:
        # Soft penalty proportional to shortfall
        confidence *= max(0.5, probabilities.size / max(float(min_usable_trials), 1.0))

    confidence = float(np.clip(confidence, 0.0, 100.0))

    evidence = cfi_evidence_label(cfi)
    conf_label = confidence_label(confidence)
    qual_label = quality_label(quality)

    return {
        "CFI": cfi,
        "Confidence": confidence,
        "QualityScore": quality,
        "consistency": consistency,
        "mean_imagery_probability": mean_prob,
        "ci_lower": boot["ci_lower"],
        "ci_upper": boot["ci_upper"],
        "ci_width": boot["ci_width"],
        "sharpness": sharpness,
        "usable_trial_score": usable_trial_score,
        "reliability_score": reliability_score,
        "ci_tightness": ci_tightness,
        "evidence_label": evidence,
        "confidence_label": conf_label,
        "quality_label": qual_label,
        "clinical_summary": clinical_summary_text(cfi, confidence, quality),
    }
