"""
External validation engine.

Computes validation metrics (AUC, Brier, ECE, sensitivity, specificity, PPV,
NPV, PR-AUC, calibration slope/intercept) on held-out splits.

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schemas.clinical_validation import ValidationMetrics, ValidationSplitSpec

logger = logging.getLogger(__name__)


def _safe_metric(func: Any, *args: Any, default: float = 0.0, **kwargs: Any) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return default


def compute_validation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> ValidationMetrics:
    """Compute a full set of validation metrics from labels + probabilities."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Need both classes present for AUC
    if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
        logger.warning("Only one class present in y_true; metrics may be degenerate.")
        auc = 0.5
        pr_auc = float(y_true.mean()) if len(y_true) > 0 else 0.0
    else:
        auc = _safe_metric(roc_auc_score, y_true, y_prob, default=0.5)
        pr_auc = _safe_metric(average_precision_score, y_true, y_prob, default=0.0)

    brier = _safe_metric(brier_score_loss, y_true, y_prob, default=1.0)

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # ECE
    ece = _compute_ece(y_true, y_prob, n_bins=n_bins)

    # Calibration slope / intercept (logistic)
    cal_slope, cal_intercept = _calibration_curve_fit(y_true, y_prob)

    return ValidationMetrics(
        auc=auc,
        pr_auc=pr_auc,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        brier=brier,
        ece=ece,
        calibration_slope=cal_slope,
        calibration_intercept=cal_intercept,
        n_samples=len(y_true),
    )


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_prob = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(avg_prob - avg_true)
    return float(ece)


def _calibration_curve_fit(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """Fit logistic calibration curve; return (slope, intercept)."""
    try:
        from sklearn.linear_model import LogisticRegression

        y_prob_safe = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logits = np.log(y_prob_safe / (1 - y_prob_safe)).reshape(-1, 1)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        lr.fit(logits, y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return 1.0, 0.0


def run_external_validation(
    results_df: pd.DataFrame,
    split_spec: ValidationSplitSpec,
    prob_col: str = "response_probability",
    label_col: str = "reference_label",
) -> dict[str, Any]:
    """Run external validation on a pre-split results DataFrame.

    Returns dict with metrics, fold_df, and predictions.
    """
    if label_col not in results_df.columns:
        logger.warning("Label column '%s' not in results; skipping validation", label_col)
        return {"metrics": None, "error": f"Missing label column: {label_col}"}

    if prob_col not in results_df.columns:
        if "SERI" in results_df.columns:
            prob_col = "SERI"
            results_df = results_df.copy()
            results_df[prob_col] = results_df[prob_col] / 100.0
        else:
            return {"metrics": None, "error": f"Missing prob column: {prob_col}"}

    train_ids = set(split_spec.train_ids)
    test_ids = set(split_spec.test_ids)

    if "session_id" in results_df.columns:
        test_mask = results_df["session_id"].astype(str).isin(test_ids)
    else:
        test_mask = pd.Series(False, index=results_df.index)

    test_df = results_df[test_mask].dropna(subset=[label_col, prob_col])

    if len(test_df) < 5:
        logger.warning("Too few test samples (%d) for external validation", len(test_df))
        return {"metrics": None, "error": f"Too few test samples: {len(test_df)}"}

    y_true = test_df[label_col].values
    y_prob = test_df[prob_col].values

    metrics = compute_validation_metrics(y_true, y_prob)
    metrics.split_type = split_spec.split_type.value

    predictions_df = test_df[["session_id", label_col, prob_col]].copy() if "session_id" in test_df.columns else test_df[[label_col, prob_col]].copy()
    predictions_df["predicted_class"] = (y_prob >= 0.5).astype(int)

    return {
        "metrics": metrics,
        "predictions": predictions_df,
        "split_spec": split_spec,
    }


def save_validation_outputs(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save external validation outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    metrics = result.get("metrics")
    if metrics is not None:
        p = output_dir / "external_validation_metrics.json"
        p.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
        paths["metrics"] = p

    predictions = result.get("predictions")
    if predictions is not None:
        p = output_dir / "predictions_external.csv"
        predictions.to_csv(p, index=False)
        paths["predictions"] = p

    spec = result.get("split_spec")
    if spec is not None:
        p = output_dir / "split_spec.json"
        p.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        paths["split_spec"] = p

    # fold metrics (single-fold for now)
    if metrics is not None:
        fold_df = pd.DataFrame([metrics.model_dump()])
        p = output_dir / "fold_metrics.csv"
        fold_df.to_csv(p, index=False)
        paths["fold_metrics"] = p

    logger.info("Saved validation outputs: %s", list(paths.keys()))
    return paths
