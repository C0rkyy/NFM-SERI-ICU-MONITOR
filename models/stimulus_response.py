"""
NFM — Stimulus Response Model
================================
Primary model target: detect stimulus-evoked response presence/strength,
NOT intentional command-following.

Approach:
  Classify stimulus-response vs baseline/non-response patterns at trial level,
  then aggregate to session score.

Keeps calibrated probabilities (sigmoid/isotonic based on sample size).
Supports parallel CV and training with n_jobs.
Preserves across-subject and within-subject validation modes.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_N_JOBS, MODEL_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

_N_JOBS: int = max(1, int(os.environ.get("NFM_N_JOBS", DEFAULT_N_JOBS)))

# Developer note:
# keep this deny-list explicit so we can audit leakage risk quickly.
# Any label-derived or post-model column belongs here.
EXCLUDE_COLS: set[str] = {
    "epoch_idx", "subject_id", "session_id", "event_code", "description",
    "sample", "time_s", "stimulus_label", "wav_file", "block_id", "trial_idx",
    "event_type", "stimulus_type", "is_response", "response_probability",
    "SERI", "Confidence", "QualityScore", "Evidence",
    # Exclude direct response-strength proxies from training features to reduce leakage.
    "evoked_snr",
    "response_rule_score",
}


def _prepare_matrix(
    df: pd.DataFrame,
    label_col: str = "is_response",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrix X and label vector y."""
    work = df.copy()

    if label_col in work.columns:
        y = work[label_col].astype(int).to_numpy()
    else:
        raise ValueError(f"Missing required label column '{label_col}'.")

    # We intentionally stay numeric-only for model stability across sessions.
    numeric = work.select_dtypes(include=[np.number]).copy()
    feature_columns = [c for c in numeric.columns if c not in EXCLUDE_COLS]
    if not feature_columns:
        raise ValueError("No numeric feature columns available.")

    x = numeric[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    return x, y, feature_columns


def _build_model() -> Pipeline:
    """Build the stimulus response classification pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def _within_subject_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> float | None:
    """Per-subject cross-validated AUC, averaged across subjects."""
    unique_groups = np.unique(groups)
    aucs: list[float] = []

    for g in unique_groups:
        mask = groups == g
        xg, yg = x[mask], y[mask]
        if len(np.unique(yg)) < 2 or len(yg) < 6:
            continue
        try:
            cv = StratifiedKFold(n_splits=min(3, len(yg) // 2), shuffle=True, random_state=RANDOM_STATE)
            model = _build_model()
            preds = cross_val_predict(model, xg, yg, cv=cv, method="predict_proba", n_jobs=1)
            auc = roc_auc_score(yg, preds[:, 1])
            aucs.append(float(auc))
        except Exception:
            continue

    return float(np.mean(aucs)) if aucs else None


def _across_subject_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Leave-one-subject-out cross-validation."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {"across_subject_auc": None, "fold_aucs": []}

    logo = LeaveOneGroupOut()
    probas = np.full(len(y), np.nan)
    fold_aucs: list[float] = []

    for train_idx, test_idx in logo.split(x, y, groups):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = _build_model()
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)[:, 1]
        probas[test_idx] = p

        if len(np.unique(y_test)) >= 2:
            fold_aucs.append(float(roc_auc_score(y_test, p)))

    valid_mask = ~np.isnan(probas)
    if valid_mask.sum() > 0 and len(np.unique(y[valid_mask])) >= 2:
        overall_auc = float(roc_auc_score(y[valid_mask], probas[valid_mask]))
    else:
        overall_auc = None

    return {"across_subject_auc": overall_auc, "fold_aucs": fold_aucs}


def _oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Build out-of-fold probabilities for unbiased session scoring."""
    unique_groups = np.unique(groups)

    # First choice for clinical realism: leave one subject out.
    if len(unique_groups) >= 2:
        logo = LeaveOneGroupOut()
        probas = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in logo.split(x, y, groups):
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            model = _build_model()
            model.fit(x[train_idx], y_train)
            probas[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        valid = ~np.isnan(probas)
        if valid.any():
            return probas, "leave_one_subject_out"

    # Fallback for small/single-subject subsets during early experiments.
    min_class_count = int(np.min(np.bincount(y))) if len(np.unique(y)) >= 2 else 0
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        probas = cross_val_predict(_build_model(), x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        return probas, f"stratified_{n_splits}fold"

    # Last resort so downstream scoring code can still run without crashing.
    return np.full(len(y), 0.5, dtype=float), "none"


def train_stimulus_response(
    trial_df: pd.DataFrame,
    label_col: str = "is_response",
) -> dict[str, Any]:
    """Train stimulus response detection model.

    Parameters
    ----------
    trial_df : trial-level feature DataFrame with subject_id column
    label_col : binary label column name

    Returns
    -------
    Dict with: model, probabilities, metrics, calibration_method, artifact_paths
    """
    t_start = time.perf_counter()

    x, y, feature_cols = _prepare_matrix(trial_df, label_col=label_col)

    groups = trial_df["subject_id"].to_numpy() if "subject_id" in trial_df.columns else np.zeros(len(y))

    # OOF probabilities are used for scoring/reporting to avoid optimistic bias.
    oof_probabilities, scoring_probability_source = _oof_probabilities(x, y, groups)
    valid_oof = ~np.isnan(oof_probabilities)

    # Train a final deployable model on all currently available data.
    model = _build_model()
    model.fit(x, y)

    # Get calibrated probabilities
    n_samples = len(y)
    calibration_method = "isotonic" if n_samples >= 50 else "sigmoid"

    try:
        cal_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv=min(3, max(2, n_samples // 10)),
        )
        cal_model.fit(x, y)
        probabilities = cal_model.predict_proba(x)[:, 1]
    except Exception:
        logger.warning("Calibration failed; using raw probabilities")
        calibration_method = "none"
        probabilities = model.predict_proba(x)[:, 1]
        cal_model = model

    # Report metrics from OOF predictions whenever possible.
    metric_probs = oof_probabilities[valid_oof] if valid_oof.any() else probabilities
    metric_y = y[valid_oof] if valid_oof.any() else y
    train_auc = float(roc_auc_score(metric_y, metric_probs)) if len(np.unique(metric_y)) >= 2 else None
    brier = float(brier_score_loss(metric_y, metric_probs)) if len(np.unique(metric_y)) >= 2 else None
    accuracy = float(accuracy_score(metric_y, (metric_probs >= 0.5).astype(int)))

    # Cross-validation
    within_auc = _within_subject_auc(x, y, groups)
    across_cv = _across_subject_cv(x, y, groups)

    # Save model artifacts
    model_path = MODEL_DIR / "stimulus_response_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal_model, model_path)

    metrics_data = {
        "train_auc": train_auc,
        "brier_score": brier,
        "accuracy": accuracy,
        "within_subject_auc": within_auc,
        "across_subject_auc": across_cv["across_subject_auc"],
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "calibration_method": calibration_method,
        "scoring_probability_source": scoring_probability_source,
        "feature_columns": feature_cols,
    }

    metrics_path = MODEL_DIR / "stimulus_response_metrics.json"
    metrics_path.write_text(
        json.dumps({k: v for k, v in metrics_data.items() if k != "feature_columns"}, indent=2),
        encoding="utf-8",
    )

    t_end = time.perf_counter()
    logger.info(
        "Stimulus response model: AUC=%.3f, Brier=%.4f, time=%.1fs",
        train_auc or 0, brier or 0, t_end - t_start,
    )

    return {
        "model": cal_model,
        "probabilities": probabilities,
        "scoring_probabilities": oof_probabilities,
        "metrics": metrics_data,
        "calibration_method": calibration_method,
        "artifact_paths": {
            "model": str(model_path),
            "metrics": str(metrics_path),
        },
    }


def predict_stimulus_response(
    model: Any,
    trial_df: pd.DataFrame,
) -> np.ndarray:
    """Predict stimulus response probabilities for new trials."""
    x, _, _ = _prepare_matrix(trial_df)
    return model.predict_proba(x)[:, 1]
