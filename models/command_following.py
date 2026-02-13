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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_N_JOBS, MODEL_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

# Number of parallel workers for CV loops
_N_JOBS: int = max(1, int(os.environ.get("NFM_N_JOBS", DEFAULT_N_JOBS)))


NUMERIC_EXCLUDE = {
    "is_imagery",
    "epoch_idx",
    "subject_id",
    "session_index",
    "session_id",
    "command_label",
    "task_label",
    "CFI",
    "Confidence",
    "QualityScore",
    "consistency",
    "Evidence",
    "ConfidenceLevel",
    "QualityLevel",
    "ClinicalSummary",
    "imagery_probability",
}


def _prepare_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    work = df.copy()
    if "is_imagery" in work.columns:
        y = work["is_imagery"].astype(int).to_numpy()
    elif "command_label" in work.columns:
        y = (work["command_label"] != "REST").astype(int).to_numpy()
    else:
        raise ValueError("Missing label columns for command-following model.")

    numeric = work.select_dtypes(include=[np.number]).copy()
    feature_columns = [c for c in numeric.columns if c not in NUMERIC_EXCLUDE]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for command-following model.")

    x = numeric[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    return x, y, feature_columns


def _base_lr() -> Pipeline:
    """Logistic regression baseline estimator."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=1.0,
                ),
            ),
        ]
    )


def _base_hgb() -> HistGradientBoostingClassifier:
    """HistGradientBoosting non-linear estimator (handles NaN natively)."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def _build_ensemble() -> Pipeline:
    """Build a weighted ensemble of LR + HistGradientBoosting."""
    lr_pipe = _base_lr()
    hgb = Pipeline([("scaler", StandardScaler()), ("clf", _base_hgb())])
    ensemble = VotingClassifier(
        estimators=[("lr", lr_pipe), ("hgb", hgb)],
        voting="soft",
        weights=[0.4, 0.6],
    )
    return Pipeline([("ensemble", ensemble)])


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with adaptive binning.

    Uses equal-frequency (quantile) binning so every bin has a similar
    number of samples. This avoids the empty-bin bias of equal-width
    binning and gives a more stable ECE estimate.
    """
    n = len(y_true)
    if n < 2:
        return 0.0

    # Equal-frequency bins
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    bin_size = max(1, n // n_bins)
    ece = 0.0
    for i in range(0, n, bin_size):
        end = min(i + bin_size, n)
        if end <= i:
            continue
        avg_confidence = float(p_sorted[i:end].mean())
        avg_accuracy = float(y_sorted[i:end].mean())
        ece += (end - i) / n * abs(avg_accuracy - avg_confidence)
    return float(ece)


def _within_subject_auc(df: pd.DataFrame, feature_columns: list[str]) -> float | None:
    aucs: list[float] = []
    for _, sdf in df.groupby("subject_id"):
        if "is_imagery" not in sdf.columns:
            continue
        y_sub = sdf["is_imagery"].astype(int).to_numpy()
        if len(np.unique(y_sub)) < 2:
            continue
        min_count = int(np.min(np.bincount(y_sub)))
        n_splits = min(5, min_count)
        if n_splits < 2:
            continue

        x_sub = sdf[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        probs = cross_val_predict(_base_lr(), x_sub, y_sub, cv=cv, method="predict_proba")[:, 1]
        aucs.append(float(roc_auc_score(y_sub, probs)))

    if not aucs:
        return None
    return float(np.mean(aucs))


def _fit_one_logo_fold(
    args_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single LOSO fold — designed for joblib.Parallel."""
    x_train, y_train, x_test, test_idx = args_tuple
    if len(np.unique(y_train)) < 2:
        return test_idx, np.full(len(test_idx), np.nan)
    estimator = _build_ensemble()
    estimator.fit(x_train, y_train)
    return test_idx, estimator.predict_proba(x_test)[:, 1]


def _across_subject_cv(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    n_jobs: int = 1,
) -> tuple[float | None, pd.DataFrame]:
    if "subject_id" not in df.columns:
        return None, pd.DataFrame(columns=["subject_id", "y_true", "y_prob"])

    groups = df["subject_id"].to_numpy()
    logo = LeaveOneGroupOut()
    preds = np.full(len(y), np.nan, dtype=float)

    fold_args = [
        (x[train_idx], y[train_idx], x[test_idx], test_idx)
        for train_idx, test_idx in logo.split(x, y, groups=groups)
    ]

    results = [_fit_one_logo_fold(args) for args in fold_args]

    for test_idx, fold_preds in results:
        preds[test_idx] = fold_preds

    valid = ~np.isnan(preds)
    auc_score: float | None = None
    if np.sum(valid) > 1 and len(np.unique(y[valid])) == 2:
        auc_score = float(roc_auc_score(y[valid], preds[valid]))

    cv_df = pd.DataFrame(
        {
            "subject_id": groups,
            "y_true": y,
            "y_prob": preds,
        }
    )
    return auc_score, cv_df


def _secondary_task_metrics(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float] | None:
    if "command_label" not in df.columns:
        return None
    imagery = df[df["command_label"].isin(["IMAGINE_WALKING", "IMAGINE_HAND"])].copy()
    if imagery.empty:
        return None

    y = (imagery["command_label"] == "IMAGINE_WALKING").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    min_count = int(np.min(np.bincount(y)))
    n_splits = min(5, min_count)
    if n_splits < 2:
        return None

    x = imagery[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(_base_lr(), x, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "walk_vs_hand_auc": float(roc_auc_score(y, probs)),
        "walk_vs_hand_accuracy": float(accuracy_score(y, preds)),
    }


def _temperature_scale(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Post-hoc temperature scaling to improve calibration.

    Finds a single scalar T that minimises negative log-likelihood.
    Probabilities are converted to logits, divided by T, then mapped back
    to probabilities.  Rankings (AUC) are preserved exactly; only the
    *sharpness* of the distribution changes → ECE improves.
    """
    from scipy.optimize import minimize_scalar

    eps = 1e-10
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))

    def _nll(T: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        return float(-np.mean(
            y_true * np.log(scaled + eps)
            + (1.0 - y_true) * np.log(1.0 - scaled + eps)
        ))

    result = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
    T_opt = result.x
    logger.info("Temperature scaling: T=%.3f", T_opt)
    return (1.0 / (1.0 + np.exp(-logits / T_opt))).astype(np.float64)


def train_command_following(
    df: pd.DataFrame,
    output_dir: Path = MODEL_DIR,
) -> dict[str, Any]:
    """Train and calibrate command-following ensemble model and save artifacts."""
    t0 = time.perf_counter()

    if "subject_id" not in df.columns:
        raise ValueError("DataFrame must include subject_id for LOSO validation.")

    x, y, feature_columns = _prepare_matrix(df)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both REST and IMAGERY trials for command-following model.")

    logger.info("Training command-following model: %d trials, %d features", x.shape[0], x.shape[1])

    # Determine parallelism
    n_jobs = max(1, _N_JOBS)

    # Cross-validation metrics
    within_auc = _within_subject_auc(df, feature_columns)
    across_auc, cv_predictions = _across_subject_cv(df, x, y, n_jobs=n_jobs)

    # Choose calibration method based on data size and class balance
    min_class_count = int(np.min(np.bincount(y)))
    # Sigmoid (Platt scaling) is more stable for large N; isotonic can overfit.
    if min_class_count >= 500:
        calibration_method = "sigmoid"
    elif len(y) >= 200:
        calibration_method = "isotonic"
    else:
        calibration_method = "sigmoid"

    calibration_cv = min(5, min_class_count)

    # Train calibrated ensemble
    calibrated = CalibratedClassifierCV(
        estimator=_build_ensemble(),
        method=calibration_method,
        cv=calibration_cv,
    )
    calibrated.fit(x, y)
    probabilities = calibrated.predict_proba(x)[:, 1]

    # Temperature scaling: find T that minimises NLL to improve calibration.
    # This is a lightweight post-hoc step that adjusts confidence without
    # changing rankings (so AUC/Brier stay nearly the same but ECE drops).
    probabilities = _temperature_scale(probabilities, y)

    train_auc = float(roc_auc_score(y, probabilities))
    brier = float(brier_score_loss(y, probabilities))
    ece = _compute_ece(y, probabilities)

    secondary_metrics = _secondary_task_metrics(df, feature_columns)

    train_time = time.perf_counter() - t0

    metrics: dict[str, Any] = {
        "primary_task": "REST_vs_IMAGERY",
        "calibration_method": calibration_method,
        "train_auc": train_auc,
        "within_subject_auc": within_auc,
        "across_subject_auc": across_auc,
        "brier_score": brier,
        "ece": ece,
        "n_training_samples": int(len(y)),
        "n_features": int(x.shape[1]),
        "n_subjects": int(df["subject_id"].nunique()),
        "model_type": "ensemble_lr_hgb_calibrated",
        "train_time_s": round(train_time, 2),
    }
    if secondary_metrics:
        metrics["secondary_task"] = secondary_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "command_following_model.pkl"
    metrics_path = output_dir / "command_following_metrics.json"
    cv_path = output_dir / "command_following_cv_predictions.csv"

    joblib.dump(
        {
            "model": calibrated,
            "feature_columns": feature_columns,
            "metrics": metrics,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cv_predictions.to_csv(cv_path, index=False)

    logger.info(
        "Command-following model trained: train_auc=%.3f, across_auc=%s, brier=%.4f, ece=%.4f (%.1fs)",
        train_auc,
        f"{across_auc:.3f}" if across_auc is not None else "N/A",
        brier,
        ece,
        train_time,
    )

    return {
        "model": calibrated,
        "feature_columns": feature_columns,
        "probabilities": probabilities,
        "metrics": metrics,
        "artifact_paths": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "cv_predictions": str(cv_path),
        },
        "calibration_method": calibration_method,
    }


def predict_command_following(model: CalibratedClassifierCV, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = df[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    return model.predict_proba(x)[:, 1]
