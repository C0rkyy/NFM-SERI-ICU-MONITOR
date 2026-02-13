"""
NFM — Random Forest Classifier
================================
Trains a RandomForest model to predict *Functional Response*
(binary: 1 = responsive, 0 = non-responsive) from the extracted
epoch-level feature vectors.

Labelling heuristic
--------------------
Because PhysioNet EEGBCI does not ship with ground-truth
"functional responsiveness" labels, we derive pseudo-labels:

    label = 1  if the epoch's P300 amplitude is above the
               population median (i.e., the upper half of the
               amplitude distribution is deemed "responsive").

This is standard in exploratory EEG-BCI literature when no
clinician-annotated labels exist.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_SIZE, CV_FOLDS, RANDOM_STATE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, MODEL_DIR,
)

logger = logging.getLogger(__name__)

# Columns used as model inputs (must exist in the feature DF)
FEATURE_COLS = [
    "p300_amplitude", "p300_latency_ms", "p300_auc",
    "delta_power", "theta_power", "alpha_power", "beta_power",
    "avg_induced_power_change",
    "gci",
]


def _prepare_Xy(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build X matrix and binary y vector from the feature DataFrame.

    Missing feature columns are filled with 0.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns (filled with 0): %s", missing)
        for c in missing:
            df[c] = 0.0

    X = df[FEATURE_COLS].values.astype(np.float64)

    # Pseudo-label: above-median P300 amplitude → responsive (1)
    if "label" in df.columns:
        y = df["label"].values.astype(int)
    else:
        median_amp = np.median(df["p300_amplitude"].values)
        y = (df["p300_amplitude"].values >= median_amp).astype(int)

    return X, y, FEATURE_COLS


def train_random_forest(
    df: pd.DataFrame,
) -> Dict:
    """
    Train and evaluate a RandomForest classifier.

    Returns
    -------
    dict with keys:
        model, scaler, metrics, cv_results,
        y_test, y_prob, fpr, tpr, roc_auc
    """
    X, y, feat_names = _prepare_Xy(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )

    # Fit
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }
    logger.info("RF hold-out metrics: %s", metrics)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        clf, X_scaled, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "roc_auc"],
        return_train_score=False,
    )
    cv_summary = {k: float(np.mean(v)) for k, v in cv_results.items()
                  if k.startswith("test_")}
    logger.info("RF %d-fold CV: %s", CV_FOLDS, cv_summary)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "model": clf,
        "scaler": scaler,
        "feature_names": feat_names,
        "metrics": metrics,
        "cv_results": cv_summary,
        "y_test": y_test,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": metrics["roc_auc"],
    }


def predict_proba(
    model, scaler, df: pd.DataFrame,
) -> np.ndarray:
    """
    Return P(Functional Response) for every epoch in df.
    """
    X, _, _ = _prepare_Xy(df)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


def save_model(result: Dict, tag: str = "rf") -> Path:
    path = MODEL_DIR / f"{tag}_model.pkl"
    joblib.dump({"model": result["model"],
                 "scaler": result["scaler"],
                 "feature_names": result["feature_names"]}, path)
    logger.info("Saved RF model → %s", path)
    return path


def load_model(path: Path) -> Dict:
    return joblib.load(path)
