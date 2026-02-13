"""
Validation split generators with leakage checks.

Implements center-split, time-split, and center+time split strategies.
Strict leakage guards: no session-ID overlap, optional same-patient isolation.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from schemas.clinical_validation import SplitType, ValidationSplitSpec

logger = logging.getLogger(__name__)


def _check_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    check_patient: bool = True,
) -> list[str]:
    """Return leakage warnings. Empty list → clean."""
    issues: list[str] = []

    if "session_id" in train_df.columns and "session_id" in test_df.columns:
        overlap = set(train_df["session_id"]) & set(test_df["session_id"])
        if overlap:
            issues.append(f"Session-ID leakage: {len(overlap)} shared IDs")

    if check_patient and "subject_id" in train_df.columns and "subject_id" in test_df.columns:
        subj_overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
        if subj_overlap:
            issues.append(f"Subject leakage: {len(subj_overlap)} shared subjects")

    return issues


def center_split(
    df: pd.DataFrame,
    test_centers: list[str],
    center_col: str = "center_id",
    *,
    check_patient_leakage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSplitSpec]:
    """Split by center: train on some centers, test on held-out centers."""
    if center_col not in df.columns:
        raise ValueError(f"Column '{center_col}' not found in DataFrame")

    test_mask = df[center_col].isin(test_centers)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    issues = _check_leakage(train_df, test_df, check_patient=check_patient_leakage)
    if issues:
        logger.warning("Center split leakage: %s", "; ".join(issues))

    train_ids = sorted(train_df["session_id"].astype(str).unique()) if "session_id" in train_df.columns else []
    test_ids = sorted(test_df["session_id"].astype(str).unique()) if "session_id" in test_df.columns else []

    spec = ValidationSplitSpec(
        split_type=SplitType.CENTER,
        train_ids=train_ids,
        test_ids=test_ids,
        description=f"Center split: test centers = {test_centers}",
    )

    logger.info("Center split: train=%d, test=%d (test_centers=%s)",
                len(train_df), len(test_df), test_centers)
    return train_df, test_df, spec


def time_split(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    time_col: str = "created_at",
    *,
    check_patient_leakage: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSplitSpec]:
    """Split by time: train on earlier sessions, test on later ones."""
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)

    train_df = df[df[time_col] <= cutoff_ts].copy()
    test_df = df[df[time_col] > cutoff_ts].copy()

    issues = _check_leakage(train_df, test_df, check_patient=check_patient_leakage)
    if issues:
        logger.warning("Time split leakage: %s", "; ".join(issues))

    train_ids = sorted(train_df["session_id"].astype(str).unique()) if "session_id" in train_df.columns else []
    test_ids = sorted(test_df["session_id"].astype(str).unique()) if "session_id" in test_df.columns else []

    spec = ValidationSplitSpec(
        split_type=SplitType.TIME,
        train_ids=train_ids,
        test_ids=test_ids,
        description=f"Time split: cutoff = {cutoff_ts.isoformat()}",
    )

    logger.info("Time split: train=%d (before %s), test=%d (after)",
                len(train_df), cutoff_ts.date(), len(test_df))
    return train_df, test_df, spec


def center_time_split(
    df: pd.DataFrame,
    test_centers: list[str],
    cutoff: str | pd.Timestamp,
    center_col: str = "center_id",
    time_col: str = "created_at",
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSplitSpec]:
    """Combined center + time split (strictest)."""
    if center_col not in df.columns:
        raise ValueError(f"Column '{center_col}' not found")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)

    test_mask = df[center_col].isin(test_centers) | (df[time_col] > cutoff_ts)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    issues = _check_leakage(train_df, test_df, check_patient=True)
    if issues:
        logger.warning("Center+time split issues: %s", "; ".join(issues))

    train_ids = sorted(train_df["session_id"].astype(str).unique()) if "session_id" in train_df.columns else []
    test_ids = sorted(test_df["session_id"].astype(str).unique()) if "session_id" in test_df.columns else []

    spec = ValidationSplitSpec(
        split_type=SplitType.CENTER_TIME,
        train_ids=train_ids,
        test_ids=test_ids,
        description=f"Center+time split: test_centers={test_centers}, cutoff={cutoff_ts.isoformat()}",
    )

    logger.info("Center+time split: train=%d, test=%d", len(train_df), len(test_df))
    return train_df, test_df, spec


def auto_split(
    df: pd.DataFrame,
    strategy: str,
    center_col: str = "center_id",
    time_col: str = "created_at",
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSplitSpec]:
    """Automatic split based on strategy name and data contents."""
    if strategy == "center":
        if center_col not in df.columns:
            raise ValueError(f"Center split requires '{center_col}' column")
        centers = sorted(df[center_col].unique())
        if len(centers) < 2:
            raise ValueError(f"Center split requires ≥2 centers, found {len(centers)}")
        test_centers = [centers[-1]]
        return center_split(df, test_centers, center_col)

    elif strategy == "time":
        if time_col not in df.columns:
            raise ValueError(f"Time split requires '{time_col}' column")
        df_tmp = df.copy()
        df_tmp[time_col] = pd.to_datetime(df_tmp[time_col], errors="coerce")
        cutoff = df_tmp[time_col].quantile(0.7)
        return time_split(df, cutoff, time_col)

    elif strategy == "center_time":
        has_center = center_col in df.columns and df[center_col].nunique() >= 2
        has_time = time_col in df.columns

        if has_center and has_time:
            centers = sorted(df[center_col].unique())
            test_centers = [centers[-1]]
            df_tmp = df.copy()
            df_tmp[time_col] = pd.to_datetime(df_tmp[time_col], errors="coerce")
            cutoff = df_tmp[time_col].quantile(0.7)
            return center_time_split(df, test_centers, cutoff, center_col, time_col)
        elif has_center:
            return center_split(df, [sorted(df[center_col].unique())[-1]], center_col)
        elif has_time:
            df_tmp = df.copy()
            df_tmp[time_col] = pd.to_datetime(df_tmp[time_col], errors="coerce")
            cutoff = df_tmp[time_col].quantile(0.7)
            return time_split(df, cutoff, time_col)
        else:
            raise ValueError("center_time split requires 'center_id' or 'created_at' column")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
