"""
Prospective workflow simulation.

Implements "silent mode" — model runs but does not affect clinical decisions.
Tracks operational KPIs: latency, failure rate, invalid sessions, repeatability.

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_prospective_run(
    results_df: pd.DataFrame,
    prob_col: str = "response_probability",
    score_col: str = "SERI",
    session_col: str = "session_id",
    subject_col: str = "subject_id",
    quality_col: str = "QualityScore",
    min_quality: float = 40.0,
    min_trials: int = 5,
) -> dict[str, Any]:
    """Simulate prospective workflow and compute operational metrics.

    "Silent mode": model produces scores but they are logged, not acted on.
    Returns workflow metrics and per-session ops log.
    """
    t_start = time.perf_counter()

    ops_log: list[dict[str, Any]] = []
    invalid_sessions: list[dict[str, Any]] = []

    if results_df.empty:
        return {
            "metrics": _empty_metrics(),
            "ops_log": pd.DataFrame(),
            "invalid_sessions": pd.DataFrame(),
        }

    group_cols = [c for c in [subject_col, session_col] if c in results_df.columns]
    if not group_cols:
        group_cols = [results_df.index.name or "index"]

    for keys, grp in results_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        entry: dict[str, Any] = {}
        for col, val in zip(group_cols, keys):
            entry[col] = val

        n_trials = len(grp)
        entry["n_trials"] = n_trials

        quality = float(grp[quality_col].mean()) if quality_col in grp.columns else 50.0
        entry["quality"] = quality

        score = float(grp[score_col].mean()) if score_col in grp.columns else 0.0
        entry["score"] = score

        has_prob = prob_col in grp.columns
        probs = grp[prob_col].values if has_prob else np.array([])
        entry["mean_prob"] = float(probs.mean()) if len(probs) > 0 else 0.0

        # Validity checks
        is_valid = True
        reason = ""
        if n_trials < min_trials:
            is_valid = False
            reason = f"too_few_trials ({n_trials} < {min_trials})"
        elif quality < min_quality:
            is_valid = False
            reason = f"low_quality ({quality:.1f} < {min_quality})"
        elif has_prob and np.isnan(probs).any():
            is_valid = False
            reason = "nan_probabilities"

        entry["valid"] = is_valid
        entry["invalid_reason"] = reason
        entry["mode"] = "silent"

        ops_log.append(entry)
        if not is_valid:
            invalid_sessions.append(entry)

    t_end = time.perf_counter()

    ops_df = pd.DataFrame(ops_log)
    invalid_df = pd.DataFrame(invalid_sessions)

    # Repeatability: for subjects with >1 session, compute score ICC proxy
    repeatability = _compute_repeatability(ops_df, subject_col, "score")

    n_total = len(ops_df)
    n_valid = int(ops_df["valid"].sum()) if "valid" in ops_df.columns else 0
    n_invalid = n_total - n_valid

    metrics = {
        "n_sessions": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_rate": n_invalid / n_total if n_total > 0 else 0.0,
        "mean_quality": float(ops_df["quality"].mean()) if not ops_df.empty else 0.0,
        "mean_score": float(ops_df["score"].mean()) if not ops_df.empty else 0.0,
        "latency_s": round(t_end - t_start, 4),
        "repeatability_icc_proxy": repeatability,
        "failure_rate": n_invalid / n_total if n_total > 0 else 0.0,
    }

    return {
        "metrics": metrics,
        "ops_log": ops_df,
        "invalid_sessions": invalid_df,
    }


def _compute_repeatability(
    ops_df: pd.DataFrame,
    subject_col: str,
    score_col: str,
) -> float:
    """Compute a simple ICC proxy (within-subject variance / total variance)."""
    if subject_col not in ops_df.columns or score_col not in ops_df.columns:
        return 0.0

    multi = ops_df.groupby(subject_col).filter(lambda g: len(g) > 1)
    if len(multi) < 4:
        return 0.0

    total_var = float(multi[score_col].var())
    if total_var < 1e-12:
        return 1.0

    within_var = float(multi.groupby(subject_col)[score_col].var().mean())
    icc_proxy = 1.0 - (within_var / total_var)
    return float(np.clip(icc_proxy, 0.0, 1.0))


def _empty_metrics() -> dict[str, Any]:
    return {
        "n_sessions": 0,
        "n_valid": 0,
        "n_invalid": 0,
        "invalid_rate": 0.0,
        "mean_quality": 0.0,
        "mean_score": 0.0,
        "latency_s": 0.0,
        "repeatability_icc_proxy": 0.0,
        "failure_rate": 0.0,
    }


def save_prospective_outputs(
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save prospective workflow outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    ops_log = result.get("ops_log")
    if isinstance(ops_log, pd.DataFrame) and not ops_log.empty:
        p = output_dir / "prospective_workflow_metrics.csv"
        ops_log.to_csv(p, index=False)
        paths["ops_log"] = p

    invalid = result.get("invalid_sessions")
    if isinstance(invalid, pd.DataFrame) and not invalid.empty:
        p = output_dir / "invalid_sessions_log.csv"
        invalid.to_csv(p, index=False)
        paths["invalid_log"] = p
    else:
        # Write empty file with headers
        p = output_dir / "invalid_sessions_log.csv"
        pd.DataFrame(columns=["session_id", "subject_id", "valid", "invalid_reason"]).to_csv(p, index=False)
        paths["invalid_log"] = p

    logger.info("Saved prospective outputs: %s", list(paths.keys()))
    return paths
