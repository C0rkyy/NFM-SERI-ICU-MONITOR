"""
Drift monitoring module.

Detects feature drift (PSI / KS-based), prediction distribution shift,
and calibration drift. Produces aggregate alert levels (green/yellow/red).

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schemas.clinical_validation import AlertLevel, DriftSnapshot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Feature drift (PSI-based)
# ═══════════════════════════════════════════════════════════════

def _psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-8
    bins = np.linspace(
        min(baseline.min(), current.min()) - eps,
        max(baseline.max(), current.max()) + eps,
        n_bins + 1,
    )
    bl_counts = np.histogram(baseline, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    bl_pct = bl_counts / (bl_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Avoid log(0)
    bl_pct = np.clip(bl_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = float(np.sum((cur_pct - bl_pct) * np.log(cur_pct / bl_pct)))
    return psi


def _ks_stat(baseline: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic."""
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(baseline, current)
    return float(stat)


def compute_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """Compute PSI for each numeric feature column."""
    if feature_cols is None:
        feature_cols = [c for c in baseline_df.columns if baseline_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]
        feature_cols = [c for c in feature_cols if c in current_df.columns]

    results: dict[str, float] = {}
    for col in feature_cols:
        bl = baseline_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(bl) < 5 or len(cur) < 5:
            continue
        results[col] = _psi(bl, cur)
    return results


# ═══════════════════════════════════════════════════════════════
# Prediction drift
# ═══════════════════════════════════════════════════════════════

def compute_prediction_drift(
    baseline_probs: np.ndarray,
    current_probs: np.ndarray,
) -> dict[str, float]:
    """Compute distribution shift metrics on predicted probabilities."""
    baseline_probs = np.asarray(baseline_probs, dtype=float)
    current_probs = np.asarray(current_probs, dtype=float)

    bl_clean = baseline_probs[~np.isnan(baseline_probs)]
    cur_clean = current_probs[~np.isnan(current_probs)]

    if len(bl_clean) < 5 or len(cur_clean) < 5:
        return {"psi": 0.0, "ks_stat": 0.0, "mean_shift": 0.0}

    return {
        "psi": _psi(bl_clean, cur_clean),
        "ks_stat": _ks_stat(bl_clean, cur_clean),
        "mean_shift": float(cur_clean.mean() - bl_clean.mean()),
    }


# ═══════════════════════════════════════════════════════════════
# Calibration drift
# ═══════════════════════════════════════════════════════════════

def compute_calibration_drift(
    baseline_ece: float,
    current_ece: float,
) -> float:
    """Absolute change in ECE."""
    return abs(current_ece - baseline_ece)


# ═══════════════════════════════════════════════════════════════
# Aggregate alerting
# ═══════════════════════════════════════════════════════════════

def _aggregate_alert(
    feature_shifts: dict[str, float],
    prediction_shifts: dict[str, float],
    calibration_drift: float,
    *,
    psi_yellow: float = 0.10,
    psi_red: float = 0.25,
    cal_yellow: float = 0.05,
    cal_red: float = 0.10,
) -> AlertLevel:
    """Determine aggregate alert level."""
    max_feat_psi = max(feature_shifts.values()) if feature_shifts else 0.0
    pred_psi = prediction_shifts.get("psi", 0.0)
    max_psi = max(max_feat_psi, pred_psi)

    if max_psi >= psi_red or calibration_drift >= cal_red:
        return AlertLevel.RED
    if max_psi >= psi_yellow or calibration_drift >= cal_yellow:
        return AlertLevel.YELLOW
    return AlertLevel.GREEN


def build_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_probs: np.ndarray | None = None,
    current_probs: np.ndarray | None = None,
    baseline_ece: float = 0.0,
    current_ece: float = 0.0,
    feature_cols: list[str] | None = None,
) -> DriftSnapshot:
    """Build a full drift snapshot."""
    feature_shifts = compute_feature_drift(baseline_df, current_df, feature_cols)

    pred_shifts: dict[str, float] = {}
    if baseline_probs is not None and current_probs is not None:
        pred_shifts = compute_prediction_drift(baseline_probs, current_probs)

    cal_drift = compute_calibration_drift(baseline_ece, current_ece)
    alert = _aggregate_alert(feature_shifts, pred_shifts, cal_drift)

    details_parts: list[str] = []
    if feature_shifts:
        top_feat = max(feature_shifts, key=feature_shifts.get)  # type: ignore[arg-type]
        details_parts.append(f"Top feature PSI: {top_feat}={feature_shifts[top_feat]:.4f}")
    if pred_shifts:
        details_parts.append(f"Prediction PSI={pred_shifts.get('psi', 0):.4f}")
    details_parts.append(f"Calibration drift={cal_drift:.4f}")

    return DriftSnapshot(
        feature_shift_metrics=feature_shifts,
        prediction_shift_metrics=pred_shifts,
        calibration_shift=cal_drift,
        alert_level=alert,
        details="; ".join(details_parts),
    )


def save_drift_report(snapshot: DriftSnapshot, output_dir: Path) -> Path:
    """Save drift report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "drift_report.json"
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved drift report: %s (alert=%s)", path, snapshot.alert_level.value)
    return path


def plot_drift_dashboard(
    snapshot: DriftSnapshot,
    save_path: Path,
) -> None:
    """Plot drift dashboard summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Feature PSI
    ax = axes[0]
    feats = snapshot.feature_shift_metrics
    if feats:
        sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:10]
        names, vals = zip(*sorted_feats)
        short_names = [n[:15] for n in names]
        colors = ["#d9534f" if v >= 0.25 else "#f0ad4e" if v >= 0.10 else "#5cb85c" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("PSI")
        ax.set_title("Feature Drift (top 10)")
        ax.axvline(0.10, color="orange", ls="--", lw=0.8)
        ax.axvline(0.25, color="red", ls="--", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No feature data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Feature Drift")

    # 2) Prediction drift
    ax = axes[1]
    pred = snapshot.prediction_shift_metrics
    if pred:
        bars = ["PSI", "KS stat", "Mean shift"]
        vals = [pred.get("psi", 0), pred.get("ks_stat", 0), abs(pred.get("mean_shift", 0))]
        ax.bar(bars, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title("Prediction Drift")
    else:
        ax.text(0.5, 0.5, "No prediction data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Prediction Drift")

    # 3) Alert level
    ax = axes[2]
    alert_color = {"green": "#5cb85c", "yellow": "#f0ad4e", "red": "#d9534f"}
    c = alert_color.get(snapshot.alert_level.value, "#999999")
    ax.add_patch(plt.Circle((0.5, 0.5), 0.35, color=c, transform=ax.transAxes))
    ax.text(0.5, 0.5, snapshot.alert_level.value.upper(),
            ha="center", va="center", fontsize=18, fontweight="bold",
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Overall Alert")

    fig.suptitle("Drift Monitoring Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved drift dashboard: %s", save_path)
