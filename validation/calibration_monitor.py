"""
Calibration monitoring module.

Computes reliability curves, ECE, MCE, Brier scores, and calibration drift
over time buckets and by center.

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute reliability (calibration) curve data."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_means: list[float] = []
    bin_true_fracs: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_means.append(float(y_prob[mask].mean()))
        bin_true_fracs.append(float(y_true[mask].mean()))
        bin_counts.append(count)

    return {
        "bin_predicted_means": bin_means,
        "bin_true_fractions": bin_true_fracs,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    total = sum(curve["bin_counts"])
    if total == 0:
        return 0.0
    ece = 0.0
    for pred, true, cnt in zip(
        curve["bin_predicted_means"],
        curve["bin_true_fractions"],
        curve["bin_counts"],
    ):
        ece += (cnt / total) * abs(pred - true)
    return float(ece)


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if not curve["bin_predicted_means"]:
        return 0.0
    return float(max(
        abs(p - t)
        for p, t in zip(curve["bin_predicted_means"], curve["bin_true_fractions"])
    ))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_by_center(
    df: pd.DataFrame,
    prob_col: str = "response_probability",
    label_col: str = "reference_label",
    center_col: str = "center_id",
) -> dict[str, dict[str, float]]:
    """Compute calibration metrics per center."""
    result: dict[str, dict[str, float]] = {}
    if center_col not in df.columns or label_col not in df.columns or prob_col not in df.columns:
        return result

    for center, grp in df.groupby(center_col):
        y_true = grp[label_col].values
        y_prob = grp[prob_col].values
        valid = ~(np.isnan(y_true) | np.isnan(y_prob))
        if valid.sum() < 5:
            continue
        result[str(center)] = {
            "ece": compute_ece(y_true[valid], y_prob[valid]),
            "mce": compute_mce(y_true[valid], y_prob[valid]),
            "brier": compute_brier(y_true[valid], y_prob[valid]),
            "n_samples": int(valid.sum()),
        }
    return result


def calibration_by_time_bucket(
    df: pd.DataFrame,
    prob_col: str = "response_probability",
    label_col: str = "reference_label",
    time_col: str = "created_at",
    n_buckets: int = 4,
) -> dict[str, dict[str, float]]:
    """Compute calibration metrics over time quantile buckets."""
    result: dict[str, dict[str, float]] = {}
    if time_col not in df.columns or label_col not in df.columns or prob_col not in df.columns:
        return result

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, label_col, prob_col])

    if len(df) < n_buckets * 2:
        return result

    df["_bucket"] = pd.qcut(df[time_col].astype(int), q=n_buckets, labels=False, duplicates="drop")

    for bucket, grp in df.groupby("_bucket"):
        y_true = grp[label_col].values
        y_prob = grp[prob_col].values
        result[f"bucket_{bucket}"] = {
            "ece": compute_ece(y_true, y_prob),
            "mce": compute_mce(y_true, y_prob),
            "brier": compute_brier(y_true, y_prob),
            "n_samples": len(grp),
            "time_range": f"{grp[time_col].min()} — {grp[time_col].max()}",
        }
    return result


def build_calibration_report(
    df: pd.DataFrame,
    prob_col: str = "response_probability",
    label_col: str = "reference_label",
) -> dict[str, Any]:
    """Build a full calibration report."""
    valid = df.dropna(subset=[label_col, prob_col])
    if len(valid) < 5:
        return {"error": "Too few labelled samples for calibration analysis", "n_samples": len(valid)}

    y_true = valid[label_col].values
    y_prob = valid[prob_col].values

    curve = reliability_curve(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    mce = compute_mce(y_true, y_prob)
    brier = compute_brier(y_true, y_prob)

    by_center = calibration_by_center(valid, prob_col, label_col)
    by_time = calibration_by_time_bucket(valid, prob_col, label_col)

    report = {
        "overall": {
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "n_samples": len(valid),
        },
        "reliability_curve": curve,
        "by_center": by_center,
        "by_time": by_time,
        "interpretation": _interpret_calibration(ece, mce, brier),
    }
    return report


def _interpret_calibration(ece: float, mce: float, brier: float) -> str:
    """Plain-language interpretation of calibration metrics."""
    parts: list[str] = []
    if ece < 0.05:
        parts.append("Calibration is excellent (ECE < 0.05).")
    elif ece < 0.10:
        parts.append("Calibration is acceptable (ECE < 0.10).")
    else:
        parts.append(f"Calibration needs improvement (ECE = {ece:.3f}).")

    if brier < 0.15:
        parts.append("Brier score indicates good probability accuracy.")
    elif brier < 0.25:
        parts.append("Brier score is moderate — predictions have room for improvement.")
    else:
        parts.append(f"Brier score is high ({brier:.3f}) — probability estimates are unreliable.")

    return " ".join(parts)


def save_calibration_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Save calibration report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_report.json"

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(report, indent=2, default=_default), encoding="utf-8")
    logger.info("Saved calibration report: %s", path)
    return path


def plot_reliability_curve(
    report: dict[str, Any],
    save_path: Path,
) -> None:
    """Plot reliability (calibration) diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = report.get("reliability_curve", {})
    bin_means = curve.get("bin_predicted_means", [])
    bin_trues = curve.get("bin_true_fractions", [])

    if not bin_means:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(bin_means, bin_trues, "s-", color="#1f77b4", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    overall = report.get("overall", {})
    ece = overall.get("ece", 0)
    brier = overall.get("brier", 0)
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}\nBrier = {brier:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved reliability curve: %s", save_path)
