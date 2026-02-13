"""
NFM — Model Evaluation Utilities
===================================
Shared helper functions for metric computation, ROC plotting,
and results serialisation.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_roc_curve(
    results: Dict,
    title: str = "ROC Curve",
    save_path: Path | None = None,
) -> Path:
    """
    Plot and save the ROC curve from a model result dict.

    Parameters
    ----------
    results : dict with keys fpr, tpr, roc_auc
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(results["fpr"], results["tpr"],
            lw=2, label=f'AUC = {results["roc_auc"]:.3f}')
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = save_path or RESULTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)
    return path


def plot_dual_roc(
    rf_results: Dict,
    cnn_results: Dict,
    save_path: Path | None = None,
) -> Path:
    """Plot RF and CNN ROC curves on the same axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rf_results["fpr"], rf_results["tpr"], lw=2,
            label=f'RF  AUC = {rf_results["roc_auc"]:.3f}')
    ax.plot(cnn_results["fpr"], cnn_results["tpr"], lw=2,
            label=f'CNN AUC = {cnn_results["roc_auc"]:.3f}')
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="FPR", ylabel="TPR",
           title="Model Comparison — ROC",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = save_path or RESULTS_DIR / "roc_curve_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved dual ROC → %s", path)
    return path


def summarise_metrics(*results_dicts, names=None) -> str:
    """Pretty-print metrics from one or more model result dicts."""
    lines = []
    for i, r in enumerate(results_dicts):
        name = names[i] if names else f"Model {i+1}"
        lines.append(f"\n{'='*40}")
        lines.append(f"  {name}")
        lines.append(f"{'='*40}")
        for k, v in r.get("metrics", {}).items():
            lines.append(f"  {k:>12s}: {v:.4f}")
        cv = r.get("cv_results", {})
        if cv:
            lines.append("  — Cross-validation —")
            for k, v in cv.items():
                lines.append(f"  {k:>25s}: {v:.4f}")
    return "\n".join(lines)
