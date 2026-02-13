"""
NFM — FRS Sensitivity Analysis
================================
Quantifies how the final FRS changes when each weight is varied
by ±Δ while the others are re-normalised to sum to 1.

Output: a DataFrame with columns
    component, weight_delta, mean_frs_change, max_frs_change
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRS_WEIGHTS
from scoring.frs import compute_frs

logger = logging.getLogger(__name__)

DELTA_RANGE = np.arange(-0.10, 0.11, 0.02)   # ±10 % in 2 % steps


def sensitivity_analysis(
    df: pd.DataFrame,
    model_proba: np.ndarray | None = None,
    delta_range: np.ndarray = DELTA_RANGE,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity sweep.

    For each FRS weight component, perturb the weight by δ and
    re-normalise the remaining weights so they still sum to 1.
    Compute the resulting FRS and record:
        • mean absolute change in FRS vs. baseline
        • maximum absolute change

    Parameters
    ----------
    df : feature DataFrame (same format as compute_frs expects).
    model_proba : optional classifier probabilities.
    delta_range : array of deltas to test.

    Returns
    -------
    DataFrame with one row per (component, delta).
    """
    base_weights = dict(FRS_WEIGHTS)
    base_df = compute_frs(df, model_proba=model_proba, weights=base_weights)
    base_frs = base_df["frs"].values.astype(float)

    rows: List[Dict] = []
    components = list(base_weights.keys())

    for comp in components:
        for delta in delta_range:
            new_w = dict(base_weights)
            new_w[comp] = max(0.0, new_w[comp] + delta)

            # re-normalise remaining
            others = [k for k in components if k != comp]
            rem_sum = sum(new_w[k] for k in others)
            target = 1.0 - new_w[comp]
            if rem_sum > 0:
                for k in others:
                    new_w[k] = new_w[k] / rem_sum * target

            perturbed = compute_frs(df, model_proba=model_proba, weights=new_w)
            pfrs = perturbed["frs"].values.astype(float)

            diff = np.abs(pfrs - base_frs)
            rows.append({
                "component": comp,
                "delta": float(delta),
                "weight_new": new_w[comp],
                "mean_frs_change": float(diff.mean()),
                "max_frs_change": float(diff.max()),
            })

    result = pd.DataFrame(rows)
    logger.info("Sensitivity analysis complete → %s", result.shape)
    return result
