"""
NFM — Functional Responsiveness Score (FRS)
=============================================

Mathematical Definition
-----------------------
The FRS is a composite index ranging from **0** (no measurable
functional response) to **100** (maximal response).  It aggregates
four normalised sub-scores:

    FRS_raw = w₁·x₁ + w₂·x₂ + w₃·x₃ + w₄·x₄

where

    x₁ = normalised ERP amplitude        (P300 peak)
    x₂ = normalised band-power shift     (mean relative change)
    x₃ = model probability               (P(Response) from classifier)
    x₄ = global connectivity index       (mean upper-triangle coherence)

Each component is min-max scaled to [0, 1] across the full sample
*before* weighting.

    FRS = round(100 × FRS_raw)   ∈ {0, 1, …, 100}

Weight Justification
--------------------
+---+----------------------------+--------+------------------------------------+
|   | Component                  | Weight | Rationale                          |
+---+----------------------------+--------+------------------------------------+
| 1 | ERP amplitude (P300)       |  0.30  | Gold-standard cortical response    |
| 2 | Band-power shift           |  0.25  | Reflects engagement / arousal      |
| 3 | Model probability          |  0.30  | Integrates multivariate features   |
| 4 | Connectivity index         |  0.15  | Network-level coherence measure    |
+---+----------------------------+--------+------------------------------------+
Sum = 1.00

The ERP amplitude and model probability receive the highest weights
because peer-reviewed literature (Polich 2007; Lulé et al. 2013)
consistently shows P300 amplitude and machine-learning–derived
probabilities are the most reliable single predictors of functional
cortical processing.  Connectivity is weighted lower because it is
noisier with low-density montages.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRS_WEIGHTS

logger = logging.getLogger(__name__)


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Scale to [0, 1].  Constant arrays map to 0.5."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def compute_frs(
    df: pd.DataFrame,
    model_proba: np.ndarray | None = None,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute the Functional Responsiveness Score for each epoch.

    Parameters
    ----------
    df : DataFrame with columns
        p300_amplitude, *_rel_change (at least one band),
        gci
    model_proba : ndarray, optional
        P(Functional Response) from the classifier.
        If None, the component is set to 0.5 (neutral).
    weights : dict, optional
        Override default FRS_WEIGHTS.

    Returns
    -------
    DataFrame with added columns:
        frs_raw, frs, and the four normalised components.
    """
    w = weights or FRS_WEIGHTS
    out = df.copy()

    # ── Component 1: normalised ERP amplitude ─────────────────
    erp = df["p300_amplitude"].values.astype(float)
    x1 = _minmax(erp)
    out["norm_erp_amplitude"] = x1

    # ── Component 2: normalised band-power shift ──────────────
    rel_cols = [c for c in df.columns if c.endswith("_rel_change")]
    if rel_cols:
        bps = df[rel_cols].mean(axis=1).values.astype(float)
    else:
        # fallback: use absolute alpha power
        bps = df.get("alpha_power", pd.Series(np.zeros(len(df)))).values.astype(float)
    x2 = _minmax(bps)
    out["norm_band_power_shift"] = x2

    # ── Component 3: model probability ────────────────────────
    if model_proba is not None:
        x3 = model_proba.astype(float)
    else:
        x3 = np.full(len(df), 0.5)
    out["norm_model_probability"] = x3

    # ── Component 4: connectivity index ───────────────────────
    gci = df.get("gci", pd.Series(np.zeros(len(df)))).values.astype(float)
    x4 = _minmax(gci)
    out["norm_connectivity_index"] = x4

    # ── Weighted sum ──────────────────────────────────────────
    frs_raw = (
        w["erp_amplitude"]     * x1 +
        w["band_power_shift"]  * x2 +
        w["model_probability"] * x3 +
        w["connectivity_index"] * x4
    )
    out["frs_raw"] = frs_raw
    out["frs"] = np.clip(np.round(100 * frs_raw), 0, 100).astype(int)

    logger.info("FRS computed — mean %.1f, std %.1f, range [%d, %d]",
                out["frs"].mean(), out["frs"].std(),
                out["frs"].min(), out["frs"].max())
    return out


def subject_frs_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate epoch-level FRS into per-subject summaries.

    Returns
    -------
    DataFrame indexed by subject_id with columns:
        frs_mean, frs_std, erp_amplitude_mean, psd_shift_mean
    """
    rel_cols = [c for c in df.columns if c.endswith("_rel_change")]
    agg = df.groupby("subject_id").agg(
        frs_mean=("frs", "mean"),
        frs_std=("frs", "std"),
        frs_var=("frs", "var"),
        erp_amplitude_mean=("p300_amplitude", "mean"),
        psd_shift_mean=(rel_cols[0] if rel_cols else "p300_amplitude", "mean"),
        model_prob_mean=("norm_model_probability", "mean"),
    ).reset_index()
    return agg
