"""
NFM — Group Comparison Statistics
====================================
Splits subjects into **High-response** and **Low-response** groups
using a median-split on mean FRS, then computes:

• Independent two-sample t-test (Welch's)
• Effect size  (Cohen's d)
• ROC curve    (group membership as label)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def split_groups(
    subject_summary: pd.DataFrame,
    metric: str = "frs_mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Median-split subjects into High and Low groups.

    Returns
    -------
    high_df, low_df  — subsets of subject_summary.
    """
    median_val = subject_summary[metric].median()
    high = subject_summary[subject_summary[metric] >= median_val].copy()
    low = subject_summary[subject_summary[metric] < median_val].copy()
    high["group"] = "High"
    low["group"] = "Low"
    logger.info("Group split (median %.1f): High n=%d, Low n=%d",
                median_val, len(high), len(low))
    return high, low


def independent_ttest(
    high: pd.DataFrame,
    low: pd.DataFrame,
    metric: str = "frs_mean",
) -> Dict:
    """
    Welch's t-test for unequal variances.

    Returns dict: t_stat, p_value, df
    """
    t_stat, p_val = stats.ttest_ind(
        high[metric].values,
        low[metric].values,
        equal_var=False,
    )
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


def cohens_d(
    high: pd.DataFrame,
    low: pd.DataFrame,
    metric: str = "frs_mean",
) -> float:
    """
    Compute Cohen's d effect size:
        d = (M₁ − M₂) / s_pooled
    """
    a, b = high[metric].values, low[metric].values
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / (pooled_std + 1e-12)
    return float(d)


def group_roc(
    subject_summary: pd.DataFrame,
    metric: str = "frs_mean",
) -> Dict:
    """
    ROC analysis treating group membership as the positive class.

    Label: 1 = High, 0 = Low.
    Score: the continuous metric value.

    Returns fpr, tpr, roc_auc.
    """
    median_val = subject_summary[metric].median()
    labels = (subject_summary[metric].values >= median_val).astype(int)
    scores = subject_summary[metric].values
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = roc_auc_score(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "roc_auc": float(auc_val)}


def full_group_comparison(
    subject_summary: pd.DataFrame,
    metric: str = "frs_mean",
) -> Dict:
    """
    Run the complete group comparison battery.

    Returns
    -------
    dict with: high_df, low_df, ttest, cohens_d, roc
    """
    high, low = split_groups(subject_summary, metric)
    tt = independent_ttest(high, low, metric)
    d = cohens_d(high, low, metric)
    roc = group_roc(subject_summary, metric)

    logger.info("t(%.1f) = %.3f, p = %.4f, d = %.3f, AUC = %.3f",
                tt["t_stat"], tt["t_stat"], tt["p_value"], d, roc["roc_auc"])

    combined = pd.concat([high, low], ignore_index=True)

    return {
        "combined_df": combined,
        "high_df": high,
        "low_df": low,
        "ttest": tt,
        "cohens_d": d,
        "roc": roc,
    }
