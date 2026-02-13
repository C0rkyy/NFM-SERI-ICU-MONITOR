from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from config import ACTIVE_BASELINE_WINDOW, ACTIVE_IMAGERY_WINDOW

logger = logging.getLogger(__name__)

MU_BAND: tuple[float, float] = (8.0, 12.0)
BETA_BAND: tuple[float, float] = (13.0, 30.0)
ROI_CHANNELS: tuple[str, ...] = ("C3", "C4", "CZ")


def _window_mask(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    return (times >= window[0]) & (times <= window[1])


def _bandpower(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    nperseg = min(data.shape[-1], int(max(sfreq, 32)))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if len(idx) == 0:
        return np.zeros(data.shape[:2], dtype=float)
    return np.trapezoid(psd[..., idx], freqs[idx], axis=-1)


def _safe_erd_ers(imagery: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return ((imagery - baseline) / (np.abs(baseline) + 1e-12)) * 100.0


def _resolve_roi_indices(ch_names: list[str]) -> tuple[list[int], list[str]]:
    lookup = {name.upper(): idx for idx, name in enumerate(ch_names)}
    present_names: list[str] = []
    indices: list[int] = []
    for roi in ROI_CHANNELS:
        if roi in lookup:
            indices.append(lookup[roi])
            present_names.append(roi)
    return indices, present_names


def _build_metadata_if_missing(epochs: mne.Epochs) -> pd.DataFrame:
    code_to_label = {code: label for label, code in epochs.event_id.items()}
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(epochs.events):
        label = code_to_label.get(int(event[2]), "UNKNOWN")
        rows.append(
            {
                "epoch_idx": idx,
                "command_label": label,
                "is_imagery": int(label != "REST"),
                "task_label": label.replace("IMAGINE_", "") if label != "REST" else "REST",
            }
        )
    return pd.DataFrame(rows)


def _add_csp_features(trial_df: pd.DataFrame, imagery_data: np.ndarray, labels: np.ndarray) -> bool:
    unique = np.unique(labels)
    if len(unique) != 2:
        return False
    class_counts = [int(np.sum(labels == c)) for c in unique]
    if min(class_counts) < 3:
        return False

    try:
        from mne.decoding import CSP

        csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
        csp_features = csp.fit_transform(imagery_data, labels)
        for idx in range(csp_features.shape[1]):
            trial_df[f"csp_{idx + 1}"] = csp_features[:, idx]
        return True
    except Exception as exc:
        logger.warning("CSP feature extraction skipped: %s", exc)
        return False


def _add_riemann_features(trial_df: pd.DataFrame, imagery_data: np.ndarray) -> bool:
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except Exception:
        return False

    try:
        covs = Covariances().fit_transform(imagery_data)
        tangent = TangentSpace().fit_transform(covs)
        n_cols = min(6, tangent.shape[1])
        for idx in range(n_cols):
            trial_df[f"riemann_{idx + 1}"] = tangent[:, idx]
        return True
    except Exception as exc:
        logger.warning("Riemannian feature extraction skipped: %s", exc)
        return False


def _strongest_markers(summary_source: pd.DataFrame) -> list[str]:
    candidates: dict[str, float] = {}
    if "roi_mu_erd_ers_pct" in summary_source.columns:
        candidates["mu ERD/ERS ROI mean"] = float(summary_source["roi_mu_erd_ers_pct"].mean())
    if "roi_beta_erd_ers_pct" in summary_source.columns:
        candidates["beta ERD/ERS ROI mean"] = float(summary_source["roi_beta_erd_ers_pct"].mean())
    if "mu_erd_ers_pct" in summary_source.columns:
        candidates["mu ERD/ERS global mean"] = float(summary_source["mu_erd_ers_pct"].mean())
    if "beta_erd_ers_pct" in summary_source.columns:
        candidates["beta ERD/ERS global mean"] = float(summary_source["beta_erd_ers_pct"].mean())

    ordered = sorted(candidates.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [f"{name}: {value:+.2f}%" for name, value in ordered[:5]]


def extract_motor_imagery_features(
    epochs: mne.Epochs,
    metadata: pd.DataFrame | None = None,
    baseline_window: tuple[float, float] = ACTIVE_BASELINE_WINDOW,
    imagery_window: tuple[float, float] = ACTIVE_IMAGERY_WINDOW,
    include_csp: bool = True,
    include_riemann: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract motor-imagery features from command-locked EEG epochs."""
    trial_meta = metadata.copy() if metadata is not None else _build_metadata_if_missing(epochs)
    if "epoch_idx" not in trial_meta.columns:
        trial_meta["epoch_idx"] = np.arange(len(epochs))

    data = epochs.get_data()
    sfreq = float(epochs.info["sfreq"])
    times = epochs.times

    baseline_mask = _window_mask(times, baseline_window)
    imagery_mask = _window_mask(times, imagery_window)
    if not baseline_mask.any() or not imagery_mask.any():
        raise ValueError("Baseline or imagery window does not intersect epoch times.")

    baseline_data = data[:, :, baseline_mask]
    imagery_data = data[:, :, imagery_mask]

    mu_baseline = _bandpower(baseline_data, sfreq=sfreq, band=MU_BAND)
    mu_imagery = _bandpower(imagery_data, sfreq=sfreq, band=MU_BAND)
    beta_baseline = _bandpower(baseline_data, sfreq=sfreq, band=BETA_BAND)
    beta_imagery = _bandpower(imagery_data, sfreq=sfreq, band=BETA_BAND)

    mu_erd = _safe_erd_ers(mu_imagery, mu_baseline)
    beta_erd = _safe_erd_ers(beta_imagery, beta_baseline)

    roi_indices, roi_names = _resolve_roi_indices(epochs.ch_names)

    rows: list[dict[str, Any]] = []
    for idx in range(data.shape[0]):
        row = {
            "epoch_idx": int(trial_meta.loc[idx, "epoch_idx"] if idx < len(trial_meta) else idx),
            "mu_power_baseline": float(mu_baseline[idx].mean()),
            "mu_power_imagery": float(mu_imagery[idx].mean()),
            "beta_power_baseline": float(beta_baseline[idx].mean()),
            "beta_power_imagery": float(beta_imagery[idx].mean()),
            "mu_erd_ers_pct": float(mu_erd[idx].mean()),
            "beta_erd_ers_pct": float(beta_erd[idx].mean()),
        }

        if roi_indices:
            row["roi_mu_erd_ers_pct"] = float(mu_erd[idx, roi_indices].mean())
            row["roi_beta_erd_ers_pct"] = float(beta_erd[idx, roi_indices].mean())
            for channel_name, ch_idx in zip(roi_names, roi_indices):
                row[f"mu_erd_ers_pct_{channel_name}"] = float(mu_erd[idx, ch_idx])
                row[f"beta_erd_ers_pct_{channel_name}"] = float(beta_erd[idx, ch_idx])
        else:
            row["roi_mu_erd_ers_pct"] = float(mu_erd[idx].mean())
            row["roi_beta_erd_ers_pct"] = float(beta_erd[idx].mean())

        rows.append(row)

    trial_df = pd.DataFrame(rows)
    if len(trial_meta) == len(trial_df):
        for col in ["command_label", "is_imagery", "task_label"]:
            if col in trial_meta.columns:
                trial_df[col] = trial_meta[col].values

    if "is_imagery" not in trial_df.columns:
        trial_df["is_imagery"] = (trial_df.get("command_label", "REST") != "REST").astype(int)

    labels = trial_df["is_imagery"].astype(int).values
    csp_used = include_csp and _add_csp_features(trial_df, imagery_data, labels)
    riemann_used = include_riemann and _add_riemann_features(trial_df, imagery_data)

    imagery_trials = trial_df[trial_df["is_imagery"] == 1]
    if len(imagery_trials) > 1:
        mu_vals = imagery_trials["roi_mu_erd_ers_pct"].to_numpy(dtype=float)
        consistency = float(np.clip(1.0 - (np.std(mu_vals) / (abs(np.mean(mu_vals)) + 1e-6)), 0.0, 1.0))
    else:
        consistency = 0.0

    strongest = _strongest_markers(imagery_trials if len(imagery_trials) else trial_df)
    summary_row = {
        "n_trials": int(len(trial_df)),
        "n_imagery_trials": int(trial_df["is_imagery"].sum()),
        "roi_channels_available": ",".join(roi_names),
        "trial_consistency": consistency,
        "mu_erd_mean": float(imagery_trials["roi_mu_erd_ers_pct"].mean()) if len(imagery_trials) else 0.0,
        "beta_erd_mean": float(imagery_trials["roi_beta_erd_ers_pct"].mean()) if len(imagery_trials) else 0.0,
        "csp_used": csp_used,
        "riemann_used": riemann_used,
        "strongest_markers": " | ".join(strongest),
    }
    summary_df = pd.DataFrame([summary_row])

    summary_details: dict[str, Any] = {
        "roi_channels_available": roi_names,
        "strongest_markers": strongest,
        "trial_consistency": consistency,
        "csp_used": csp_used,
        "riemann_used": riemann_used,
    }

    return trial_df, summary_df, summary_details
