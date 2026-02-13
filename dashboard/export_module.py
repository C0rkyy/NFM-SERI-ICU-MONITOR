"""
NFM — Data Export Module
==========================
Handles generation and download of all structured research exports:

1. Full-dataset CSV  (features + model outputs + FRS + clinical category)
2. Per-subject summary JSON
3. Dashboard-ready download buttons
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from dashboard.interpretation_engine import clinical_category
from dashboard.utils import _to_float


def build_full_export(
    frs_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all available data into one research-grade export DataFrame.

    Adds:
    • clinical_category (text)
    • clinical_color     (hex)
    """
    if features_df is not None and len(features_df) > 0:
        # Developer note:
        # we merge conservatively and keep feature columns intact.
        # FRS columns are appended only when missing.
        extra_cols = [c for c in frs_df.columns if c not in features_df.columns]
        merge_keys = ["subject_id", "epoch_idx"]
        merge_keys = [k for k in merge_keys if k in frs_df.columns and k in features_df.columns]
        if merge_keys:
            export = features_df.merge(
                frs_df[merge_keys + extra_cols],
                on=merge_keys, how="left",
            )
        else:
            export = frs_df.copy()
    else:
        export = frs_df.copy()

    if "frs" in export.columns:
        export["clinical_category"] = export["frs"].apply(
            lambda v: clinical_category(v)["label"]
        )
        export["clinical_color"] = export["frs"].apply(
            lambda v: clinical_category(v)["color_name"]
        )

    return export


def build_subject_json(
    frs_df: pd.DataFrame,
) -> Dict:
    """
    Build a nested dict {subject_id: {...summary...}} for JSON export.
    """
    result = {}
    for sid, grp in frs_df.groupby("subject_id"):
        frs_mean = _to_float(grp["frs"].mean(), 0.0)
        cat = clinical_category(frs_mean)
        frs_std = _to_float(grp["frs"].std(), 0.0)
        frs_min = _to_float(grp["frs"].min(), 0.0)
        frs_max = _to_float(grp["frs"].max(), 0.0)
        entry = {
            "subject_id": int(sid),
            "frs_mean": round(frs_mean, 1),
            "frs_std": round(frs_std, 1),
            "frs_min": int(frs_min),
            "frs_max": int(frs_max),
            "n_epochs": len(grp),
            "clinical_category": cat["label"],
        }
        # Add component means opportunistically; keep schema tolerant to partial runs.
        for col in ["norm_erp_amplitude", "norm_band_power_shift",
                     "norm_model_probability", "norm_connectivity_index",
                     "p300_amplitude", "p300_latency_ms", "gci"]:
            if col in grp.columns:
                entry[col] = round(_to_float(grp[col].mean(), 0.0), 4)
        result[int(sid)] = entry

    return result


def save_exports(
    frs_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame],
    output_dir: Path,
):
    """
    Write export files to disk (called from main.py or on-demand).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full CSV is the most common artifact for external stats work.
    full = build_full_export(frs_df, features_df)
    full.to_csv(output_dir / "nfm_full_dataset.csv", index=False)

    # Subject JSON is easier for downstream apps/services to ingest.
    subj_json = build_subject_json(frs_df)
    with open(output_dir / "nfm_subject_summaries.json", "w") as f:
        json.dump(subj_json, f, indent=2)


@st.cache_data(show_spinner=False)
def _cached_full_csv(frs_ser: bytes, feat_ser: Optional[bytes]) -> tuple[str, int, int]:
    """Cache the expensive full-dataset CSV generation."""
    frs_df = pd.read_json(io.BytesIO(frs_ser))
    features_df = pd.read_json(io.BytesIO(feat_ser)) if feat_ser else None
    full_df = build_full_export(frs_df, features_df)
    csv_buf = io.StringIO()
    full_df.to_csv(csv_buf, index=False)
    return csv_buf.getvalue(), len(full_df), len(full_df.columns)


@st.cache_data(show_spinner=False)
def _cached_subject_json(frs_ser: bytes) -> tuple[str, int]:
    """Cache the expensive subject-JSON generation."""
    frs_df = pd.read_json(io.BytesIO(frs_ser))
    subj_json = build_subject_json(frs_df)
    return json.dumps(subj_json, indent=2), len(subj_json)


@st.cache_data(show_spinner=False)
def _cached_frs_csv(frs_ser: bytes) -> tuple[str, int]:
    """Cache the FRS-only CSV generation."""
    frs_df = pd.read_json(io.BytesIO(frs_ser))
    csv_buf = io.StringIO()
    frs_df.to_csv(csv_buf, index=False)
    return csv_buf.getvalue(), len(frs_df)


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to bytes for use as a cache key."""
    buf = io.BytesIO()
    df.to_json(buf)
    return buf.getvalue()


def render_export_panel(
    frs_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
):
    """Render export buttons in the dashboard."""
    st.markdown("##### 📦 Research Data Export")
    st.caption("Download complete datasets for external analysis.")

    # Serialize once; these byte payloads are stable cache keys for Streamlit.
    try:
        frs_ser = _df_to_bytes(frs_df)
        feat_ser = _df_to_bytes(features_df) if features_df is not None else None
    except Exception:
        st.warning("Unable to prepare data for export.")
        return

    c1, c2, c3 = st.columns(3)

    # Full CSV (cached): can be large, so we precompute once.
    with c1:
        try:
            csv_data, nrows, ncols = _cached_full_csv(frs_ser, feat_ser)
            st.download_button(
                "📥  Full Dataset (CSV)",
                data=csv_data,
                file_name="nfm_full_dataset.csv",
                mime="text/csv",
                key="exp_full_csv",
            )
            st.caption(f"{nrows} rows × {ncols} columns")
        except Exception:
            st.caption("Full dataset export unavailable.")

    # Subject JSON (cached): useful for API-style post-processing.
    with c2:
        try:
            json_str, n_subjects = _cached_subject_json(frs_ser)
            st.download_button(
                "📥  Subject Summaries (JSON)",
                data=json_str,
                file_name="nfm_subject_summaries.json",
                mime="application/json",
                key="exp_subj_json",
            )
            st.caption(f"{n_subjects} subject(s)")
        except Exception:
            st.caption("Subject JSON export unavailable.")

    # Lightweight fallback export for quick charting.
    with c3:
        try:
            frs_csv, n_epochs = _cached_frs_csv(frs_ser)
            st.download_button(
                "📥  FRS Results (CSV)",
                data=frs_csv,
                file_name="nfm_frs_results.csv",
                mime="text/csv",
                key="exp_frs_csv",
            )
            st.caption(f"{n_epochs} epochs")
        except Exception:
            st.caption("FRS export unavailable.")
