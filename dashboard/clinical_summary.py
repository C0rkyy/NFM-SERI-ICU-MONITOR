"""
NFM — Clinical Summary Panel
==============================
Streamlit components for the top-priority clinical summary:

• Large FRS gauge  (0–100)
• Colour-coded clinical category badge
• Auto-generated clinical interpretation paragraph
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.interpretation_engine import (
    clinical_category,
    generate_clinical_explanation,
)

logger = logging.getLogger(__name__)


def _safe_mean(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in df.columns:
        return default
    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return default
    return float(series.mean())


def _safe_unit_interval(value: float, default: float = 0.5) -> float:
    if not np.isfinite(value):
        return default
    return float(np.clip(value, 0.0, 1.0))


def render_frs_gauge(frs: float, subject_label: str = "") -> go.Figure:
    """Build a Plotly indicator gauge for the FRS score."""
    cat = clinical_category(frs)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=frs,
        number={"font": {"size": 72, "color": cat["color_hex"]},
                "suffix": ""},
        title={"text": subject_label,
               "font": {"size": 18, "color": "#555"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2,
                     "tickcolor": "#ccc", "dtick": 20},
            "bar": {"color": cat["color_hex"], "thickness": 0.35},
            "bgcolor": "#f5f5f5",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],  "color": "#ffcdd2"},
                {"range": [21, 40], "color": "#ffe0b2"},
                {"range": [41, 70], "color": "#fff9c4"},
                {"range": [71, 100],"color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": cat["color_hex"], "width": 4},
                "thickness": 0.85,
                "value": frs,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_clinical_summary(
    frs_df: pd.DataFrame,
    subject_id: int,
):
    """
    Render the full clinical summary panel for one subject.

    Parameters
    ----------
    frs_df : epoch-level FRS DataFrame (already filtered for subject)
    subject_id : int
    """
    mean_frs = _safe_mean(frs_df, "frs", default=0.0)
    cat = clinical_category(mean_frs)

    # ── Header row: Gauge + Category badge + KPIs ──────────────
    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        try:
            fig = render_frs_gauge(mean_frs, f"Patient S{subject_id:03d}")
            st.plotly_chart(fig, key=f"frs_gauge_{subject_id}", width="stretch")
        except Exception as exc:
            logger.warning("FRS gauge render failed for S%03d: %s", subject_id, exc)
            st.metric("FRS Score", f"{mean_frs:.0f}")

    with col_info:
        # Category badge
        st.markdown(
            f'<div style="background:{cat["color_hex"]};color:white;'
            f'padding:12px 24px;border-radius:8px;display:inline-block;'
            f'font-size:24px;font-weight:700;margin-bottom:12px;">'
            f'{cat["label"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # KPI metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        erp_val = _safe_unit_interval(_safe_mean(frs_df, "norm_erp_amplitude", default=0.5), default=0.5)
        psd_val = _safe_unit_interval(_safe_mean(frs_df, "norm_band_power_shift", default=0.5), default=0.5)
        prob_val = _safe_unit_interval(_safe_mean(frs_df, "norm_model_probability", default=0.5), default=0.5)
        gci_val = _safe_unit_interval(_safe_mean(frs_df, "norm_connectivity_index", default=0.5), default=0.5)
        with kpi1:
            st.metric("Cortical Response", f"{erp_val:.0%}")
        with kpi2:
            st.metric("Spectral Shift", f"{psd_val:.0%}")
        with kpi3:
            st.metric("Model Confidence", f"{prob_val:.0%}")
        with kpi4:
            st.metric("Network Integrity", f"{gci_val:.0%}")

        # Clinical explanation
        explanation = generate_clinical_explanation(
            frs=mean_frs,
            norm_erp=erp_val,
            norm_psd=psd_val,
            model_prob=prob_val,
            norm_gci=gci_val,
        )
        if explanation:
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:4px solid {cat["color_hex"]};'
                f'padding:16px;border-radius:4px;font-size:15px;line-height:1.6;'
                f'color:#212529;">'
                f'{explanation}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Clinical interpretation unavailable for this patient.")
