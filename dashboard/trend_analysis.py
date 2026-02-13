"""
NFM — Trend Analysis Panel
============================
Visualises how FRS evolves across sessions / time-points for a
given subject and across the full cohort.

Components
----------
• Line chart of per-subject mean FRS (session axis)
• Percentage change from previous session with arrow indicators
• Stability classification (Improving / Stable / Declining)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.interpretation_engine import (
    clinical_category,
    trend_label,
)

logger = logging.getLogger(__name__)


def _build_subject_trend(frs_df: pd.DataFrame, subject_id: int) -> pd.DataFrame:
    """
    Aggregate epoch-level FRS into session-level means for one subject.

    If only one session exists (common for batch processing), we
    synthesise a "per-epoch-block" trend so the clinician still sees
    temporal variation.
    """
    sdf = frs_df[frs_df["subject_id"] == subject_id].copy()

    if "session_id" in sdf.columns and sdf["session_id"].nunique() > 1:
        trend = (
            sdf.groupby("session_id")["frs"]
            .mean()
            .reset_index()
            .rename(columns={"frs": "mean_frs"})
        )
        trend["session_label"] = trend["session_id"].astype(str)
    else:
        # Synthesise 5-epoch rolling blocks as pseudo-sessions
        sdf = sdf.sort_values("epoch_idx").reset_index(drop=True)
        block_size = max(1, len(sdf) // 5)
        sdf["block"] = sdf.index // block_size
        trend = (
            sdf.groupby("block")["frs"]
            .mean()
            .reset_index()
            .rename(columns={"frs": "mean_frs", "block": "session_id"})
        )
        trend["session_label"] = [f"Block {i+1}" for i in range(len(trend))]

    # Compute deltas
    trend["mean_frs"] = pd.to_numeric(trend["mean_frs"], errors="coerce").fillna(0)
    trend["prev_frs"] = trend["mean_frs"].shift(1)
    trend["pct_change"] = (
        (trend["mean_frs"] - trend["prev_frs"]) / trend["prev_frs"].clip(lower=1) * 100
    )
    trend["pct_change"] = trend["pct_change"].fillna(0).replace([np.inf, -np.inf], 0)

    return trend


def render_trend_analysis(
    frs_df: pd.DataFrame,
    subject_id: int,
):
    """Render the trend analysis panel."""
    try:
        trend = _build_subject_trend(frs_df, subject_id)
    except Exception as exc:
        logger.warning("Trend computation failed for S%03d: %s", subject_id, exc)
        st.info("Unable to compute trend for this patient.")
        return

    if len(trend) < 2:
        st.info("Insufficient sessions for trend analysis.")
        return

    # ── Line chart ─────────────────────────────────────────────
    fig = go.Figure()
    # Sanitise series for Plotly (no NaN or inf)
    y_vals = trend["mean_frs"].tolist()
    y_vals = [float(v) if pd.notna(v) and np.isfinite(v) else 0.0 for v in y_vals]
    fig.add_trace(go.Scatter(
        x=trend["session_label"].tolist(),
        y=y_vals,
        mode="lines+markers+text",
        text=[f"{v:.0f}" for v in y_vals],
        textposition="top center",
        line=dict(color="#1565c0", width=3),
        marker=dict(size=10, color=[
            clinical_category(v)["color_hex"] for v in trend["mean_frs"]
        ]),
        name="FRS",
    ))

    # Reference bands
    fig.add_hrect(y0=0,  y1=20,  fillcolor="#ffcdd2", opacity=0.15,
                  annotation_text="Minimal", annotation_position="bottom left")
    fig.add_hrect(y0=21, y1=40,  fillcolor="#ffe0b2", opacity=0.12)
    fig.add_hrect(y0=41, y1=70,  fillcolor="#fff9c4", opacity=0.10)
    fig.add_hrect(y0=71, y1=100, fillcolor="#c8e6c9", opacity=0.12,
                  annotation_text="Strong", annotation_position="top left")

    fig.update_layout(
        yaxis=dict(range=[0, 105], title="FRS Score", dtick=20),
        xaxis_title="Session / Time Block",
        height=300,
        margin=dict(t=30, b=30, l=40, r=20),
        showlegend=False,
    )
    try:
        st.plotly_chart(fig, key=f"trend_chart_{subject_id}", width="stretch")
    except Exception as exc:
        logger.warning("Trend chart render failed for S%03d: %s", subject_id, exc)
        st.warning("Unable to render trend chart.")

    # ── Delta indicators ───────────────────────────────────────
    if trend.empty:
        return
    latest = trend.iloc[-1]
    pct = float(latest["pct_change"]) if pd.notna(latest["pct_change"]) else 0.0
    tl = trend_label(pct)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Latest FRS",
            f"{latest['mean_frs']:.0f}",
            delta=f"{pct:+.1f}%" if not np.isnan(pct) else None,
        )
    with c2:
        st.markdown(
            f'<div style="font-size:36px;text-align:center;color:{tl["color"]}">'
            f'{tl["arrow"]}</div>'
            f'<div style="text-align:center;font-weight:600;color:{tl["color"]}">'
            f'{tl["label"]}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        overall_change = float(trend["mean_frs"].iloc[-1] - trend["mean_frs"].iloc[0])
        st.metric("Overall Change", f"{overall_change:+.1f} pts")
