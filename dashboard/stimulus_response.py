"""
NFM — Stimulus Response Dashboard Tab
========================================
Replaces the command-following tab with stimulus-evoked response monitoring.

Uses language:
- "stimulus-evoked response indicator"
- "functional cortical responsiveness to external stimuli"
- "decision-support only, not a diagnosis"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import SESSION_STORE_DIR
from dashboard.utils import _to_float
from scoring.seri import seri_evidence_label, seri_confidence_label, seri_quality_label

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Clinical disclaimer (always visible)
# ═══════════════════════════════════════════════════════════════
_DISCLAIMER = (
    "**Decision-support only. Not a diagnosis.** "
    "This stimulus-evoked response indicator reflects functional cortical "
    "responsiveness to external stimuli. It does not measure consciousness "
    "and must be interpreted within the full clinical context by a qualified clinician."
)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
def load_stimulus_sessions(session_store_dir: Path = SESSION_STORE_DIR) -> pd.DataFrame:
    """Load stored stimulus-response session data from JSON files."""
    rows: list[dict[str, Any]] = []
    for path in sorted(session_store_dir.glob("session_s*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "scores" not in payload:
            continue

        scores = payload.get("scores", {})
        mode = str(payload.get("mode", "")).strip().lower()
        has_seri = scores.get("SERI") is not None

        # Keep stimulus sessions only. Never backfill SERI from CFI.
        if mode == "active":
            continue
        if not has_seri:
            continue

        feature_summary = payload.get("feature_summary", {})
        patient = payload.get("patient", {})

        strongest = feature_summary.get("strongest_markers", [])
        if isinstance(strongest, str):
            strongest = [m.strip() for m in strongest.split("|") if m.strip()]

        seri = _to_float(scores.get("SERI", 0.0))
        confidence = _to_float(scores.get("Confidence", 0.0))
        quality = _to_float(scores.get("Quality", scores.get("QualityScore", 0.0)))

        subject_raw = patient.get("subject_id", 0)
        try:
            subject_id = int(subject_raw)
        except (TypeError, ValueError):
            subject_id = 0
        if subject_id <= 0:
            # Try parsing from patient_id
            pid = patient.get("patient_id", "")
            if isinstance(pid, str):
                import re
                m = re.search(r"(\d+)", pid)
                if m:
                    subject_id = int(m.group(1))
            if subject_id <= 0:
                continue

        rows.append({
            "subject_id": subject_id,
            "session_id": payload.get("session_id", path.stem),
            "created_at": payload.get("created_at"),
            "SERI": seri,
            "Confidence": confidence,
            "QualityScore": quality,
            "evidence": seri_evidence_label(seri),
            "confidence_level": seri_confidence_label(confidence),
            "quality_level": seri_quality_label(quality),
            "trial_consistency": _to_float(feature_summary.get("trial_consistency", 0.0)),
            "strongest_markers": strongest,
            "mean_evoked_snr": _to_float(feature_summary.get("mean_evoked_snr", 0.0)),
            "disclaimer": payload.get("disclaimer", "Decision-support only. Not a diagnosis."),
            "mode": payload.get("mode", "stimulus"),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id", "session_id", "created_at", "SERI", "Confidence",
                "QualityScore", "evidence", "confidence_level", "quality_level",
                "trial_consistency", "strongest_markers", "mean_evoked_snr",
                "disclaimer", "mode",
            ]
        )

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df.sort_values("created_at", na_position="last").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Interpretation helpers
# ═══════════════════════════════════════════════════════════════
def _evidence_statement(evidence: str) -> str:
    if evidence == "High":
        return (
            "**High evidence** of stimulus-locked cortical modulation. "
            "The EEG pattern shows clear, time-locked responses to external stimuli."
        )
    if evidence == "Moderate":
        return (
            "**Moderate evidence** of stimulus-locked cortical modulation. "
            "Some EEG features show stimulus-evoked changes, but the response is not fully consistent."
        )
    return (
        "**Low evidence** of stimulus-locked cortical modulation. "
        "The EEG pattern does not show reliable stimulus-evoked responses in this session."
    )


def _clinical_note(seri: float, confidence: float, quality: float) -> str:
    if quality < 50:
        return (
            "**Data quality is limited.** Repeat acquisition after improving electrode contacts "
            "and reducing movement artifacts before making any clinical interpretation."
        )
    if seri >= 70 and confidence >= 70:
        return (
            "**Pattern is consistent with functional cortical responsiveness to external stimuli.** "
            "Use as a supportive decision-signal alongside the full clinical picture. "
            "Serial assessments strengthen interpretation."
        )
    if seri < 40:
        return (
            "**Current session does not show strong stimulus-evoked response signal.** "
            "Consider repeating the assessment and evaluating sedation level, time of day, "
            "sensory environment, and electrode quality. A single low result does not rule out "
            "cortical responsiveness."
        )
    return (
        "**Signal is intermediate.** Repeat in serial sessions and track the trend over time "
        "before incorporating into clinical decisions. Day-to-day variability is expected."
    )


def _what_this_means(seri: float, confidence: float, quality: float) -> str:
    parts: list[str] = []

    if seri >= 70:
        parts.append(
            "The brain showed clear, time-locked electrical changes in response to "
            "external stimuli, compared to pre-stimulus baseline."
        )
    elif seri >= 40:
        parts.append(
            "The brain showed some change in electrical patterns following stimulus "
            "presentation, but the response was not strong enough to be fully definitive."
        )
    else:
        parts.append(
            "In this session, the brain's electrical patterns following stimulus "
            "presentation were not clearly different from pre-stimulus baseline."
        )

    if confidence >= 70:
        parts.append(
            "We have high confidence in this measurement: data quality was adequate, "
            "the model is well-calibrated, and trial responses were consistent."
        )
    elif confidence >= 45:
        parts.append(
            "Confidence is moderate. Some factors (data quality, trial count, "
            "or response variability) limit certainty."
        )
    else:
        parts.append(
            "Confidence is low — interpret with extra caution. The data may have quality "
            "issues, too few usable trials, or inconsistent responses."
        )

    return " ".join(parts)


def _explain_marker(marker: str) -> str:
    lower = marker.lower()
    if "n100" in lower:
        return (
            "The N100 is an early negative wave (~100ms after stimulus), reflecting "
            "initial cortical detection of the stimulus. Its presence indicates the auditory "
            "or visual cortex is processing the input."
        )
    if "p200" in lower:
        return (
            "The P200 positive wave (~200ms) reflects higher-order stimulus evaluation. "
            "Its presence suggests cortical processing beyond basic detection."
        )
    if "p300" in lower:
        return (
            "The P300 positive wave (~300ms) is associated with attention and cognitive processing. "
            "Its presence suggests the brain is actively evaluating the stimulus."
        )
    if "alpha" in lower:
        return (
            "Alpha band (8-13 Hz) power change after stimulus indicates cortical engagement. "
            "Alpha suppression (decrease) typically reflects active processing."
        )
    if "beta" in lower:
        return (
            "Beta band (13-30 Hz) change during the response window contributes evidence "
            "of cortical activation above resting baseline levels."
        )
    if "snr" in lower:
        return (
            "Evoked SNR measures the signal-to-noise ratio of the stimulus response. "
            "Higher SNR means the brain's response is clearly distinguishable from background noise."
        )
    if "consistency" in lower:
        return (
            "Response consistency measures how stable the stimulus-evoked response was "
            "across repeated trials. Higher consistency strengthens the finding."
        )
    if "delta" in lower:
        return (
            "Delta band (0.5-4 Hz) changes may reflect large-scale cortical processing "
            "or arousal modulation in response to stimulation."
        )
    if "theta" in lower:
        return (
            "Theta band (4-8 Hz) modulation is associated with memory encoding and "
            "attention allocation to the stimulus."
        )
    return "This feature contributes to evidence of functional cortical responsiveness to stimuli."


# ═══════════════════════════════════════════════════════════════
# Full glossary
# ═══════════════════════════════════════════════════════════════
def _full_glossary() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Term": "SERI (Stimulus-Evoked Response Index)",
            "What it is": "Overall score (0-100) summarizing how strongly the EEG responds to external stimuli vs. baseline.",
            "How to read it": "Low (<40): weak response. Moderate (40-69): some response. High (>=70): clear response.",
            "Clinical note": "A single-session SERI should not be used alone. Track trend over serial assessments.",
        },
        {
            "Term": "Confidence (0-100)",
            "What it is": "How reliable this session's SERI is. Combines data quality, model calibration, trial consistency, sample size, and statistical uncertainty.",
            "How to read it": "Low (<45): interpret with caution. Moderate (45-69): reasonable certainty. High (>=70): well-supported result.",
            "Clinical note": "Low confidence means repeating the session under better conditions could give a different result.",
        },
        {
            "Term": "QualityScore (0-100)",
            "What it is": "Technical quality of the EEG recording for this session.",
            "How to read it": "Driven by: how many epochs survived artifact rejection, bad electrode channels, residual noise, and signal-to-noise ratio.",
            "Clinical note": "Quality < 50 means the recording had major technical issues — repeat before interpreting.",
        },
        {
            "Term": "Stimulus-evoked response",
            "What it is": "A measurable, time-locked change in brain electrical activity following presentation of an external stimulus (auditory or visual).",
            "How to read it": "Presence and strength are quantified by SERI. Stronger responses suggest more preserved cortical processing.",
            "Clinical note": "This is NOT a measure of consciousness. It indicates functional cortical responsiveness to external input.",
        },
        {
            "Term": "ERP (Event-Related Potential)",
            "What it is": "A specific pattern of voltage changes in the EEG that are time-locked to a stimulus event.",
            "How to read it": "Key components: N100 (early detection), P200 (evaluation), P300 (attention/cognition).",
            "Clinical note": "Presence of later components (P300) suggests higher-level cognitive processing of the stimulus.",
        },
        {
            "Term": "N100 component",
            "What it is": "A negative voltage peak ~100ms after stimulus, reflecting initial cortical detection.",
            "How to read it": "Larger (more negative) N100 = stronger initial stimulus detection.",
            "Clinical note": "Absence may indicate impaired sensory cortex function or heavily sedated state.",
        },
        {
            "Term": "P300 component",
            "What it is": "A positive voltage peak ~300ms after stimulus, reflecting attention and cognitive evaluation.",
            "How to read it": "Larger P300 = brain is actively processing and evaluating the stimulus.",
            "Clinical note": "P300 is one of the most clinically informative ERP components for assessing cortical function.",
        },
        {
            "Term": "Evoked SNR",
            "What it is": "Signal-to-noise ratio of the stimulus-evoked response compared to pre-stimulus baseline.",
            "How to read it": "SNR > 1 indicates the response exceeds baseline noise. Higher = clearer response.",
            "Clinical note": "Low SNR can result from artifacts, poor electrode contact, or absent cortical response.",
        },
        {
            "Term": "Spectral power change",
            "What it is": "Change in brain-wave frequency band power after stimulus vs. before stimulus.",
            "How to read it": "Negative change = power decrease (e.g., alpha suppression). Positive = power increase.",
            "Clinical note": "Alpha suppression after stimulus is a well-established indicator of cortical engagement.",
        },
        {
            "Term": "Trial consistency",
            "What it is": "How stable the stimulus-evoked response is across repeated stimulus presentations.",
            "How to read it": "0 = highly variable, 1 = perfectly consistent. Higher is more reliable.",
            "Clinical note": "Consistency across sessions (trend) is clinically more meaningful than one session's value.",
        },
        {
            "Term": "Evidence level",
            "What it is": "Summary label derived from SERI: Low (<40), Moderate (40-69), or High (>=70).",
            "How to read it": "A quick clinical shorthand for the strength of stimulus-evoked cortical modulation.",
            "Clinical note": "Always consider alongside Confidence and Quality — evidence level alone is insufficient.",
        },
    ])


# ═══════════════════════════════════════════════════════════════
# Dashboard rendering
# ═══════════════════════════════════════════════════════════════
def render_stimulus_response_tab(
    stimulus_results: pd.DataFrame,
    session_df: pd.DataFrame,
    selected_subject: int | None,
) -> None:
    """Render the Stimulus Response tab in the dashboard."""
    # Always-visible disclaimer
    st.error(_DISCLAIMER)

    # ── HOW TO INTERPRET ──────────────────────────────────────
    with st.expander("How to Interpret This Dashboard", expanded=True):
        st.markdown(
            "**This dashboard shows stimulus-evoked response indicators** — measures of whether "
            "the brain's electrical activity shows time-locked modulation in response to external stimuli.\n\n"
            "**Quick guide:**\n"
            "1. **Look at the three colored cards** (SERI, Confidence, Quality). "
            "Green = strong. Yellow = borderline. Red = weak or unreliable.\n"
            "2. **Read the 'What This Means' box** for a plain-language summary.\n"
            "3. **Check the trend** (if multiple sessions exist) — a consistent pattern is more informative than a single session.\n"
            "4. **Review Clinical Guidance** for recommended next steps.\n\n"
            "*All terms are explained in the Glossary at the bottom.*"
        )

    if selected_subject is None:
        st.info("Select a patient in the sidebar to view stimulus response outputs.")
        return

    subject_sessions = (
        session_df.loc[session_df["subject_id"] == selected_subject].copy()
        if not session_df.empty and "subject_id" in session_df.columns
        else pd.DataFrame()
    )
    subject_results = (
        stimulus_results.loc[stimulus_results["subject_id"] == selected_subject].copy()
        if not stimulus_results.empty and "subject_id" in stimulus_results.columns
        else pd.DataFrame()
    )

    if not subject_sessions.empty and "created_at" in subject_sessions.columns:
        subject_sessions = subject_sessions.sort_values("created_at", na_position="last").reset_index(drop=True)

    logger.info(
        "Stimulus tab: S%03d — %d sessions, %d result rows",
        selected_subject, len(subject_sessions), len(subject_results),
    )

    if subject_sessions.empty and subject_results.empty:
        st.info("No stimulus response outputs found for this patient. Run stimulus mode first.")
        return

    # ── Extract latest metrics ────────────────────────────────
    try:
        if not subject_sessions.empty:
            latest = subject_sessions.iloc[-1]
            seri = _to_float(latest.get("SERI", 0.0))
            confidence = _to_float(latest.get("Confidence", 0.0))
            quality = _to_float(latest.get("QualityScore", 0.0))
            evidence = str(latest.get("evidence", "Low"))
            strongest_markers = latest.get("strongest_markers", [])
            if strongest_markers is None:
                strongest_markers = []
            trial_consistency = _to_float(latest.get("trial_consistency", 0.0))
        else:
            latest = subject_results.iloc[-1]
            seri = _to_float(latest.get("SERI", 0.0))
            confidence = _to_float(latest.get("Confidence", 0.0))
            quality = _to_float(latest.get("QualityScore", 0.0))
            evidence = seri_evidence_label(seri)
            strongest_markers = []
            trial_consistency = 0.0
    except Exception as exc:
        logger.warning("Failed to extract metrics for S%03d: %s", selected_subject, exc)
        st.warning("Could not extract stimulus response metrics for this patient.")
        return

    # ── SECTION 1: Key Metrics Cards ──────────────────────────
    st.subheader("Stimulus-Evoked Response Indicator")
    st.caption("Most recent session summary for this patient.")

    c1, c2, c3 = st.columns(3)

    seri_color = "\U0001f534" if seri < 40 else "\U0001f7e1" if seri < 70 else "\U0001f7e2"
    c1.metric("SERI", f"{seri:.1f} {seri_color}")
    c1.caption(f"**{seri_evidence_label(seri)}** evidence level")

    conf_color = "\U0001f534" if confidence < 45 else "\U0001f7e1" if confidence < 70 else "\U0001f7e2"
    c2.metric("Confidence", f"{confidence:.1f} {conf_color}")
    c2.caption(f"**{seri_confidence_label(confidence)}** reliability")

    qual_color = "\U0001f534" if quality < 50 else "\U0001f7e1" if quality < 75 else "\U0001f7e2"
    c3.metric("QualityScore", f"{quality:.1f} {qual_color}")
    c3.caption(f"**{seri_quality_label(quality)}** data quality")

    # ── SECTION 2: Evidence Statement ─────────────────────────
    st.markdown("---")
    st.markdown("#### Evidence Summary")
    st.info(_evidence_statement(evidence))

    # ── SECTION 3: Plain-language interpretation ──────────────
    st.markdown("#### What This Means")
    st.success(_what_this_means(seri, confidence, quality))

    # ── SECTION 4: Clinical note ──────────────────────────────
    st.markdown("#### Clinical Guidance")
    st.warning(_clinical_note(seri=seri, confidence=confidence, quality=quality))

    # ── SECTION 5: Strongest markers ──────────────────────────
    st.markdown("---")
    st.markdown("#### Why This Result")
    st.caption(
        "Below are the strongest EEG markers that drove this session's scores. "
        "Each marker is explained in plain language."
    )

    if strongest_markers:
        for marker in strongest_markers[:5]:
            with st.container():
                st.markdown(f"- **{marker}**")
                st.caption(f"   \u2192 {_explain_marker(marker)}")
    else:
        st.caption("No marker summary available for this session.")

    # ── SECTION 6: SERI trend over time ───────────────────────
    st.markdown("---")
    st.markdown("#### SERI Trend Over Time")
    st.caption(
        "**How to read this chart:** Each point represents one session's SERI score. "
        "An upward trend suggests improving stimulus-evoked response. "
        "Day-to-day variation is normal — look at the overall trend direction. "
        "The green zone (>=70) indicates high-evidence sessions."
    )

    trend_df = subject_sessions.copy()
    if trend_df.empty and not subject_results.empty:
        if "session_id" in subject_results.columns and "SERI" in subject_results.columns:
            trend_df = (
                subject_results[["session_id", "SERI"]]
                .dropna(subset=["SERI"])
                .drop_duplicates(subset=["session_id"], keep="last")
                .reset_index(drop=True)
                .assign(created_at=lambda d: pd.RangeIndex(start=0, stop=len(d), step=1))
            )

    if not trend_df.empty and "SERI" in trend_df.columns:
        trend_df = trend_df.copy()
        trend_df["SERI"] = pd.to_numeric(trend_df["SERI"], errors="coerce")
        trend_df = trend_df.dropna(subset=["SERI"]).reset_index(drop=True)

    try:
        if not trend_df.empty and len(trend_df) > 1 and "SERI" in trend_df.columns:
            if "created_at" in trend_df.columns:
                trend_df = trend_df.sort_values("created_at")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["created_at"] if "created_at" in trend_df.columns else list(range(len(trend_df))),
                y=trend_df["SERI"],
                mode="lines+markers",
                name="SERI",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
            ))
            fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.08, line_width=0,
                         annotation_text="High evidence", annotation_position="top left")
            fig.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.06, line_width=0)
            fig.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.06, line_width=0)
            fig.update_layout(
                height=350,
                xaxis_title="Session",
                yaxis_title="SERI (0-100)",
                yaxis_range=[0, 105],
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, width="stretch", key=f"seri_trend_{selected_subject}")
        elif not trend_df.empty:
            st.caption("Only one session available. Trend chart appears after repeated sessions.")
        else:
            st.caption("No trend data available.")
    except Exception as exc:
        logger.warning("SERI trend chart failed for S%03d: %s", selected_subject, exc)
        st.caption("Unable to render SERI trend chart.")

    # ── SECTION 7: Confidence breakdown ───────────────────────
    st.markdown("---")
    st.markdown("#### Confidence Breakdown")
    st.caption(
        "**How to read this chart:** Confidence is built from six components. "
        "A balanced profile means confidence is well-supported. "
        "If one bar is much lower, that is the main limitation for this session."
    )

    _conf_components: dict[str, float] = {
        "Consistency\n(20%)": trial_consistency * 100,
        "Quality\n(15%)": quality,
        "Evidence\n(SERI)": seri,
    }

    if not subject_results.empty and "sharpness" in subject_results.columns:
        row = subject_results.iloc[-1]
        _conf_components = {
            "Sharpness\n(15%)": _to_float(row.get("sharpness", 0)),
            "Consistency\n(20%)": trial_consistency * 100,
            "Trial count\n(15%)": _to_float(row.get("usable_trial_score", 0)),
            "Quality\n(15%)": quality,
            "Reliability\n(20%)": _to_float(row.get("reliability_score", 0)),
            "CI tightness\n(15%)": _to_float(row.get("ci_tightness", 0)),
        }

    try:
        comp_df = pd.DataFrame(
            {"Component": list(_conf_components.keys()), "Score": list(_conf_components.values())}
        )
        comp_df["Score"] = comp_df["Score"].fillna(0).clip(0, 100)
        fig_conf = px.bar(
            comp_df,
            x="Component",
            y="Score",
            color="Score",
            color_continuous_scale=["#d9534f", "#f0ad4e", "#5cb85c"],
            range_color=[0, 100],
        )
        fig_conf.update_layout(
            height=300,
            yaxis_range=[0, 105],
            yaxis_title="Score (0-100)",
            showlegend=False,
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig_conf, width="stretch", key=f"conf_breakdown_{selected_subject}")
    except Exception as exc:
        logger.warning("Confidence breakdown chart failed for S%03d: %s", selected_subject, exc)
        st.caption("Unable to render confidence breakdown chart.")

    # ── SECTION 8: Full Glossary ──────────────────────────────
    st.markdown("---")
    with st.expander("Glossary — Every Term Explained for Clinicians", expanded=False):
        st.caption(
            "This glossary explains all technical terms used in this dashboard. "
            "No prior EEG or machine-learning expertise is required."
        )
        glossary_df = _full_glossary()
        for _, row in glossary_df.iterrows():
            st.markdown(f"**{row['Term']}**")
            st.markdown(f"- *What it is:* {row['What it is']}")
            st.markdown(f"- *How to read it:* {row['How to read it']}")
            st.markdown(f"- *Clinical note:* {row['Clinical note']}")
            st.markdown("")

    # ── SECTION 9: Trial-level table ──────────────────────────
    if not subject_results.empty:
        with st.expander("Session Trial-Level Data (Advanced)", expanded=False):
            st.caption(
                "This table shows each individual trial in the session. "
                "'description' is the stimulus marker. "
                "'response_probability' is the model's confidence that a stimulus response occurred."
            )
            try:
                keep_cols = [
                    c for c in [
                        "session_id", "epoch_idx", "description",
                        "n100_amplitude", "p300_amplitude", "evoked_snr",
                        "response_probability", "SERI", "Confidence", "QualityScore",
                    ]
                    if c in subject_results.columns
                ]
                if keep_cols:
                    st.dataframe(
                        subject_results[keep_cols].head(500),
                        width="stretch",
                        key=f"trial_table_{selected_subject}",
                    )
                else:
                    st.caption("No trial-level columns available.")
            except Exception as exc:
                logger.warning("Trial table render failed for S%03d: %s", selected_subject, exc)
                st.caption("Unable to render trial-level data.")

    # ── Bottom disclaimer ─────────────────────────────────────
    st.markdown("---")
    st.error(_DISCLAIMER)
