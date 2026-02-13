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
from scoring.cfi import cfi_evidence_label, confidence_label, quality_label

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Clinical disclaimer (always visible)
# ═══════════════════════════════════════════════════════════════
_DISCLAIMER = (
    "**Decision-support only. Not a diagnosis.** "
    "This command-following indicator reflects functional responsiveness "
    "to mental commands. It does not measure consciousness and must be "
    "interpreted within the full clinical context by a qualified clinician."
)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
def load_active_sessions(session_store_dir: Path = SESSION_STORE_DIR) -> pd.DataFrame:
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
        has_cfi = scores.get("CFI") is not None

        # Developer note:
        # we hard-separate legacy CFI sessions from stimulus sessions so tabs do
        # not cross-contaminate each other.
        if mode == "stimulus":
            continue
        if not has_cfi:
            # Allow legacy payloads with empty mode as long as CFI exists.
            continue
        feature_summary = payload.get("feature_summary", {})
        patient = payload.get("patient", {})

        strongest = feature_summary.get("strongest_markers", [])
        if isinstance(strongest, str):
            strongest = [m.strip() for m in strongest.split("|") if m.strip()]

        cfi = _to_float(scores.get("CFI", 0.0))
        confidence = _to_float(scores.get("Confidence", 0.0))
        quality = _to_float(scores.get("Quality", scores.get("QualityScore", 0.0)))

        subject_raw = patient.get("subject_id", 0)
        try:
            subject_id = int(subject_raw)
        except (TypeError, ValueError):
            subject_id = 0
        if subject_id <= 0:
            continue

        rows.append(
            {
                "subject_id": subject_id,
                "session_id": payload.get("session_id", path.stem),
                "created_at": payload.get("created_at"),
                "CFI": cfi,
                "Confidence": confidence,
                "QualityScore": quality,
                "evidence": cfi_evidence_label(cfi),
                "confidence_level": confidence_label(confidence),
                "quality_level": quality_label(quality),
                "trial_consistency": _to_float(feature_summary.get("trial_consistency", 0.0)),
                "strongest_markers": strongest,
                "disclaimer": payload.get(
                    "disclaimer",
                    "Decision-support only. Not a diagnosis.",
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "session_id",
                "created_at",
                "CFI",
                "Confidence",
                "QualityScore",
                "evidence",
                "confidence_level",
                "quality_level",
                "trial_consistency",
                "strongest_markers",
                "disclaimer",
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
            "**High evidence** of command-locked cortical response. "
            "The EEG pattern during imagery blocks is clearly distinct from rest periods."
        )
    if evidence == "Moderate":
        return (
            "**Moderate evidence** of command-locked cortical response. "
            "Some EEG features shift during imagery blocks compared to rest, but the pattern is not fully consistent."
        )
    return (
        "**Low evidence** of command-locked cortical response. "
        "The EEG pattern during imagery blocks is not reliably different from rest periods in this session."
    )


def _clinical_note(cfi: float, confidence: float, quality: float) -> str:
    if quality < 50:
        return (
            "**Data quality is limited.** Repeat acquisition after improving electrode contacts "
            "and reducing movement artifacts before making any clinical interpretation."
        )
    if cfi >= 70 and confidence >= 70:
        return (
            "**Pattern is consistent with functional responsiveness to mental commands.** "
            "Use as a supportive decision-signal alongside the full clinical picture. "
            "Serial assessments strengthen interpretation."
        )
    if cfi < 40:
        return (
            "**Current session does not show strong command-following signal.** "
            "Consider repeating the assessment and evaluating sedation level, time of day, "
            "sensory environment, and electrode quality. A single low result does not rule out "
            "responsiveness."
        )
    return (
        "**Signal is intermediate.** Repeat in serial sessions and track the trend over time "
        "before incorporating into clinical decisions. Day-to-day variability is expected."
    )


def _what_this_means(cfi: float, confidence: float, quality: float) -> str:
    """Plain-language explanation of what the current result means for the clinician."""
    parts: list[str] = []

    if cfi >= 70:
        parts.append(
            "The brain showed a clear, repeatable change in electrical patterns when the patient "
            "was asked to imagine movement, compared to resting periods."
        )
    elif cfi >= 40:
        parts.append(
            "The brain showed some change in electrical patterns during imagery tasks, but the "
            "signal was not strong enough to be fully definitive in this session."
        )
    else:
        parts.append(
            "In this session, the brain's electrical patterns during imagery tasks were not clearly "
            "different from resting periods."
        )

    if confidence >= 70:
        parts.append(
            "We have high confidence in this measurement: data quality was adequate, the model is "
            "well-calibrated, and trial responses were consistent."
        )
    elif confidence >= 45:
        parts.append(
            "Confidence in this measurement is moderate. Some factors (data quality, trial count, "
            "or response variability) limit certainty."
        )
    else:
        parts.append(
            "Confidence is low — interpret with extra caution. The data may have quality issues, "
            "too few usable trials, or inconsistent responses."
        )

    return " ".join(parts)


def _explain_marker(marker: str) -> str:
    lower = marker.lower()
    if "mu" in lower and "cz" in lower:
        return (
            "The mu rhythm (8-12 Hz brain wave) changed at the Cz electrode (top of head). "
            "This suggests the motor cortex responded to the imagery command."
        )
    if "mu" in lower:
        return (
            "The mu rhythm (8-12 Hz) showed a change during imagery vs. rest. "
            "Mu suppression is a well-established marker of motor imagery engagement."
        )
    if "beta" in lower and ("c3" in lower or "c4" in lower):
        return (
            "Beta waves (13-30 Hz) decreased over C3/C4 (hand motor areas). "
            "This pattern supports task-locked motor cortex activation."
        )
    if "beta" in lower:
        return (
            "Beta band (13-30 Hz) power changed during imagery. "
            "This contributes evidence of cortical activation above resting levels."
        )
    if "consistency" in lower:
        return (
            "Trial consistency measures how stable the command-following response was "
            "across repeated trials. Higher consistency strengthens the finding."
        )
    return "This feature contributes to evidence of functional responsiveness to mental commands."


# ═══════════════════════════════════════════════════════════════
# Full glossary of all terms (expanded)
# ═══════════════════════════════════════════════════════════════
def _full_glossary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Term": "CFI (Command-Following Index)",
                "What it is": "Overall score (0-100) summarizing how strongly the EEG responds to imagery commands vs. rest.",
                "How to read it": "Low (<40): weak response. Moderate (40-69): some response. High (>=70): clear response.",
                "Clinical note": "A single-session CFI should not be used alone. Track trend over serial assessments.",
            },
            {
                "Term": "Confidence (0-100)",
                "What it is": "How reliable this particular session's CFI is. Combines data quality, model calibration, trial consistency, sample size, and statistical uncertainty.",
                "How to read it": "Low (<45): interpret with caution. Moderate (45-69): reasonable certainty. High (>=70): well-supported result.",
                "Clinical note": "Low confidence means repeating the session under better conditions could give a different result.",
            },
            {
                "Term": "QualityScore (0-100)",
                "What it is": "Technical quality of the EEG recording for this session.",
                "How to read it": "Driven by: how many trials survived artifact rejection (epoch drop rate), bad electrode channels, residual power-line noise, and signal-to-noise ratio.",
                "Clinical note": "Quality < 50 means the recording had major technical issues — repeat before interpreting.",
            },
            {
                "Term": "Trial consistency",
                "What it is": "How stable the imagery response is across repeated trials within the session.",
                "How to read it": "0 = highly variable, 1 = perfectly consistent. Higher is better.",
                "Clinical note": "Consistency across sessions (trend) is clinically more meaningful than one session's value.",
            },
            {
                "Term": "Imagery probability",
                "What it is": "The model's estimated probability that a given trial is an imagery trial (vs. rest).",
                "How to read it": "0.5 = chance level. Values well above 0.5 suggest the brain pattern matches imagery. Values near or below 0.5 suggest it matches rest.",
                "Clinical note": "Calibrated so that predicted probability closely tracks actual frequency of correct classification.",
            },
            {
                "Term": "ERD/ERS (Event-Related Desynchronization / Synchronization)",
                "What it is": "A decrease (ERD) or increase (ERS) in brain-wave power during imagery compared to rest baseline.",
                "How to read it": "Negative % = ERD (power decrease during imagery). Positive % = ERS (power increase). Both can be informative.",
                "Clinical note": "ERD in mu/beta bands over motor cortex is the classic hallmark of motor imagery.",
            },
            {
                "Term": "Mu band (8-12 Hz)",
                "What it is": "Brain-wave frequency range associated with the sensorimotor cortex at rest.",
                "How to read it": "Mu suppression (ERD) during imagery suggests the motor system is engaged.",
                "Clinical note": "Most relevant at electrodes C3, C4, and Cz (motor cortex areas).",
            },
            {
                "Term": "Beta band (13-30 Hz)",
                "What it is": "Higher-frequency brain-wave range also linked to motor cortex activity.",
                "How to read it": "Beta ERD during imagery provides complementary evidence to mu band changes.",
                "Clinical note": "Less specific than mu but adds confidence when both show consistent changes.",
            },
            {
                "Term": "ROI (Region of Interest)",
                "What it is": "Electrodes C3, C4, and Cz — the channels overlying hand and midline motor cortex.",
                "How to read it": "Features computed from ROI channels are most relevant for motor imagery detection.",
                "Clinical note": "If ROI channels have poor contact, the entire analysis may be unreliable.",
            },
            {
                "Term": "CSP (Common Spatial Patterns)",
                "What it is": "A spatial filter that finds the combination of electrodes that best separates imagery from rest.",
                "How to read it": "Used internally by the model. CSP features improve discriminability.",
                "Clinical note": "Requires multiple channels and enough trials per class to be effective.",
            },
            {
                "Term": "Calibration",
                "What it is": "Process of adjusting model outputs so that predicted probabilities are accurate.",
                "How to read it": "A well-calibrated model saying '80% probability' is correct about 80% of the time.",
                "Clinical note": "Isotonic calibration is used when enough data is available; sigmoid otherwise.",
            },
            {
                "Term": "Evidence level",
                "What it is": "Summary label derived from CFI: Low (<40), Moderate (40-69), or High (>=70).",
                "How to read it": "A quick clinical shorthand for the strength of the command-following signal.",
                "Clinical note": "Always consider alongside Confidence and Quality — evidence level alone is insufficient.",
            },
        ]
    )


# ═══════════════════════════════════════════════════════════════
# Dashboard rendering
# ═══════════════════════════════════════════════════════════════
def render_command_following_tab(
    command_results: pd.DataFrame,
    session_df: pd.DataFrame,
    selected_subject: int | None,
) -> None:
    # Always-visible disclaimer
    st.error(_DISCLAIMER)

    # ── HOW TO INTERPRET (always visible) ─────────────────────
    with st.expander("📋 How to Interpret This Dashboard", expanded=True):
        st.markdown(
            "**This dashboard shows command-following indicators** — measures of whether "
            "the brain's electrical activity changes in response to mental imagery commands.\n\n"
            "**Quick guide:**\n"
            "1. **Look at the three colored cards** (CFI, Confidence, Quality). "
            "Green (🟢) = strong. Yellow (🟡) = borderline. Red (🔴) = weak or unreliable.\n"
            "2. **Read the 'What This Means' box** for a plain-language summary of what the result means for this patient.\n"
            "3. **Check the trend** (if multiple sessions exist) — a consistent pattern is more informative than a single session.\n"
            "4. **Review Clinical Guidance** for recommended next steps.\n\n"
            "*All terms are explained in the Glossary at the bottom.*"
        )

    if selected_subject is None:
        st.info("Select a patient in the sidebar to view command-following outputs.")
        return

    subject_sessions = (
        session_df.loc[session_df["subject_id"] == selected_subject].copy()
        if not session_df.empty and "subject_id" in session_df.columns
        else pd.DataFrame()
    )
    subject_results = (
        command_results.loc[command_results["subject_id"] == selected_subject].copy()
        if not command_results.empty and "subject_id" in command_results.columns
        else pd.DataFrame()
    )
    if not subject_sessions.empty and "created_at" in subject_sessions.columns:
        subject_sessions = subject_sessions.sort_values("created_at", na_position="last").reset_index(drop=True)
    if not subject_results.empty and "session_id" in subject_results.columns:
        subject_results = subject_results.sort_values("session_id").reset_index(drop=True)

    logger.info(
        "Command tab: S%03d — %d sessions, %d result rows",
        selected_subject, len(subject_sessions), len(subject_results),
    )

    if subject_sessions.empty and subject_results.empty:
        st.info("No command-following outputs found for this patient. Run active mode first.")
        return

    # Extract latest session metrics with defensive parsing.
    # Goal: never crash the whole tab because one malformed field is null/string.
    try:
        if not subject_sessions.empty:
            latest = subject_sessions.iloc[-1]
            cfi = _to_float(latest.get("CFI", 0.0), default=0.0)
            confidence = _to_float(latest.get("Confidence", 0.0), default=0.0)
            quality = _to_float(latest.get("QualityScore", 0.0), default=0.0)
            evidence = str(latest.get("evidence", "Low"))
            strongest_markers = latest.get("strongest_markers", [])
            if strongest_markers is None:
                strongest_markers = []
            trial_consistency = _to_float(latest.get("trial_consistency", 0.0), default=0.0)
        else:
            latest = subject_results.iloc[-1]
            cfi = _to_float(latest.get("CFI", 0.0), default=0.0)
            confidence = _to_float(latest.get("Confidence", 0.0), default=0.0)
            quality = _to_float(latest.get("QualityScore", 0.0), default=0.0)
            evidence = cfi_evidence_label(cfi)
            strongest_markers = []
            trial_consistency = (
                float(subject_results["consistency"].mean())
                if "consistency" in subject_results.columns and not subject_results["consistency"].isna().all()
                else 0.0
            )
    except Exception as exc:
        logger.warning("Failed to extract metrics for S%03d: %s", selected_subject, exc)
        st.warning("Could not extract command-following metrics for this patient.")
        return

    # ── SECTION 1: Key Metrics Cards ──────────────────────────
    st.subheader("Command-Following Indicator")
    st.caption("Most recent session summary for this patient.")

    c1, c2, c3 = st.columns(3)

    # CFI card with color coding
    cfi_color = "🔴" if cfi < 40 else "🟡" if cfi < 70 else "🟢"
    c1.metric("CFI", f"{cfi:.1f} {cfi_color}")
    c1.caption(f"**{cfi_evidence_label(cfi)}** evidence level")

    conf_color = "🔴" if confidence < 45 else "🟡" if confidence < 70 else "🟢"
    c2.metric("Confidence", f"{confidence:.1f} {conf_color}")
    c2.caption(f"**{confidence_label(confidence)}** reliability")

    qual_color = "🔴" if quality < 50 else "🟡" if quality < 75 else "🟢"
    c3.metric("QualityScore", f"{quality:.1f} {qual_color}")
    c3.caption(f"**{quality_label(quality)}** data quality")

    # ── SECTION 2: Evidence Statement ─────────────────────────
    st.markdown("---")
    st.markdown("#### Evidence Summary")
    st.info(_evidence_statement(evidence))

    # ── SECTION 3: Plain-language interpretation ──────────────
    st.markdown("#### What This Means")
    st.success(_what_this_means(cfi, confidence, quality))

    # ── SECTION 4: Clinical note ──────────────────────────────
    st.markdown("#### Clinical Guidance")
    st.warning(_clinical_note(cfi=cfi, confidence=confidence, quality=quality))

    # ── SECTION 5: Why this result ────────────────────────────
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
                st.caption(f"   → {_explain_marker(marker)}")
    elif not subject_results.empty:
        candidates: list[str] = []
        for col, label in [
            ("roi_mu_erd_ers_pct", "ROI mu ERD/ERS"),
            ("roi_beta_erd_ers_pct", "ROI beta ERD/ERS"),
            ("consistency", "Trial consistency"),
        ]:
            if col in subject_results.columns:
                col_mean = subject_results[col].mean()
                if pd.notna(col_mean):
                    candidates.append(f"{label}: {float(col_mean):+.2f}")
        if candidates:
            for marker in candidates[:5]:
                st.markdown(f"- **{marker}**")
                st.caption(f"   → {_explain_marker(marker)}")
        else:
            st.caption("No marker summary available for this session.")

    # ── SECTION 6: CFI trend over time ────────────────────────
    st.markdown("---")
    st.markdown("#### CFI Trend Over Time")
    st.caption(
        "**How to read this chart:** Each point represents one session's CFI score. "
        "An upward trend over time suggests improving command-following response. "
        "Day-to-day variation is normal — look at the overall trend direction, not individual points. "
        "The green zone (>=70) indicates high-evidence sessions."
    )

    trend_df = subject_sessions.copy()
    if trend_df.empty and not subject_results.empty:
        if "session_id" in subject_results.columns and "CFI" in subject_results.columns:
            trend_df = (
                subject_results[["session_id", "CFI"]]
                .dropna(subset=["CFI"])
                .drop_duplicates(subset=["session_id"], keep="last")
                .reset_index(drop=True)
                .assign(created_at=lambda d: pd.RangeIndex(start=0, stop=len(d), step=1))
            )
        else:
            trend_df = pd.DataFrame()
    if not trend_df.empty:
        trend_df = trend_df.copy()
        trend_df["CFI"] = pd.to_numeric(trend_df["CFI"], errors="coerce")
        trend_df = trend_df.dropna(subset=["CFI"]).reset_index(drop=True)

    try:
        if not trend_df.empty and len(trend_df) > 1:
            if "created_at" in trend_df.columns:
                trend_df = trend_df.sort_values("created_at")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["created_at"] if "created_at" in trend_df.columns else list(range(len(trend_df))),
                y=trend_df["CFI"],
                mode="lines+markers",
                name="CFI",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
            ))
            fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.08, line_width=0, annotation_text="High evidence", annotation_position="top left")
            fig.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.06, line_width=0)
            fig.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.06, line_width=0)
            fig.update_layout(
                height=350,
                xaxis_title="Session",
                yaxis_title="CFI (0-100)",
                yaxis_range=[0, 105],
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, width="stretch", key=f"cfi_trend_{selected_subject}")
        elif not trend_df.empty:
            st.caption("Only one session available. Trend chart appears after repeated sessions.")
        else:
            st.caption("No trend data available.")
    except Exception as exc:
        logger.warning("CFI trend chart failed for S%03d: %s", selected_subject, exc)
        st.caption("Unable to render CFI trend chart.")

    # ── SECTION 7: Confidence breakdown ───────────────────────
    st.markdown("---")
    st.markdown("#### Confidence Breakdown")
    st.caption(
        "**How to read this chart:** Confidence is built from six components. "
        "A balanced profile (all bars similar height) means confidence is well-supported. "
        "If one bar is much lower, that is the main limitation for this session. "
        "Each bar shows 0–100. The overall Confidence is a weighted average of these."
    )

    # Prefer persisted confidence components when available; otherwise use a
    # reduced fallback so clinicians still see what limits confidence.
    _conf_components: dict[str, float] = {}
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
    else:
        _conf_components = {
            "Consistency\n(20%)": trial_consistency * 100,
            "Quality\n(15%)": quality,
            "Evidence\n(CFI)": cfi,
        }

    try:
        comp_df = pd.DataFrame(
            {"Component": list(_conf_components.keys()), "Score": list(_conf_components.values())}
        )
        # Replace any remaining NaN with 0 for safe chart rendering
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
    with st.expander("📖 Glossary — Every Term Explained for Clinicians", expanded=False):
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

    # ── SECTION 9: Trial-level table (advanced) ───────────────
    if not subject_results.empty:
        with st.expander("📊 Session Trial-Level Data (Advanced)", expanded=False):
            st.caption(
                "This table shows each individual trial in the session. "
                "'command' is what the patient was asked to do. "
                "'imagery_probability' is the model's confidence that the trial was imagery (vs. rest). "
                "Values closer to 1.0 for imagery trials and 0.0 for rest trials indicate a strong signal."
            )
            try:
                keep_cols = [
                    c
                    for c in [
                        "session_id",
                        "epoch_idx",
                        "command_label",
                        "is_imagery",
                        "imagery_probability",
                        "CFI",
                        "Confidence",
                        "QualityScore",
                        "Evidence",
                        "ConfidenceLevel",
                        "QualityLevel",
                    ]
                    if c in subject_results.columns
                ]
                if keep_cols:
                    table = subject_results[keep_cols].head(500).copy()
                    rename_map = {
                        "command_label": "command",
                        "is_imagery": "imagery_flag",
                        "epoch_idx": "trial_number",
                    }
                    st.dataframe(
                        table.rename(columns=rename_map),
                        width="stretch",
                        key=f"trial_table_{selected_subject}",
                    )
                else:
                    st.caption("No trial-level columns available.")
            except Exception as exc:
                logger.warning("Trial table render failed for S%03d: %s", selected_subject, exc)
                st.caption("Unable to render trial-level data.")

    # ── Bottom disclaimer (always visible) ────────────────────
    st.markdown("---")
    st.error(_DISCLAIMER)
