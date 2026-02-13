"""
Dashboard — Clinical Validation tab.

Renders external validation metrics, calibration reliability,
drift status, and prospective workflow KPIs when outputs exist.

Decision-support only, not a diagnosis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import FIGURES_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

CLINICAL_DIR = OUTPUT_DIR / "clinical"


def _load_json(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
    return None


def clinical_outputs_exist() -> bool:
    """Check whether any clinical validation outputs are available."""
    return any([
        (CLINICAL_DIR / "external_validation_metrics.json").exists(),
        (CLINICAL_DIR / "calibration_report.json").exists(),
        (CLINICAL_DIR / "drift_report.json").exists(),
        (CLINICAL_DIR / "prospective_workflow_metrics.csv").exists(),
    ])


def render_clinical_validation_tab() -> None:
    """Render the clinical validation dashboard tab."""

    st.markdown(
        "⚠️ **Decision-support only. Not a diagnosis.** "
        "This is a stimulus-evoked response indicator measuring "
        "functional cortical responsiveness to external stimuli."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── External validation metrics ──────────────────────────
    with col1:
        st.subheader("External Validation")
        val_data = _load_json(CLINICAL_DIR / "external_validation_metrics.json")
        if val_data:
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("AUC", f"{val_data.get('auc', 0):.3f}")
            metrics_cols[1].metric("Sensitivity", f"{val_data.get('sensitivity', 0):.3f}")
            metrics_cols[2].metric("Specificity", f"{val_data.get('specificity', 0):.3f}")

            metrics_cols2 = st.columns(3)
            metrics_cols2[0].metric("PPV", f"{val_data.get('ppv', 0):.3f}")
            metrics_cols2[1].metric("NPV", f"{val_data.get('npv', 0):.3f}")
            metrics_cols2[2].metric("N (test)", f"{val_data.get('n_samples', 0)}")

            st.caption(
                "**How to read:** AUC measures discrimination (>0.7 = acceptable, >0.8 = good). "
                "Sensitivity = true positive rate. Specificity = true negative rate. "
                "PPV/NPV = predictive values. Higher is better for all metrics."
            )
        else:
            st.info(
                "No external validation results. Run the pipeline with "
                "`--clinical-validation retrospective --clinical-labels <path>` to generate."
            )

    # ── Calibration ──────────────────────────────────────────
    with col2:
        st.subheader("Calibration")
        cal_data = _load_json(CLINICAL_DIR / "calibration_report.json")
        if cal_data and "overall" in cal_data:
            overall = cal_data["overall"]
            cal_cols = st.columns(3)
            cal_cols[0].metric("ECE", f"{overall.get('ece', 0):.4f}")
            cal_cols[1].metric("Brier", f"{overall.get('brier', 0):.4f}")
            cal_cols[2].metric("MCE", f"{overall.get('mce', 0):.4f}")

            interp = cal_data.get("interpretation", "")
            if interp:
                st.info(interp)

            cal_fig_path = FIGURES_DIR / "calibration_reliability.png"
            if cal_fig_path.exists():
                st.image(str(cal_fig_path), caption="Reliability Diagram", use_container_width=True)

            st.caption(
                "**How to read:** ECE (Expected Calibration Error) measures how well "
                "predicted probabilities match actual outcomes. ECE < 0.05 is excellent. "
                "Brier score measures overall probability accuracy (lower is better)."
            )
        else:
            st.info("No calibration data. Requires clinical reference labels.")

    st.markdown("---")

    col3, col4 = st.columns(2)

    # ── Drift status ─────────────────────────────────────────
    with col3:
        st.subheader("Drift Monitoring")
        drift_data = _load_json(CLINICAL_DIR / "drift_report.json")
        if drift_data:
            alert = drift_data.get("alert_level", "green")
            if isinstance(alert, dict):
                alert = alert.get("value", "green") if isinstance(alert, dict) else str(alert)
            alert = str(alert).lower()

            if alert == "green":
                st.success("🟢 No significant drift detected")
            elif alert == "yellow":
                st.warning("🟡 Moderate drift detected — review recommended")
            else:
                st.error("🔴 Significant drift detected — recalibration recommended")

            details = drift_data.get("details", "")
            if details:
                st.caption(details)

            cal_shift = drift_data.get("calibration_shift", 0)
            st.metric("Calibration Drift", f"{cal_shift:.4f}")

            drift_fig_path = FIGURES_DIR / "drift_dashboard.png"
            if drift_fig_path.exists():
                st.image(str(drift_fig_path), caption="Drift Dashboard", use_container_width=True)

            st.caption(
                "**How to read:** Drift measures whether the model's input data or "
                "predictions have changed compared to the reference baseline. "
                "Green = stable, Yellow = monitor closely, Red = retrain/recalibrate."
            )
        else:
            st.info("No drift baseline. Run with `--drift-baseline <path>` to enable monitoring.")

    # ── Prospective workflow ─────────────────────────────────
    with col4:
        st.subheader("Prospective Workflow KPIs")
        prosp_path = CLINICAL_DIR / "prospective_workflow_metrics.csv"
        if prosp_path.exists():
            try:
                prosp_df = pd.read_csv(prosp_path)
                n_total = len(prosp_df)
                n_valid = int(prosp_df["valid"].sum()) if "valid" in prosp_df.columns else n_total
                n_invalid = n_total - n_valid
                invalid_rate = n_invalid / n_total if n_total > 0 else 0.0

                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Sessions", n_total)
                kpi_cols[1].metric("Valid", n_valid)
                kpi_cols[2].metric("Invalid Rate", f"{invalid_rate:.1%}")

                if "quality" in prosp_df.columns:
                    st.metric("Mean Quality", f"{prosp_df['quality'].mean():.1f}")

                if "score" in prosp_df.columns:
                    st.metric("Mean Score (SERI)", f"{prosp_df['score'].mean():.1f}")

                st.caption(
                    "**How to read:** These KPIs track operational readiness. "
                    "Invalid rate should be < 10%. Mean quality > 60 indicates "
                    "reliable data. Mode 'silent' means scores are logged but not used "
                    "for clinical decisions."
                )
            except Exception as exc:
                st.warning(f"Could not load prospective data: {exc}")
        else:
            st.info("No prospective workflow data. Run with `--clinical-validation prospective`.")

    # ── Clinical report link ─────────────────────────────────
    st.markdown("---")
    report_md = CLINICAL_DIR / "clinical_evidence_report.md"
    if report_md.exists():
        with st.expander("📄 Full Clinical Evidence Report"):
            st.markdown(report_md.read_text(encoding="utf-8"))
    else:
        st.caption("Full clinical report will appear here after running with `--generate-clinical-report`.")
