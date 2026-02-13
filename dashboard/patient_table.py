"""
NFM - Patient Table Panel
===========================
Full-cohort data table with sorting, filtering, clinical category
column, and one-click CSV export.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.interpretation_engine import clinical_category

logger = logging.getLogger(__name__)


def _build_patient_table(
    frs_df: pd.DataFrame,
    subject_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the presentation-ready patient table.

    If a pre-aggregated subject_summary is supplied we use it directly.
    Otherwise we aggregate from the epoch-level frs_df.
    """
    if subject_summary is not None and len(subject_summary) > 0:
        tbl = subject_summary.copy()
    else:
        # Build aggregation dict only for columns that actually exist
        agg_spec: dict[str, tuple[str, str]] = {}
        agg_spec["FRS"] = ("frs", "mean")
        for col, name in [
            ("norm_model_probability", "Model_Probability"),
            ("p300_amplitude", "ERP_Amplitude"),
            ("p300_latency_ms", "P300_Latency"),
            ("gci", "Coherence_Index"),
        ]:
            if col in frs_df.columns:
                agg_spec[name] = (col, "mean")
        if "alpha_rel_change" in frs_df.columns:
            agg_spec["Alpha_Power_Shift"] = ("alpha_rel_change", "mean")

        tbl = frs_df.groupby("subject_id").agg(**agg_spec).reset_index()

    col_map = {
        "subject_id": "Subject ID",
        "frs_mean": "FRS",
        "frs": "FRS",
        "model_prob_mean": "Model Probability",
        "norm_model_probability": "Model Probability",
        "Model_Probability": "Model Probability",
        "erp_amplitude_mean": "ERP Amplitude",
        "p300_amplitude": "ERP Amplitude",
        "ERP_Amplitude": "ERP Amplitude",
        "p300_latency_ms": "P300 Latency (ms)",
        "P300_Latency": "P300 Latency (ms)",
        "psd_shift_mean": "Alpha Power Shift",
        "alpha_rel_change": "Alpha Power Shift",
        "Alpha_Power_Shift": "Alpha Power Shift",
        "gci": "Coherence Index",
        "Coherence_Index": "Coherence Index",
    }
    tbl = tbl.rename(columns={k: v for k, v in col_map.items() if k in tbl.columns})

    if "FRS" not in tbl.columns:
        for col in tbl.columns:
            if "frs" in col.lower():
                tbl = tbl.rename(columns={col: "FRS"})
                break

    if "FRS" in tbl.columns:
        tbl["FRS"] = pd.to_numeric(tbl["FRS"], errors="coerce").fillna(0)
        tbl["Clinical Category"] = tbl["FRS"].apply(
            lambda v: clinical_category(float(v))["label"]
        )

    # Coerce all numeric columns to avoid mixed-type issues
    for col in tbl.columns:
        if col in ("Subject ID", "Clinical Category"):
            continue
        tbl[col] = pd.to_numeric(tbl[col], errors="coerce")

    return tbl


def render_patient_table(
    frs_df: pd.DataFrame,
    subject_summary: Optional[pd.DataFrame] = None,
) -> None:
    """Render the interactive patient table with filters and export."""
    try:
        tbl = _build_patient_table(frs_df, subject_summary)
    except Exception as exc:
        logger.warning("Failed to build patient table: %s", exc)
        st.warning("Unable to build patient table.")
        return

    if tbl.empty:
        st.info("No patient data available.")
        return

    st.markdown("##### Filter Patients")
    fc1, fc2 = st.columns(2)

    with fc1:
        if "Clinical Category" in tbl.columns:
            categories = ["All"] + sorted(
                tbl["Clinical Category"].dropna().unique().tolist()
            )
            sel_cat = st.selectbox("Category", categories, key="pt_cat_filter")
        else:
            sel_cat = "All"

    with fc2:
        if "FRS" in tbl.columns:
            frs_range = st.slider(
                "FRS Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                key="pt_frs_filter",
            )
        else:
            frs_range = (0, 100)

    filtered = tbl.copy()
    if sel_cat != "All" and "Clinical Category" in filtered.columns:
        filtered = filtered.loc[filtered["Clinical Category"] == sel_cat].copy()
    if "FRS" in filtered.columns:
        filtered = filtered.loc[
            (filtered["FRS"] >= frs_range[0]) & (filtered["FRS"] <= frs_range[1])
        ].copy()

    st.markdown(f"**{len(filtered)}** patient(s) displayed")

    # Round numeric columns for clean display (avoids Styler entirely)
    display = filtered.copy()
    for col in display.columns:
        if col in ("Subject ID", "Clinical Category"):
            continue
        if pd.api.types.is_float_dtype(display[col]):
            if "latency" in col.lower() or "ms" in col.lower():
                display[col] = display[col].round(1)
            elif "probability" in col.lower():
                display[col] = (display[col] * 100).round(1).astype(str) + "%"
            elif col == "FRS":
                display[col] = display[col].round(0).astype(int)
            else:
                display[col] = display[col].round(3)

    # Replace remaining NaN with dash for display
    display = display.fillna("-")

    try:
        st.dataframe(
            display,
            height=min(400, 56 + 35 * len(display)),
            width="stretch",
        )
    except Exception as exc:
        logger.warning("Dataframe rendering failed: %s", exc)
        st.text(display.to_string(index=False))

    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button(
        label="Export Filtered Table (CSV)",
        data=csv_buf.getvalue(),
        file_name="nfm_patient_table.csv",
        mime="text/csv",
        key="pt_export_btn",
    )
