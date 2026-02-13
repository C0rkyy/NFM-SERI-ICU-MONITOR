from __future__ import annotations

import logging
import pickle
import re
from typing import Any

import pandas as pd
import streamlit as st

from config import OUTPUT_DIR, RESULTS_DIR, SESSION_STORE_DIR
from dashboard.clinical_summary import render_clinical_summary
from dashboard.clinical_validation import clinical_outputs_exist, render_clinical_validation_tab
from dashboard.command_following import load_active_sessions, render_command_following_tab
from dashboard.export_module import render_export_panel
from dashboard.patient_table import render_patient_table
from dashboard.stimulus_response import load_stimulus_sessions, render_stimulus_response_tab
from dashboard.trend_analysis import render_trend_analysis

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@st.cache_data
def load_dashboard_data() -> dict[str, object]:
    """Load all dashboard data sources. Cached globally (immutable raw data)."""
    data: dict[str, object] = {}

    feat_path = RESULTS_DIR / "features_all_subjects.csv"
    if feat_path.exists():
        try:
            data["features"] = pd.read_csv(feat_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load features_all_subjects.csv: %s", exc)

    frs_path = RESULTS_DIR / "frs_results.csv"
    if frs_path.exists():
        try:
            data["frs"] = pd.read_csv(frs_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load frs_results.csv: %s", exc)

    summary_path = RESULTS_DIR / "subject_summary.csv"
    if summary_path.exists():
        try:
            data["subject_summary"] = pd.read_csv(summary_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load subject_summary.csv: %s", exc)

    cache_path = RESULTS_DIR / "dashboard_cache.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                data["cache"] = pickle.load(f)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load dashboard_cache.pkl: %s", exc)

    command_csv = OUTPUT_DIR / "results_command_following.csv"
    if command_csv.exists():
        try:
            data["command_results"] = pd.read_csv(command_csv)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load results_command_following.csv: %s", exc)

    stimulus_csv = OUTPUT_DIR / "results_stimulus_response.csv"
    if stimulus_csv.exists():
        try:
            data["stimulus_results"] = pd.read_csv(stimulus_csv)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load results_stimulus_response.csv: %s", exc)

    try:
        data["command_sessions"] = load_active_sessions(SESSION_STORE_DIR)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load command sessions: %s", exc)
        data["command_sessions"] = pd.DataFrame()

    try:
        data["stimulus_sessions"] = load_stimulus_sessions(SESSION_STORE_DIR)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load stimulus sessions: %s", exc)
        data["stimulus_sessions"] = pd.DataFrame()
    return data


def _to_subject_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    match = re.search(r"(\d{1,3})", text)
    if match:
        return int(match.group(1))
    return None


def _format_subject(value: object) -> str:
    sid = _to_subject_int(value)
    if sid is None:
        return str(value)
    return f"Patient S{sid:03d}"


def _build_subject_pool(data: dict[str, object]) -> list[int]:
    # Developer note:
    # if LIS stimulus data exists, we intentionally scope to that cohort first.
    # This avoids mixing legacy subjects into the primary ICU workflow.
    stimulus_pool: set[int] = set()
    for key in ("stimulus_results", "stimulus_sessions"):
        df = data.get(key)
        if isinstance(df, pd.DataFrame) and "subject_id" in df.columns:
            subject_series = pd.to_numeric(df["subject_id"], errors="coerce").dropna().astype(int)
            stimulus_pool.update(v for v in subject_series.unique() if v > 0)
    if stimulus_pool:
        # Primary clinical mode: LIS stimulus-response monitoring only.
        return sorted(stimulus_pool)

    pool: set[int] = set()
    for key in ("frs", "command_results", "command_sessions", "stimulus_results", "stimulus_sessions"):
        df = data.get(key)
        if isinstance(df, pd.DataFrame) and "subject_id" in df.columns:
            subject_series = pd.to_numeric(df["subject_id"], errors="coerce").dropna().astype(int)
            pool.update(v for v in subject_series.unique() if v > 0)
    return sorted(pool)


def _df_copy(data: dict[str, object], key: str) -> pd.DataFrame:
    """Return a deep copy of *data[key]* if it is a DataFrame, else empty.

    NOTE: ``load_dashboard_data()`` is cached via ``st.cache_data``.
    Developer note:
    always deep-copy before rendering. In-place mutations inside a tab can
    poison cache state and produce confusing rerun behavior.
    """
    value = data.get(key)
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="Neuro Functional Monitor",
        page_icon="NFM",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data = load_dashboard_data()
    subjects = _build_subject_pool(data)
    stimulus_results = _df_copy(data, "stimulus_results")
    stimulus_sessions = _df_copy(data, "stimulus_sessions")
    stimulus_subjects_present = (
        ("subject_id" in stimulus_results.columns and not stimulus_results.empty)
        or ("subject_id" in stimulus_sessions.columns and not stimulus_sessions.empty)
    )

    st.sidebar.title("Neuro Functional Monitor")
    st.sidebar.caption("Clinical decision-support")
    if stimulus_subjects_present:
        # Make the active data source explicit to clinicians/operators.
        st.sidebar.success("Data source: LIS stimulus sessions only")

    if not subjects:
        st.title("Neuro Functional Monitor")
        st.info("No processed data found yet. Run `python main.py` first.")
        return

    default_subject = subjects[0]
    selected_raw = st.sidebar.selectbox(
        "Select Patient",
        subjects,
        format_func=_format_subject,
        key="subject_selector",
    )
    selected_subject = _to_subject_int(selected_raw)
    if selected_subject is None or selected_subject not in subjects:
        logger.warning("Invalid subject_selector state=%r. Falling back to %s", selected_raw, default_subject)
        selected_subject = default_subject
        st.session_state["subject_selector"] = default_subject

    st.sidebar.markdown("---")
    st.sidebar.warning("Decision-support only. Not a diagnosis.")

    if st.sidebar.button("Refresh Data", key="refresh_data_btn"):
        st.cache_data.clear()
        st.rerun()

    st.title("Neuro Functional Monitor")
    st.caption("Primary mode: stimulus-evoked cortical response monitoring for ICU decision-support")

    logger.info("Subject switched to S%03d", selected_subject)

    if stimulus_subjects_present:
        has_clinical = clinical_outputs_exist()
        if has_clinical:
            stim_tab, clin_tab = st.tabs(["Stimulus Response", "Clinical Validation"])
        else:
            stim_tab = st.container()
            clin_tab = None

        with stim_tab:
            try:
                render_stimulus_response_tab(stimulus_results, stimulus_sessions, selected_subject)
            except Exception as exc:  # pragma: no cover
                logger.exception("Error rendering stimulus panel for S%03d", selected_subject)
                st.error(
                    f"An error occurred while rendering stimulus response data for Patient S{selected_subject:03d}. "
                    f"Please try again or contact the system administrator.\n\nDetails: {exc}"
                )

        if clin_tab is not None:
            with clin_tab:
                try:
                    render_clinical_validation_tab()
                except Exception as exc:  # pragma: no cover
                    logger.exception("Error rendering clinical validation tab")
                    st.error(f"Error loading clinical validation data: {exc}")
        return

    command_results = _df_copy(data, "command_results")
    command_sessions = _df_copy(data, "command_sessions")
    has_command_following = not command_results.empty or not command_sessions.empty

    if has_command_following:
        passive_tab, stimulus_tab, command_tab = st.tabs(
            ["Passive FRS", "Stimulus Response", "Command Following (Legacy)"]
        )
    else:
        passive_tab, stimulus_tab = st.tabs(["Passive FRS", "Stimulus Response"])
        command_tab = None

    with passive_tab:
        try:
            frs_df = _df_copy(data, "frs")
            if not frs_df.empty:
                sdf = frs_df.loc[frs_df["subject_id"] == selected_subject].copy()
                if sdf.empty:
                    st.info("No passive FRS data for this patient.")
                else:
                    logger.info("Passive tab: S%03d — %d rows", selected_subject, len(sdf))
                    render_clinical_summary(sdf, selected_subject)
                    st.markdown("---")
                    render_trend_analysis(sdf, selected_subject)
                    st.markdown("---")
                    render_patient_table(frs_df, _df_copy(data, "subject_summary"))
                    st.markdown("---")
                    render_export_panel(frs_df, _df_copy(data, "features"))
            else:
                st.info("Passive FRS outputs not found.")
        except Exception as exc:  # pragma: no cover
            logger.exception("Error rendering passive tab for S%03d", selected_subject)
            st.error(
                f"An error occurred while rendering passive data for Patient S{selected_subject:03d}. "
                f"Please try again or contact the system administrator.\n\nDetails: {exc}"
            )

    with stimulus_tab:
        try:
            render_stimulus_response_tab(stimulus_results, stimulus_sessions, selected_subject)
        except Exception as exc:  # pragma: no cover
            logger.exception("Error rendering stimulus tab for S%03d", selected_subject)
            st.error(
                f"An error occurred while rendering stimulus response data for Patient S{selected_subject:03d}. "
                f"Please try again or contact the system administrator.\n\nDetails: {exc}"
            )

    if command_tab is not None:
        with command_tab:
            try:
                render_command_following_tab(command_results, command_sessions, selected_subject)
            except Exception as exc:  # pragma: no cover
                logger.exception("Error rendering command tab for S%03d", selected_subject)
                st.error(
                    f"An error occurred while rendering command-following data for Patient S{selected_subject:03d}. "
                    f"Please try again or contact the system administrator.\n\nDetails: {exc}"
                )


if __name__ == "__main__":
    main()
