"""
Smoke tests for dashboard subject switching.

Validates that:
- Data loading returns valid structures
- Subject filtering produces valid (possibly empty) DataFrames
- Chart-input builders produce safe structures for all subjects
- No unhandled exceptions during metric extraction
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import OUTPUT_DIR, RESULTS_DIR, SESSION_STORE_DIR
from dashboard.command_following import load_active_sessions
from dashboard.interpretation_engine import clinical_category, trend_label
from dashboard.utils import _to_float
from scoring.cfi import cfi_evidence_label, confidence_label, quality_label


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frs_df() -> pd.DataFrame:
    path = RESULTS_DIR / "frs_results.csv"
    if not path.exists():
        pytest.skip("frs_results.csv not found — run main.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def command_results() -> pd.DataFrame:
    path = OUTPUT_DIR / "results_command_following.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def command_sessions() -> pd.DataFrame:
    return load_active_sessions(SESSION_STORE_DIR)


@pytest.fixture(scope="module")
def subject_pool(frs_df, command_results, command_sessions) -> list[int]:
    pool: set[int] = set()
    for df in (frs_df, command_results, command_sessions):
        if isinstance(df, pd.DataFrame) and "subject_id" in df.columns:
            pool.update(int(v) for v in df["subject_id"].dropna().unique())
    return sorted(pool)


# ── Tests ─────────────────────────────────────────────────────

SAMPLE_SUBJECTS = [1, 2, 5, 10, 20, 50, 90, 100, 109]


class TestDataLoading:
    def test_frs_has_required_columns(self, frs_df):
        assert "subject_id" in frs_df.columns
        assert "frs" in frs_df.columns

    def test_command_sessions_schema(self, command_sessions):
        if command_sessions.empty:
            pytest.skip("No command sessions found")
        required = {"subject_id", "CFI", "Confidence", "QualityScore"}
        assert required.issubset(set(command_sessions.columns))

    def test_command_results_schema(self, command_results):
        if command_results.empty:
            pytest.skip("No command results found")
        assert "subject_id" in command_results.columns


class TestSubjectFiltering:
    """Ensure filtering by subject never crashes."""

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_frs_filter_safe(self, frs_df, sid):
        sdf = frs_df.loc[frs_df["subject_id"] == sid].copy()
        assert isinstance(sdf, pd.DataFrame)  # may be empty, that's fine

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_command_sessions_filter_safe(self, command_sessions, sid):
        if command_sessions.empty:
            pytest.skip("No command sessions")
        sdf = command_sessions.loc[command_sessions["subject_id"] == sid].copy()
        assert isinstance(sdf, pd.DataFrame)

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_command_results_filter_safe(self, command_results, sid):
        if command_results.empty:
            pytest.skip("No command results")
        sdf = command_results.loc[command_results["subject_id"] == sid].copy()
        assert isinstance(sdf, pd.DataFrame)


class TestMetricExtraction:
    """Ensure metric extraction logic never raises."""

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_extract_metrics_no_crash(self, command_sessions, command_results, sid):
        subject_sessions = (
            command_sessions.loc[command_sessions["subject_id"] == sid].copy()
            if not command_sessions.empty and "subject_id" in command_sessions.columns
            else pd.DataFrame()
        )
        subject_results = (
            command_results.loc[command_results["subject_id"] == sid].copy()
            if not command_results.empty and "subject_id" in command_results.columns
            else pd.DataFrame()
        )

        if subject_sessions.empty and subject_results.empty:
            return  # no data — dashboard shows info message

        if not subject_sessions.empty:
            latest = subject_sessions.iloc[-1]
            cfi = _to_float(latest.get("CFI", 0.0))
            confidence = _to_float(latest.get("Confidence", 0.0))
            quality = _to_float(latest.get("QualityScore", 0.0))
        else:
            latest = subject_results.iloc[-1]
            cfi = _to_float(latest.get("CFI", 0.0))
            confidence = _to_float(latest.get("Confidence", 0.0))
            quality = _to_float(latest.get("QualityScore", 0.0))

        # These must not raise
        assert 0 <= cfi <= 200  # generous range
        assert 0 <= confidence <= 200
        assert 0 <= quality <= 200
        _ = cfi_evidence_label(cfi)
        _ = confidence_label(confidence)
        _ = quality_label(quality)


class TestClinicalHelpers:
    @pytest.mark.parametrize("frs", [0, 10, 25, 50, 75, 100])
    def test_clinical_category(self, frs):
        cat = clinical_category(frs)
        assert "label" in cat
        assert "color_hex" in cat

    @pytest.mark.parametrize("pct", [-50, -5, 0, 5, 50])
    def test_trend_label(self, pct):
        tl = trend_label(pct)
        assert "label" in tl
        assert "arrow" in tl

    @pytest.mark.parametrize("cfi", [0, 30, 60, 90])
    def test_cfi_evidence_label(self, cfi):
        label = cfi_evidence_label(cfi)
        assert isinstance(label, str)


class TestChartInputBuilders:
    """Ensure chart data structures are valid before passing to Plotly."""

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_trend_df_no_nan_crash(self, command_sessions, command_results, sid):
        subject_sessions = (
            command_sessions.loc[command_sessions["subject_id"] == sid].copy()
            if not command_sessions.empty and "subject_id" in command_sessions.columns
            else pd.DataFrame()
        )
        subject_results = (
            command_results.loc[command_results["subject_id"] == sid].copy()
            if not command_results.empty and "subject_id" in command_results.columns
            else pd.DataFrame()
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

        # Verify: if trend_df has data, CFI column should have no NaN
        if not trend_df.empty and "CFI" in trend_df.columns:
            assert trend_df["CFI"].isna().sum() == 0, "CFI column should not have NaN after dropna"

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_confidence_components_finite(self, command_results, sid):
        """Confidence breakdown components must all be finite numbers."""
        if command_results.empty:
            pytest.skip("No command results")
        subject_results = command_results.loc[command_results["subject_id"] == sid].copy()
        if subject_results.empty:
            return

        components: dict[str, float] = {}
        if "sharpness" in subject_results.columns:
            row = subject_results.iloc[-1]
            for col in ("sharpness", "usable_trial_score", "reliability_score", "ci_tightness"):
                components[col] = _to_float(row.get(col, 0))
        else:
            components["cfi"] = _to_float(subject_results.iloc[-1].get("CFI", 0))

        for k, v in components.items():
            assert pd.notna(v), f"{k} is NaN for subject {sid}"


class TestPatientTable:
    """Ensure patient table builds safely for all data."""

    def test_build_patient_table_no_crash(self, frs_df):
        from dashboard.patient_table import _build_patient_table
        tbl = _build_patient_table(frs_df)
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) > 0

    def test_patient_table_no_nan_in_frs(self, frs_df):
        from dashboard.patient_table import _build_patient_table
        tbl = _build_patient_table(frs_df)
        if "FRS" in tbl.columns:
            assert tbl["FRS"].isna().sum() == 0, "FRS should have no NaN after fillna(0)"

    def test_patient_table_clinical_category(self, frs_df):
        from dashboard.patient_table import _build_patient_table
        tbl = _build_patient_table(frs_df)
        if "Clinical Category" in tbl.columns:
            assert tbl["Clinical Category"].isna().sum() == 0

    def test_patient_table_missing_columns_safe(self):
        """Table builder must not crash when optional columns are absent."""
        from dashboard.patient_table import _build_patient_table
        minimal = pd.DataFrame({
            "subject_id": [1, 1, 2, 2],
            "frs": [50.0, 60.0, 70.0, 80.0],
        })
        tbl = _build_patient_table(minimal)
        assert "FRS" in tbl.columns
        assert len(tbl) == 2  # 2 subjects

    def test_patient_table_numeric_coercion(self, frs_df):
        """All non-text columns should be numeric dtype."""
        from dashboard.patient_table import _build_patient_table
        tbl = _build_patient_table(frs_df)
        for col in tbl.columns:
            if col in ("Subject ID", "Clinical Category"):
                continue
            assert pd.api.types.is_numeric_dtype(tbl[col]), (
                f"Column {col} should be numeric but got {tbl[col].dtype}"
            )


class TestExportModule:
    """Ensure export builders don't crash."""

    def test_build_full_export(self, frs_df):
        from dashboard.export_module import build_full_export
        export = build_full_export(frs_df)
        assert isinstance(export, pd.DataFrame)
        assert len(export) > 0

    def test_build_subject_json(self, frs_df):
        from dashboard.export_module import build_subject_json
        result = build_subject_json(frs_df)
        assert isinstance(result, dict)
        assert len(result) > 0
        for sid, entry in result.items():
            assert "frs_mean" in entry
            assert "clinical_category" in entry

    def test_full_export_csv_serializable(self, frs_df):
        """Full export must be serializable to CSV without errors."""
        import io
        from dashboard.export_module import build_full_export
        export = build_full_export(frs_df)
        buf = io.StringIO()
        export.to_csv(buf, index=False)  # should not raise
        assert len(buf.getvalue()) > 0


class TestTrendAnalysisSafety:
    """Ensure trend analysis produces safe values."""

    @pytest.mark.parametrize("sid", SAMPLE_SUBJECTS)
    def test_trend_no_inf(self, frs_df, sid):
        import numpy as np
        from dashboard.trend_analysis import _build_subject_trend
        sdf = frs_df.loc[frs_df["subject_id"] == sid].copy()
        if sdf.empty:
            return
        trend = _build_subject_trend(sdf, sid)
        for col in ("mean_frs", "pct_change"):
            if col in trend.columns:
                assert not trend[col].apply(lambda v: np.isinf(v) if pd.notna(v) else False).any(), (
                    f"{col} has inf values for subject {sid}"
                )
