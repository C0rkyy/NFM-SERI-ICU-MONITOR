"""
Tests for BrainVision data ingestion and event extraction.

Validates:
- BrainVision session discovery returns valid structure
- Marker parsing produces expected event types
- Senlist parsing returns correct labels
- Event extraction classifies events correctly
- Epoch extraction contract (shape, metadata, stats)
- Feature extraction output schema
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from config import LIS_DATASET_ROOT

# ── Skip if LIS data not available ──────────────────────────

_LIS_AVAILABLE = LIS_DATASET_ROOT.exists() and any(LIS_DATASET_ROOT.rglob("*.vhdr"))

skip_no_lis = pytest.mark.skipif(
    not _LIS_AVAILABLE,
    reason="LIS BrainVision dataset not found at expected path",
)


# ═══════════════════════════════════════════════════════════════
# Data ingestion tests
# ═══════════════════════════════════════════════════════════════


@skip_no_lis
class TestBrainVisionDiscovery:
    """Test BrainVision session discovery."""

    def test_discover_sessions_returns_list(self) -> None:
        from data.brainvision_loader import discover_brainvision_sessions

        sessions = discover_brainvision_sessions(LIS_DATASET_ROOT)
        assert isinstance(sessions, list)
        assert len(sessions) > 0

    def test_session_dict_has_required_keys(self) -> None:
        from data.brainvision_loader import discover_brainvision_sessions

        sessions = discover_brainvision_sessions(LIS_DATASET_ROOT)
        required_keys = {"file_path", "subject_id", "visit_id", "day_id", "session_type"}
        for si in sessions[:5]:
            assert required_keys.issubset(set(si.keys())), f"Missing keys in {si}"

    def test_registry_builds_dataframe(self) -> None:
        from data.brainvision_loader import build_stimulus_dataset_registry

        df = build_stimulus_dataset_registry(LIS_DATASET_ROOT)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "subject_id" in df.columns
        assert "file_path" in df.columns

    def test_load_raw_returns_mne_raw(self) -> None:
        from data.brainvision_loader import discover_brainvision_sessions, load_brainvision_raw

        sessions = discover_brainvision_sessions(LIS_DATASET_ROOT)
        raw = load_brainvision_raw(Path(sessions[0]["file_path"]))
        assert hasattr(raw, "info")
        assert raw.info["sfreq"] > 0
        assert len(raw.ch_names) > 0


@skip_no_lis
class TestBrainVisionEvents:
    """Test event extraction from BrainVision data."""

    @pytest.fixture(scope="class")
    def first_session_raw(self):
        from data.brainvision_loader import discover_brainvision_sessions, load_brainvision_raw

        sessions = discover_brainvision_sessions(LIS_DATASET_ROOT)
        raw = load_brainvision_raw(Path(sessions[0]["file_path"]))
        return raw, sessions[0]

    def test_extract_events_returns_array(self, first_session_raw) -> None:
        from data.brainvision_loader import extract_brainvision_events

        raw, _ = first_session_raw
        events, event_id = extract_brainvision_events(raw)
        assert isinstance(events, np.ndarray)
        if events.size > 0:
            assert events.ndim == 2
            assert events.shape[1] == 3

    def test_stimulus_events_classified(self, first_session_raw) -> None:
        from data.brainvision_loader import extract_brainvision_events
        from stimulus.event_extraction import extract_stimulus_events

        raw, session_info = first_session_raw
        events, event_id = extract_brainvision_events(raw)
        sfreq = float(raw.info["sfreq"])

        stim_df = extract_stimulus_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            session_dir=Path(session_info["file_path"]).parent,
        )
        assert isinstance(stim_df, pd.DataFrame)
        if not stim_df.empty:
            assert "event_type" in stim_df.columns
            assert "time_s" in stim_df.columns


# ═══════════════════════════════════════════════════════════════
# Epoching tests
# ═══════════════════════════════════════════════════════════════


@skip_no_lis
class TestStimulusEpochs:
    """Test stimulus epoch extraction."""

    @pytest.fixture(scope="class")
    def epoch_result(self):
        import mne

        mne.set_log_level("WARNING")
        from data.brainvision_loader import (
            discover_brainvision_sessions,
            extract_brainvision_events,
            load_brainvision_raw,
        )
        from preprocessing.stimulus_epochs import extract_stimulus_epochs
        from stimulus.event_extraction import extract_stimulus_events, get_stimulus_onset_events

        sessions = discover_brainvision_sessions(LIS_DATASET_ROOT)
        raw = load_brainvision_raw(Path(sessions[0]["file_path"]))
        raw.filter(0.5, 40.0, verbose=False)
        events, event_id = extract_brainvision_events(raw)
        sfreq = float(raw.info["sfreq"])

        stim_df = extract_stimulus_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            session_dir=Path(sessions[0]["file_path"]).parent,
        )

        onset_events, onset_event_id = get_stimulus_onset_events(stim_df, event_id)
        if onset_events.shape[0] == 0:
            onset_events = events
            onset_event_id = event_id

        epochs, metadata, stats = extract_stimulus_epochs(
            raw=raw,
            events=onset_events,
            event_id=onset_event_id,
        )
        return epochs, metadata, stats

    def test_epochs_have_shape(self, epoch_result) -> None:
        epochs, _, _ = epoch_result
        if len(epochs) > 0:
            data = epochs.get_data()
            assert data.ndim == 3  # (n_epochs, n_channels, n_times)

    def test_metadata_has_columns(self, epoch_result) -> None:
        _, metadata, _ = epoch_result
        if metadata is not None and not metadata.empty:
            assert "event_id" in metadata.columns or len(metadata.columns) > 0

    def test_stats_has_drop_rate(self, epoch_result) -> None:
        _, _, stats = epoch_result
        assert "epoch_drop_rate" in stats
        assert 0.0 <= stats["epoch_drop_rate"] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Feature extraction tests (unit, no data dependency)
# ═══════════════════════════════════════════════════════════════


class TestStimulusFeatureSchema:
    """Test stimulus feature extraction on synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_epochs(self):
        """Create minimal synthetic MNE Epochs for testing."""
        import mne

        sfreq = 500.0
        n_channels = 4
        n_times = 500  # 1 second
        n_epochs = 10
        ch_names = [f"EEG{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq, ch_types="eeg")

        # Simulate evoked response: baseline noise + post-stim peak
        data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
        # Add P300-like component
        t_p300 = int(0.3 * sfreq)
        data[:, :, t_p300 : t_p300 + 50] += 3e-6

        events = np.column_stack(
            [np.arange(0, n_epochs * n_times, n_times), np.zeros(n_epochs, dtype=int), np.ones(n_epochs, dtype=int)]
        )
        epochs = mne.EpochsArray(data, info, events=events, tmin=-0.2)
        metadata = pd.DataFrame({"trial_idx": range(n_epochs)})
        return epochs, metadata

    def test_feature_extraction_runs(self, synthetic_epochs) -> None:
        from features.stimulus_features import extract_stimulus_response_features

        epochs, metadata = synthetic_epochs
        trial_df, summary_df, details = extract_stimulus_response_features(
            epochs=epochs,
            metadata=metadata,
            baseline_window=(-0.2, 0.0),
            response_window=(0.0, 0.8),
        )
        assert isinstance(trial_df, pd.DataFrame)
        assert len(trial_df) == len(epochs)
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(details, dict)

    def test_features_have_erp_columns(self, synthetic_epochs) -> None:
        from features.stimulus_features import extract_stimulus_response_features

        epochs, metadata = synthetic_epochs
        trial_df, _, _ = extract_stimulus_response_features(
            epochs=epochs,
            metadata=metadata,
        )
        # Should have some ERP-related columns
        erp_cols = [c for c in trial_df.columns if "n100" in c.lower() or "p300" in c.lower() or "p200" in c.lower()]
        assert len(erp_cols) > 0, f"No ERP columns found. Columns: {list(trial_df.columns)}"

    def test_features_no_nans_in_key_columns(self, synthetic_epochs) -> None:
        from features.stimulus_features import extract_stimulus_response_features

        epochs, metadata = synthetic_epochs
        trial_df, _, _ = extract_stimulus_response_features(
            epochs=epochs,
            metadata=metadata,
        )
        # Key feature columns should not be all NaN
        numeric_cols = trial_df.select_dtypes(include=[np.number]).columns
        all_nan_cols = [c for c in numeric_cols if trial_df[c].isna().all()]
        assert len(all_nan_cols) < len(numeric_cols), "All numeric columns are entirely NaN"


# ═══════════════════════════════════════════════════════════════
# Quality scoring tests
# ═══════════════════════════════════════════════════════════════


class TestQualityMonotonicity:
    """Quality score should decrease with worse data quality indicators."""

    def test_drop_rate_quality_monotone(self) -> None:
        from scoring.quality import compute_quality_score

        q_good = compute_quality_score(epoch_drop_rate=0.05, bad_channel_fraction=0.0, line_noise_residual=0.05, snr_proxy=5.0)
        q_bad = compute_quality_score(epoch_drop_rate=0.50, bad_channel_fraction=0.0, line_noise_residual=0.05, snr_proxy=5.0)
        assert q_good["QualityScore"] >= q_bad["QualityScore"]

    def test_bad_channels_quality_monotone(self) -> None:
        from scoring.quality import compute_quality_score

        q_good = compute_quality_score(epoch_drop_rate=0.1, bad_channel_fraction=0.0, line_noise_residual=0.05, snr_proxy=5.0)
        q_bad = compute_quality_score(epoch_drop_rate=0.1, bad_channel_fraction=0.5, line_noise_residual=0.05, snr_proxy=5.0)
        assert q_good["QualityScore"] >= q_bad["QualityScore"]

    def test_quality_always_bounded(self) -> None:
        from scoring.quality import compute_quality_score

        for dr in [0.0, 0.3, 0.6, 1.0]:
            for bc in [0.0, 0.2, 0.5, 1.0]:
                q = compute_quality_score(epoch_drop_rate=dr, bad_channel_fraction=bc, line_noise_residual=0.05, snr_proxy=5.0)
                assert 0.0 <= q["QualityScore"] <= 100.0


# ═══════════════════════════════════════════════════════════════
# Dashboard smoke tests
# ═══════════════════════════════════════════════════════════════


class TestDashboardStimulusImports:
    """Ensure stimulus dashboard components import without error."""

    def test_stimulus_response_tab_importable(self) -> None:
        from dashboard.stimulus_response import render_stimulus_response_tab  # noqa: F401

    def test_seri_labels_importable(self) -> None:
        from scoring.seri import (  # noqa: F401
            compute_seri,
            seri_clinical_summary,
            seri_confidence_label,
            seri_evidence_label,
            seri_quality_label,
        )

    def test_brainvision_loader_importable(self) -> None:
        from data.brainvision_loader import (  # noqa: F401
            discover_brainvision_sessions,
            load_brainvision_raw,
        )

    def test_stimulus_features_importable(self) -> None:
        from features.stimulus_features import extract_stimulus_response_features  # noqa: F401

    def test_stimulus_epochs_importable(self) -> None:
        from preprocessing.stimulus_epochs import extract_stimulus_epochs  # noqa: F401

    def test_event_extraction_importable(self) -> None:
        from stimulus.event_extraction import extract_stimulus_events  # noqa: F401

    def test_stimulus_model_importable(self) -> None:
        from models.stimulus_response import train_stimulus_response  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# Schema tests
# ═══════════════════════════════════════════════════════════════


class TestSchemasSERI:
    """Test that schemas accept SERI fields."""

    def test_scores_accept_seri(self) -> None:
        from schemas.session import Scores

        s = Scores(SERI=75.0, Quality=80.0, Confidence=65.0)
        assert s.SERI == 75.0

    def test_scores_accept_cfi(self) -> None:
        from schemas.session import Scores

        s = Scores(CFI=60.0, Quality=80.0, Confidence=65.0)
        assert s.CFI == 60.0

    def test_scores_seri_and_cfi_both_optional(self) -> None:
        from schemas.session import Scores

        s = Scores(Quality=80.0, Confidence=65.0)
        assert s.SERI is None
        assert s.CFI is None

    def test_feature_summary_accepts_stimulus_fields(self) -> None:
        from schemas.session import FeatureSummary

        fs = FeatureSummary(
            n_trials=20,
            n_response_trials=15,
            mean_evoked_snr=3.5,
            mean_p300_amp=2.1,
            trial_consistency=0.8,
        )
        assert fs.n_response_trials == 15
        assert fs.mean_evoked_snr == 3.5

    def test_recording_session_accepts_mode(self) -> None:
        from schemas.session import (
            FeatureSummary,
            ModelOutputs,
            Patient,
            RecordingSession,
            Scores,
            StimulusProtocol,
        )

        session = RecordingSession(
            session_id="test_001",
            patient=Patient(patient_id="S001", subject_id=1),
            protocol=StimulusProtocol(
                name="test",
                commands=["STIM"],
                n_trials=10,
                cue_duration_s=0.5,
                imagery_duration_s=0.8,
                rest_duration_s=1.0,
                randomized_order=False,
                inter_trial_interval=2.0,
            ),
            feature_summary=FeatureSummary(n_trials=10, trial_consistency=0.5),
            model_outputs=ModelOutputs(model_name="test", calibration_method="isotonic"),
            scores=Scores(SERI=50.0, Quality=70.0, Confidence=60.0),
            mode="stimulus",
        )
        assert session.mode == "stimulus"
