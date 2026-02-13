"""Tests for clinical validation Pydantic schemas."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.clinical_validation import (
    AlertLevel,
    ClinicalAssessment,
    DriftSnapshot,
    ReferenceStandardLabel,
    RunManifest,
    SplitType,
    ValidationMetrics,
    ValidationSplitSpec,
)


class TestClinicalAssessment:
    def test_valid_assessment(self):
        a = ClinicalAssessment(
            patient_id="P001",
            subject_id=1,
            center_id="center_A",
            session_id="s001",
            assessment_time=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            crsr_total=12,
            crsr_subscales={"auditory": 3, "visual": 4},
            assessor_id="dr_smith",
            repeated_exam_index=1,
        )
        assert a.crsr_total == 12
        assert a.center_id == "center_A"

    def test_crsr_total_out_of_range(self):
        with pytest.raises(ValidationError):
            ClinicalAssessment(
                patient_id="P001",
                subject_id=1,
                center_id="center_A",
                session_id="s001",
                assessment_time=datetime.now(timezone.utc),
                crsr_total=30,  # max is 23
                assessor_id="dr_smith",
            )

    def test_negative_subject_id(self):
        with pytest.raises(ValidationError):
            ClinicalAssessment(
                patient_id="P001",
                subject_id=-1,
                center_id="center_A",
                session_id="s001",
                assessment_time=datetime.now(timezone.utc),
                crsr_total=10,
                assessor_id="dr_smith",
            )

    def test_empty_center_id(self):
        with pytest.raises(ValidationError):
            ClinicalAssessment(
                patient_id="P001",
                subject_id=1,
                center_id="",
                session_id="s001",
                assessment_time=datetime.now(timezone.utc),
                crsr_total=10,
                assessor_id="dr_smith",
            )


class TestReferenceStandardLabel:
    def test_valid_label(self):
        label = ReferenceStandardLabel(
            session_id="s001",
            label_target=1,
            source="CRS-R",
            confidence=0.95,
        )
        assert label.label_target == 1
        assert label.source == "CRS-R"

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            ReferenceStandardLabel(
                session_id="s001",
                label_target=1,
                source="invalid_source",
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            ReferenceStandardLabel(
                session_id="s001",
                label_target=0,
                confidence=1.5,
            )


class TestValidationSplitSpec:
    def test_valid_split(self):
        spec = ValidationSplitSpec(
            split_type=SplitType.CENTER,
            train_ids=["s001", "s002"],
            test_ids=["s003", "s004"],
            description="Center split test",
        )
        assert spec.split_type == SplitType.CENTER
        assert len(spec.train_ids) == 2

    def test_overlap_raises(self):
        with pytest.raises(ValidationError, match="overlap"):
            ValidationSplitSpec(
                split_type=SplitType.CENTER,
                train_ids=["s001", "s002"],
                test_ids=["s002", "s003"],
            )

    def test_no_overlap_passes(self):
        spec = ValidationSplitSpec(
            split_type=SplitType.TIME,
            train_ids=["s001"],
            test_ids=["s002"],
        )
        assert len(spec.test_ids) == 1


class TestValidationMetrics:
    def test_valid_metrics(self):
        m = ValidationMetrics(
            auc=0.85,
            pr_auc=0.78,
            sensitivity=0.80,
            specificity=0.90,
            ppv=0.75,
            npv=0.92,
            brier=0.12,
            ece=0.04,
            calibration_slope=1.05,
            calibration_intercept=-0.02,
            n_samples=100,
        )
        assert m.auc == 0.85

    def test_auc_out_of_range(self):
        with pytest.raises(ValidationError):
            ValidationMetrics(auc=1.5)


class TestDriftSnapshot:
    def test_valid_snapshot(self):
        snap = DriftSnapshot(
            feature_shift_metrics={"alpha_power": 0.05},
            prediction_shift_metrics={"psi": 0.08},
            calibration_shift=0.02,
            alert_level=AlertLevel.GREEN,
            details="All clear",
        )
        assert snap.alert_level == AlertLevel.GREEN

    def test_red_alert(self):
        snap = DriftSnapshot(alert_level=AlertLevel.RED)
        assert snap.alert_level == AlertLevel.RED


class TestRunManifest:
    def test_valid_manifest(self):
        m = RunManifest(
            seed=42,
            python_version="3.12.0",
            model_version="v1.0",
            data_paths=["path/to/data.csv"],
            args={"mode": "stimulus"},
            config_hash="abc123",
        )
        assert m.seed == 42
        assert "decision-support" in m.disclaimer.lower()
