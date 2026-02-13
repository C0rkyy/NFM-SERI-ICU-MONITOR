from __future__ import annotations

import pytest

from schemas.session import (
    FeatureSummary,
    ModelOutputs,
    Patient,
    RecordingSession,
    Scores,
    StimulusProtocol,
)


def test_recording_session_schema_validation() -> None:
    session = RecordingSession(
        session_id="active_s001",
        patient=Patient(patient_id="S001", subject_id=1),
        protocol=StimulusProtocol(
            name="motor_imagery_basic",
            commands=["REST", "IMAGINE_WALKING", "IMAGINE_HAND"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
            rest_duration_s=3.0,
            n_trials=20,
            randomized_order=True,
            inter_trial_interval=1.0,
        ),
        feature_summary=FeatureSummary(
            n_trials=20,
            n_imagery_trials=20,
            roi_channels_available=["C3", "C4", "CZ"],
            strongest_markers=["mu ERD at Cz"],
            trial_consistency=0.7,
            mu_erd_mean=-12.0,
            beta_erd_mean=-8.0,
        ),
        model_outputs=ModelOutputs(
            model_name="logistic_regression_calibrated",
            calibration_method="sigmoid",
            cv_within_subject_auc=0.72,
            cv_across_subject_auc=0.69,
            artifact_paths={"model": "outputs/models/command_following_model.pkl"},
        ),
        scores=Scores(FRS=None, CFI=63.2, Quality=78.4, Confidence=71.0),
    )

    assert session.scores.CFI == pytest.approx(63.2)


def test_scores_reject_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        Scores(FRS=None, CFI=120.0, Quality=80.0, Confidence=70.0)
