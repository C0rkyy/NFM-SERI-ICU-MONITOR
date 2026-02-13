from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import SESSION_STORE_DIR


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    subject_id: int | None = None
    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None


class StimulusProtocol(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    commands: list[str]
    cue_duration_s: float = Field(gt=0)
    imagery_duration_s: float = Field(gt=0)
    rest_duration_s: float = Field(gt=0)
    n_trials: int = Field(ge=2)
    randomized_order: bool
    inter_trial_interval: float = Field(ge=0)


class FeatureSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_trials: int = Field(ge=0)
    n_imagery_trials: int = Field(default=0, ge=0)
    n_response_trials: int = Field(default=0, ge=0)
    roi_channels_available: list[str] = Field(default_factory=list)
    strongest_markers: list[str] = Field(default_factory=list)
    trial_consistency: float = Field(ge=0, le=1)
    mu_erd_mean: float | None = None
    beta_erd_mean: float | None = None
    mean_evoked_snr: float | None = None
    mean_p300_amp: float | None = None


class ModelOutputs(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    calibration_method: str
    cv_within_subject_auc: float | None = None
    cv_across_subject_auc: float | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class Scores(BaseModel):
    model_config = ConfigDict(extra="allow")

    FRS: float | None = None
    CFI: float | None = None
    SERI: float | None = None
    Quality: float
    Confidence: float

    @field_validator("Quality", "Confidence")
    @classmethod
    def _validate_0_100(cls, value: float) -> float:
        if value < 0 or value > 100:
            raise ValueError("score must be within [0, 100]")
        return value

    @field_validator("CFI", "SERI", mode="before")
    @classmethod
    def _validate_optional_0_100(cls, value: float | None) -> float | None:
        if value is not None and (value < 0 or value > 100):
            raise ValueError("score must be within [0, 100]")
        return value


class RecordingSession(BaseModel):
    """Stored session for decision-support use only.

    Supports both command-following (CFI) and stimulus-evoked response (SERI) modes.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    patient: Patient
    protocol: StimulusProtocol
    feature_summary: FeatureSummary
    model_outputs: ModelOutputs
    scores: Scores
    mode: str = "active"  # "active" (CFI), "stimulus" (SERI), "passive" (FRS)
    disclaimer: str = (
        "Decision-support only. Not a diagnosis. This is a functional cortical "
        "responsiveness indicator to external stimuli."
    )


def save_recording_session(session: RecordingSession, output_dir: Path = SESSION_STORE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_id = session.patient.subject_id if session.patient.subject_id is not None else 0
    path = output_dir / f"session_s{subject_id:03d}_{session.session_id}.json"
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_recording_session(path: Path) -> RecordingSession:
    data: dict[str, Any] = __import__("json").loads(path.read_text(encoding="utf-8"))
    return RecordingSession.model_validate(data)
