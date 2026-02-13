"""
Clinical validation schemas — Pydantic data contracts for external validation,
calibration monitoring, drift detection, and prospective workflow.

All outputs are decision-support only, not a diagnosis. These schemas enforce
strict data contracts for reproducible clinical proof-of-concept studies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ═══════════════════════════════════════════════════════════════
# Clinical assessment (reference standard)
# ═══════════════════════════════════════════════════════════════

class ClinicalAssessment(BaseModel):
    """A single bedside clinical assessment (e.g. CRS-R exam)."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    subject_id: int = Field(ge=0)
    center_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    assessment_time: datetime
    crsr_total: int = Field(ge=0, le=23)
    crsr_subscales: dict[str, int] = Field(default_factory=dict)
    assessor_id: str = Field(min_length=1)
    repeated_exam_index: int = Field(ge=1, default=1)


class ReferenceStandardLabel(BaseModel):
    """Binary or ordinal label derived from clinical assessments."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    label_target: int = Field(ge=0)
    source: str = Field(default="CRS-R")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @field_validator("source")
    @classmethod
    def _allowed_sources(cls, v: str) -> str:
        allowed = {"CRS-R", "multimodal", "expert_consensus", "behavioral"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}")
        return v


# ═══════════════════════════════════════════════════════════════
# Validation split specification
# ═══════════════════════════════════════════════════════════════

class SplitType(str, Enum):
    CENTER = "center"
    TIME = "time"
    CENTER_TIME = "center_time"


class ValidationSplitSpec(BaseModel):
    """Specification for external validation splits with leakage checks."""

    model_config = ConfigDict(extra="forbid")

    split_type: SplitType
    train_ids: list[str]
    test_ids: list[str]
    description: str = ""

    @field_validator("test_ids")
    @classmethod
    def _no_overlap(cls, v: list[str], info: Any) -> list[str]:
        train = info.data.get("train_ids", [])
        overlap = set(v) & set(train)
        if overlap:
            raise ValueError(f"train/test overlap detected: {overlap}")
        return v


# ═══════════════════════════════════════════════════════════════
# Validation metrics
# ═══════════════════════════════════════════════════════════════

class ValidationMetrics(BaseModel):
    """Performance + calibration metrics from a validation fold."""

    model_config = ConfigDict(extra="allow")

    auc: float = Field(ge=0.0, le=1.0)
    pr_auc: float = Field(ge=0.0, le=1.0, default=0.0)
    sensitivity: float = Field(ge=0.0, le=1.0, default=0.0)
    specificity: float = Field(ge=0.0, le=1.0, default=0.0)
    ppv: float = Field(ge=0.0, le=1.0, default=0.0)
    npv: float = Field(ge=0.0, le=1.0, default=0.0)
    brier: float = Field(ge=0.0, le=1.0, default=0.0)
    ece: float = Field(ge=0.0, default=0.0)
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0
    n_samples: int = Field(ge=0, default=0)
    split_type: str = ""


# ═══════════════════════════════════════════════════════════════
# Drift monitoring
# ═══════════════════════════════════════════════════════════════

class AlertLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class DriftSnapshot(BaseModel):
    """Point-in-time drift measurement."""

    model_config = ConfigDict(extra="allow")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_shift_metrics: dict[str, float] = Field(default_factory=dict)
    prediction_shift_metrics: dict[str, float] = Field(default_factory=dict)
    calibration_shift: float = 0.0
    alert_level: AlertLevel = AlertLevel.GREEN
    details: str = ""


# ═══════════════════════════════════════════════════════════════
# Run manifest (reproducibility)
# ═══════════════════════════════════════════════════════════════

class RunManifest(BaseModel):
    """Deterministic run manifest for full reproducibility."""

    model_config = ConfigDict(extra="allow")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed: int
    python_version: str = ""
    git_hash: str | None = None
    model_version: str = ""
    data_paths: list[str] = Field(default_factory=list)
    args: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    disclaimer: str = (
        "Decision-support only, not a diagnosis. "
        "Stimulus-evoked response indicator for functional cortical responsiveness."
    )
