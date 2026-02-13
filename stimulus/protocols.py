from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import (
    ACTIVE_COMMANDS,
    ACTIVE_COMPATIBILITY_EVENT_MARKERS,
    ACTIVE_CUE_DURATION_S,
    ACTIVE_IMAGERY_DURATION_S,
    ACTIVE_INTER_TRIAL_INTERVAL_S,
    ACTIVE_N_TRIALS,
    ACTIVE_RANDOMIZED_ORDER,
    ACTIVE_REQUIRED_EVENT_MARKERS,
    ACTIVE_REST_DURATION_S,
)

CommandName = Literal["REST", "IMAGINE_WALKING", "IMAGINE_HAND"]


class MentalCommandProtocol(BaseModel):
    """Validated session protocol for active mental imagery tasks."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="motor_imagery_basic")
    commands: list[CommandName] = Field(default_factory=lambda: list(ACTIVE_COMMANDS))
    cue_duration_s: float = Field(default=ACTIVE_CUE_DURATION_S, gt=0)
    imagery_duration_s: float = Field(default=ACTIVE_IMAGERY_DURATION_S, gt=0)
    rest_duration_s: float = Field(default=ACTIVE_REST_DURATION_S, gt=0)
    n_trials: int = Field(default=ACTIVE_N_TRIALS, ge=2)
    randomized_order: bool = Field(default=ACTIVE_RANDOMIZED_ORDER)
    inter_trial_interval: float = Field(default=ACTIVE_INTER_TRIAL_INTERVAL_S, ge=0)
    required_event_markers: dict[Literal["IMAGINE_WALKING", "IMAGINE_HAND"], str] = Field(
        default_factory=lambda: dict(ACTIVE_REQUIRED_EVENT_MARKERS)
    )
    compatibility_event_markers: dict[CommandName, str] = Field(
        default_factory=lambda: dict(ACTIVE_COMPATIBILITY_EVENT_MARKERS)
    )

    @field_validator("required_event_markers", "compatibility_event_markers")
    @classmethod
    def _validate_marker_values(cls, value: dict[str, str]) -> dict[str, str]:
        for marker in value.values():
            if not marker:
                raise ValueError("event markers must be non-empty")
            normalized = marker.replace("_", "")
            if not normalized.isalnum() or marker != marker.upper():
                raise ValueError(f"event marker must be uppercase alphanumeric/underscore: {marker}")
        return value

    @model_validator(mode="after")
    def _validate_commands(self) -> "MentalCommandProtocol":
        required = {"REST", "IMAGINE_WALKING", "IMAGINE_HAND"}
        actual = set(self.commands)
        if required - actual:
            raise ValueError("commands must include REST, IMAGINE_WALKING, and IMAGINE_HAND")
        return self


def get_protocol(name: str) -> MentalCommandProtocol:
    """Return a validated protocol by name."""
    if name != "motor_imagery_basic":
        raise ValueError(f"Unsupported protocol '{name}'. Available: motor_imagery_basic")
    return MentalCommandProtocol(name=name)
