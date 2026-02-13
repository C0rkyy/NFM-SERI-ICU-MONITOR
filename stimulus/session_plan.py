from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import RANDOM_STATE, SESSION_STORE_DIR
from stimulus.protocols import MentalCommandProtocol


def _build_command_sequence(protocol: MentalCommandProtocol, seed: int) -> list[str]:
    imagery_commands = [cmd for cmd in protocol.commands if cmd != "REST"]
    repeats = int(np.ceil(protocol.n_trials / max(len(imagery_commands), 1)))
    sequence = (imagery_commands * repeats)[: protocol.n_trials]

    if protocol.randomized_order:
        rng = np.random.default_rng(seed)
        rng.shuffle(sequence)

    return sequence


def build_session_plan(
    protocol: MentalCommandProtocol,
    subject_id: int,
    session_id: str | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Create a machine-readable trial plan for the active command session."""
    resolved_session_id = session_id or f"active_s{subject_id:03d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    sequence = _build_command_sequence(protocol, seed + subject_id)

    trials: list[dict[str, Any]] = []
    for idx, command in enumerate(sequence):
        expected_marker = protocol.required_event_markers.get(command, protocol.compatibility_event_markers[command])
        trials.append(
            {
                "trial_index": idx,
                "command": command,
                "expected_marker": expected_marker,
                "cue_duration_s": protocol.cue_duration_s,
                "imagery_duration_s": protocol.imagery_duration_s,
                "rest_duration_s": protocol.rest_duration_s,
                "inter_trial_interval_s": protocol.inter_trial_interval,
            }
        )

    return {
        "session_id": resolved_session_id,
        "subject_id": subject_id,
        "protocol_name": protocol.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commands": list(protocol.commands),
        "required_event_markers": protocol.required_event_markers,
        "compatibility_event_markers": protocol.compatibility_event_markers,
        "trial_count": len(trials),
        "trials": trials,
    }


def save_session_plan(plan: dict[str, Any], output_dir: Path = SESSION_STORE_DIR) -> Path:
    """Persist a session plan JSON to the session store."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_id = int(plan.get("subject_id", 0))
    session_id = str(plan.get("session_id", "active_session")).replace(" ", "_")
    path = output_dir / f"session_plan_s{subject_id:03d}_{session_id}.json"
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return path
