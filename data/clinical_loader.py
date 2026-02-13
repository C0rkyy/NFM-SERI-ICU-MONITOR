"""
Clinical reference standard data loading and harmonisation.

Loads CRS-R assessments, validates against Pydantic schemas, and joins
to NFM session data with strict leakage-safe matching.

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from schemas.clinical_validation import ClinicalAssessment, ReferenceStandardLabel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════

def load_clinical_assessments(path: Path) -> pd.DataFrame:
    """Load CRS-R assessments from CSV.

    Expected columns: patient_id, subject_id, center_id, session_id,
    assessment_time, crsr_total, assessor_id, repeated_exam_index.
    Optional: crsr_subscale_* columns.
    """
    if not path.exists():
        logger.warning("Clinical assessments file not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"patient_id", "subject_id", "center_id", "session_id",
                "assessment_time", "crsr_total", "assessor_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clinical assessments CSV missing columns: {missing}")

    df["assessment_time"] = pd.to_datetime(df["assessment_time"], errors="coerce")
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(0).astype(int)
    df["crsr_total"] = pd.to_numeric(df["crsr_total"], errors="coerce").fillna(0).astype(int)

    if "repeated_exam_index" not in df.columns:
        df["repeated_exam_index"] = 1

    logger.info("Loaded %d clinical assessments from %s", len(df), path)
    return df


def harmonize_assessments(df: pd.DataFrame) -> list[ClinicalAssessment]:
    """Validate each row against the ClinicalAssessment Pydantic schema.

    Returns validated assessments, logs and skips invalid rows.
    """
    validated: list[ClinicalAssessment] = []
    for idx, row in df.iterrows():
        subscales: dict[str, int] = {}
        for col in df.columns:
            if col.startswith("crsr_subscale_"):
                key = col.replace("crsr_subscale_", "")
                val = row.get(col)
                if pd.notna(val):
                    subscales[key] = int(val)

        try:
            entry = ClinicalAssessment(
                patient_id=str(row["patient_id"]),
                subject_id=int(row["subject_id"]),
                center_id=str(row["center_id"]),
                session_id=str(row["session_id"]),
                assessment_time=row["assessment_time"],
                crsr_total=int(row["crsr_total"]),
                crsr_subscales=subscales,
                assessor_id=str(row["assessor_id"]),
                repeated_exam_index=int(row.get("repeated_exam_index", 1)),
            )
            validated.append(entry)
        except Exception as exc:
            logger.warning("Skipping invalid row %d: %s", idx, exc)

    logger.info("Harmonised %d / %d assessments", len(validated), len(df))
    return validated


def derive_reference_labels(
    assessments: list[ClinicalAssessment],
    threshold: int = 10,
) -> list[ReferenceStandardLabel]:
    """Derive binary reference labels from CRS-R total scores.

    Default threshold: CRS-R total >= 10 → label=1 (higher responsiveness).
    """
    labels: list[ReferenceStandardLabel] = []
    for a in assessments:
        label = ReferenceStandardLabel(
            session_id=a.session_id,
            label_target=1 if a.crsr_total >= threshold else 0,
            source="CRS-R",
            confidence=1.0,
        )
        labels.append(label)
    return labels


def join_assessments_to_sessions(
    session_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    *,
    time_window_hours: float = 24.0,
) -> pd.DataFrame:
    """Join clinical assessments to NFM sessions with leakage-safe matching.

    Matching rules:
    1. Match on session_id if both DataFrames have it.
    2. If no session_id match, match on subject_id within time_window_hours.
    3. Leakage guard: assessments from the FUTURE are excluded
       (assessment must precede or coincide with session).
    """
    if session_df.empty or assessments_df.empty:
        logger.warning("Empty input DataFrames; returning session_df as-is.")
        return session_df.copy()

    merged = session_df.copy()
    merged["crsr_total"] = float("nan")
    merged["reference_label"] = float("nan")
    merged["center_id"] = ""

    if "session_id" in session_df.columns and "session_id" in assessments_df.columns:
        sid_map: dict[str, Any] = {}
        for _, arow in assessments_df.iterrows():
            sid = str(arow["session_id"])
            sid_map[sid] = arow

        for idx, row in merged.iterrows():
            sess_id = str(row.get("session_id", ""))
            if sess_id in sid_map:
                arow = sid_map[sess_id]
                merged.at[idx, "crsr_total"] = float(arow["crsr_total"])
                merged.at[idx, "reference_label"] = 1.0 if int(arow["crsr_total"]) >= 10 else 0.0
                merged.at[idx, "center_id"] = str(arow["center_id"])

    # Subject-id + time-window fallback
    unmatched = merged["crsr_total"].isna()
    if unmatched.any() and "subject_id" in merged.columns and "subject_id" in assessments_df.columns:
        for idx in merged.index[unmatched]:
            subj = merged.at[idx, "subject_id"]
            cand = assessments_df[assessments_df["subject_id"] == subj]
            if cand.empty:
                continue
            if "created_at" in merged.columns and "assessment_time" in cand.columns:
                sess_time = pd.to_datetime(merged.at[idx, "created_at"], errors="coerce")
                if pd.isna(sess_time):
                    continue
                cand = cand.copy()
                cand["_dt"] = pd.to_datetime(cand["assessment_time"], errors="coerce")
                cand = cand[cand["_dt"] <= sess_time]  # no future leakage
                cand["_diff"] = (sess_time - cand["_dt"]).abs()
                cand = cand[cand["_diff"] <= pd.Timedelta(hours=time_window_hours)]
                if cand.empty:
                    continue
                best = cand.sort_values("_diff").iloc[0]
            else:
                best = cand.iloc[-1]

            merged.at[idx, "crsr_total"] = float(best["crsr_total"])
            merged.at[idx, "reference_label"] = 1.0 if int(best["crsr_total"]) >= 10 else 0.0
            merged.at[idx, "center_id"] = str(best["center_id"])

    n_matched = int(merged["crsr_total"].notna().sum())
    logger.info("Joined assessments: %d / %d sessions matched", n_matched, len(merged))
    return merged
