"""Tests for external validation splits and leakage checks."""
from __future__ import annotations

import pandas as pd
import pytest

from schemas.clinical_validation import SplitType
from validation.splits import auto_split, center_split, center_time_split, time_split


def _make_df(n: int = 20, n_centers: int = 2) -> pd.DataFrame:
    """Create a synthetic test DataFrame."""
    centers = [f"center_{chr(65 + (i % n_centers))}" for i in range(n)]
    return pd.DataFrame({
        "session_id": [f"s{i:03d}" for i in range(n)],
        "subject_id": list(range(n)),
        "center_id": centers,
        "created_at": pd.date_range("2024-01-01", periods=n, freq="D"),
        "SERI": [50 + i for i in range(n)],
    })


class TestCenterSplit:
    def test_basic_split(self):
        df = _make_df(20, 2)
        train, test, spec = center_split(df, ["center_B"])
        assert spec.split_type == SplitType.CENTER
        assert len(train) > 0
        assert len(test) > 0
        assert set(test["center_id"]) == {"center_B"}
        # No session overlap
        assert not set(train["session_id"]) & set(test["session_id"])

    def test_missing_center_col(self):
        df = pd.DataFrame({"session_id": ["s1"], "SERI": [50]})
        with pytest.raises(ValueError, match="not found"):
            center_split(df, ["center_B"])


class TestTimeSplit:
    def test_basic_split(self):
        df = _make_df(20)
        cutoff = "2024-01-11"
        train, test, spec = time_split(df, cutoff)
        assert spec.split_type == SplitType.TIME
        assert len(train) > 0
        assert len(test) > 0

    def test_no_session_overlap(self):
        df = _make_df(20)
        train, test, _ = time_split(df, "2024-01-10")
        overlap = set(train["session_id"]) & set(test["session_id"])
        assert len(overlap) == 0


class TestCenterTimeSplit:
    def test_combined_split(self):
        df = _make_df(20, 2)
        train, test, spec = center_time_split(df, ["center_B"], "2024-01-15")
        assert spec.split_type == SplitType.CENTER_TIME
        assert len(train) > 0
        assert len(test) > 0


class TestAutoSplit:
    def test_center_strategy(self):
        df = _make_df(20, 3)
        train, test, spec = auto_split(df, "center")
        assert spec.split_type == SplitType.CENTER

    def test_time_strategy(self):
        df = _make_df(20, 1)
        train, test, spec = auto_split(df, "time")
        assert spec.split_type == SplitType.TIME

    def test_center_time_strategy(self):
        df = _make_df(20, 2)
        train, test, spec = auto_split(df, "center_time")
        assert len(train) + len(test) >= len(df)  # some may overlap both conditions


class TestLeakageDetection:
    def test_overlap_caught(self):
        """Verify that overlapping session IDs in train/test cause a validation error."""
        from schemas.clinical_validation import ValidationSplitSpec
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="overlap"):
            ValidationSplitSpec(
                split_type=SplitType.CENTER,
                train_ids=["s001", "s002", "s003"],
                test_ids=["s003", "s004"],
            )

    def test_clean_split_passes(self):
        from schemas.clinical_validation import ValidationSplitSpec
        spec = ValidationSplitSpec(
            split_type=SplitType.CENTER,
            train_ids=["s001", "s002"],
            test_ids=["s003", "s004"],
        )
        assert len(spec.test_ids) == 2
