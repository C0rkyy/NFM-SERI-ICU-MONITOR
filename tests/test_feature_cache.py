"""Tests for feature_store caching and ProcessingLog."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from features.feature_store import (
    ProcessingLog,
    _cache_key,
    _file_checksum,
    load_cached_features,
    save_cached_features,
    session_cache_path,
)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_file(tmp_dir: Path) -> Path:
    """Create a tiny file to checksum."""
    p = tmp_dir / "source.vhdr"
    p.write_text("HeaderFile=source.vhdr\nDataFile=source.eeg\n")
    return p


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [1, 1, 1],
        "session_id": ["s1", "s1", "s1"],
        "epoch_idx": [0, 1, 2],
        "alpha_power": [0.5, 0.6, 0.7],
    })


class TestFileChecksum:
    def test_deterministic(self, sample_file: Path):
        a = _file_checksum(sample_file)
        b = _file_checksum(sample_file)
        assert a == b

    def test_changes_with_content(self, tmp_dir: Path):
        f = tmp_dir / "a.txt"
        f.write_text("hello")
        h1 = _file_checksum(f)
        f.write_text("world")
        h2 = _file_checksum(f)
        assert h1 != h2


class TestCacheKey:
    def test_deterministic(self, sample_file: Path):
        k1 = _cache_key(sample_file, {"skip_ica": True})
        k2 = _cache_key(sample_file, {"skip_ica": True})
        assert k1 == k2

    def test_differs_with_params(self, sample_file: Path):
        k1 = _cache_key(sample_file, {"skip_ica": True})
        k2 = _cache_key(sample_file, {"skip_ica": False})
        assert k1 != k2


class TestSessionCache:
    def test_roundtrip(self, sample_file: Path, sample_df: pd.DataFrame, tmp_dir: Path):
        cache_dir = tmp_dir / "cache"
        save_cached_features(sample_df, "test_session", sample_file, cache_dir=cache_dir)
        loaded = load_cached_features("test_session", sample_file, cache_dir=cache_dir)
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == list(sample_df.columns)

    def test_miss_returns_none(self, sample_file: Path, tmp_dir: Path):
        loaded = load_cached_features("nonexistent", sample_file, cache_dir=tmp_dir)
        assert loaded is None

    def test_invalidates_on_file_change(self, sample_df: pd.DataFrame, tmp_dir: Path):
        cache_dir = tmp_dir / "cache"
        src = tmp_dir / "data.vhdr"
        src.write_text("version1")
        save_cached_features(sample_df, "sess", src, cache_dir=cache_dir)

        # Change source file → different checksum → cache miss
        src.write_text("version2")
        loaded = load_cached_features("sess", src, cache_dir=cache_dir)
        assert loaded is None

    def test_cache_path_deterministic(self, sample_file: Path, tmp_dir: Path):
        p1 = session_cache_path("s1", sample_file, cache_dir=tmp_dir)
        p2 = session_cache_path("s1", sample_file, cache_dir=tmp_dir)
        assert p1 == p2


class TestProcessingLog:
    def test_record_success(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_success("s1", 30, 85.0, 1.5)
        assert log.n_success == 1
        assert log.n_errors == 0

    def test_record_error(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_error("s2", "Load failed: file not found", 0.1)
        assert log.n_errors == 1
        assert log.error_summary()[0]["error"] == "Load failed: file not found"

    def test_record_skip(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_skip("s3", "resume: session file exists")
        assert log.n_skipped == 1

    def test_cached_count(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_success("s1", 10, 50.0, 0.0, cached=True)
        log.record_success("s2", 20, 60.0, 1.0, cached=False)
        assert log.n_cached == 1
        assert log.n_success == 2

    def test_to_dataframe(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_success("s1", 30, 85.0, 1.5)
        log.record_error("s2", "bad data")
        df = log.to_dataframe()
        assert len(df) == 2
        assert "session_key" in df.columns
        assert "status" in df.columns

    def test_save(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        log.record_success("s1", 30, 85.0, 1.5)
        log.record_skip("s2", "already done")
        log.record_error("s3", "corrupt file", 0.5)
        path = log.save(tag="test")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total"] == 3
        assert data["success"] == 1
        assert data["skipped"] == 1
        assert data["errors"] == 1
        # CSV should also exist
        csv_path = tmp_dir / "processing_log_test.csv"
        assert csv_path.exists()

    def test_empty_log(self, tmp_dir: Path):
        log = ProcessingLog(output_dir=tmp_dir)
        assert log.n_success == 0
        df = log.to_dataframe()
        assert df.empty
