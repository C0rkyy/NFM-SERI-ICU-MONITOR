"""
NFM — Feature Store
=====================
Merges all per-epoch feature DataFrames into a single structured
DataFrame and provides I/O helpers.  Includes session-level caching
with checksum + version keying for reproducible skip-ahead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CACHE_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# Developer note:
# Bump this when we change feature semantics, not just formatting/refactors.
# This forces recompute and protects us from stale cached features.
_FEATURE_STORE_VERSION = "1.0.0"


# Session-level cache helpers.
# Intent: make repeated runs cheap while keeping provenance deterministic.

def _file_checksum(path: Path, algo: str = "sha256") -> str:
    """Compute checksum in chunks so large EEG files do not blow memory."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(source_path: Path, extra_params: dict[str, Any] | None = None) -> str:
    """Cache key ties output to data content + config knobs + feature version."""
    parts = [
        _FEATURE_STORE_VERSION,
        _file_checksum(source_path),
    ]
    if extra_params:
        # Keep key ordering stable so two equal dicts generate the same cache key.
        parts.append(json.dumps(extra_params, sort_keys=True, default=str))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


def session_cache_path(
    session_key: str,
    source_path: Path,
    extra_params: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Return the parquet cache path for a session.

    Parameters
    ----------
    session_key : str
        Human-readable session identifier (e.g. ``"P001_V1_D1_stim"``).
    source_path : Path
        Original data file whose checksum anchors the cache entry.
    extra_params : dict, optional
        Pipeline parameters that affect output (skip_ica, etc.).
    cache_dir : Path, optional
        Override default cache directory.
    """
    cdir = cache_dir or CACHE_DIR
    ck = _cache_key(source_path, extra_params)
    return cdir / f"{session_key}_{ck}.parquet"


def load_cached_features(
    session_key: str,
    source_path: Path,
    extra_params: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Load cached features if the cache entry exists and is valid.

    Returns ``None`` on cache miss (file absent or unreadable).
    """
    path = session_cache_path(session_key, source_path, extra_params, cache_dir)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        logger.debug("Cache hit: %s", path.name)
        return df
    except Exception as exc:
        logger.warning("Corrupt cache entry %s: %s — will recompute", path.name, exc)
        path.unlink(missing_ok=True)
        return None


def save_cached_features(
    df: pd.DataFrame,
    session_key: str,
    source_path: Path,
    extra_params: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Persist feature DataFrame as parquet with checksum-versioned filename."""
    path = session_cache_path(session_key, source_path, extra_params, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.debug("Cache write: %s (%d rows)", path.name, len(df))
    return path


# Run log helpers.
# Intent: make failures visible after long parallel runs without digging in console noise.

class ProcessingLog:
    """Accumulates per-session outcomes and writes a structured summary.

    Developer note:
    workers should never mutate shared state directly. We only append finalized
    entries in the parent process after each future resolves.
    """

    def __init__(self, output_dir: Path | None = None):
        self._entries: list[dict[str, Any]] = []
        self._output_dir = output_dir or RESULTS_DIR

    # Recording helpers (single responsibility: structured event capture).

    def record_success(
        self,
        session_key: str,
        n_trials: int,
        quality_score: float,
        elapsed_s: float,
        *,
        cached: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "session_key": session_key,
            "status": "success",
            "n_trials": n_trials,
            "quality_score": round(quality_score, 2),
            "elapsed_s": round(elapsed_s, 3),
            "cached": cached,
        }
        if extra:
            entry.update(extra)
        self._entries.append(entry)

    def record_skip(self, session_key: str, reason: str) -> None:
        self._entries.append({
            "session_key": session_key,
            "status": "skipped",
            "reason": reason,
        })

    def record_error(self, session_key: str, error: str, elapsed_s: float = 0.0) -> None:
        self._entries.append({
            "session_key": session_key,
            "status": "error",
            "error": error,
            "elapsed_s": round(elapsed_s, 3),
        })

    # Query helpers (used for CLI summary and final report).

    @property
    def n_success(self) -> int:
        return sum(1 for e in self._entries if e["status"] == "success")

    @property
    def n_skipped(self) -> int:
        return sum(1 for e in self._entries if e["status"] == "skipped")

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self._entries if e["status"] == "error")

    @property
    def n_cached(self) -> int:
        return sum(1 for e in self._entries if e.get("cached"))

    def error_summary(self) -> list[dict[str, str]]:
        """Return list of {session_key, error} for all failed sessions."""
        return [
            {"session_key": e["session_key"], "error": e.get("error", e.get("reason", "unknown"))}
            for e in self._entries
            if e["status"] in ("error", "skipped")
        ]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame(self._entries)

    # Persistence (JSON for machines, CSV for quick human inspection).

    def save(self, tag: str = "") -> Path:
        """Write summary as JSON + CSV to the output directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""

        json_path = self._output_dir / f"processing_log{suffix}.json"
        csv_path = self._output_dir / f"processing_log{suffix}.csv"

        summary = {
            "total": len(self._entries),
            "success": self.n_success,
            "cached": self.n_cached,
            "skipped": self.n_skipped,
            "errors": self.n_errors,
            "error_details": self.error_summary(),
            "entries": self._entries,
        }
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(csv_path, index=False)

        logger.info(
            "Processing log saved: %d total (%d success, %d cached, %d skipped, %d errors) → %s",
            len(self._entries), self.n_success, self.n_cached, self.n_skipped, self.n_errors,
            json_path.name,
        )
        return json_path


def merge_features(
    erp_df: pd.DataFrame,
    spectral_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    subject_id: int = 0,
    session_id: str = "default",
) -> pd.DataFrame:
    """
    Left-join all feature tables on ``epoch_idx`` and tag with subject.

    Returns
    -------
    pd.DataFrame — one row per epoch with all features.
    """
    df = erp_df.copy()
    for other in [spectral_df, tf_df, connectivity_df]:
        # Developer note:
        # duplicate columns happen when some extractors emit channel metadata.
        # We keep the left-most copy so downstream column names stay stable.
        merge_cols = ["epoch_idx"]
        df = df.merge(other, on=merge_cols, how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        df.drop(columns=dup_cols, inplace=True)

    df.insert(0, "subject_id", subject_id)
    df.insert(1, "session_id", session_id)

    logger.info("Merged feature matrix → %s", df.shape)
    return df


def save_features(df: pd.DataFrame, tag: str = "") -> Path:
    """Persist to CSV and return path."""
    fname = f"features{'_' + tag if tag else ''}.csv"
    path = RESULTS_DIR / fname
    df.to_csv(path, index=False)
    logger.info("Saved features → %s", path)
    return path


def load_features(path: Path) -> pd.DataFrame:
    """Read a previously saved feature CSV."""
    return pd.read_csv(path)
