"""Tests for quality score computation."""
from __future__ import annotations

from scoring.quality import compute_quality_score


def test_quality_score_bounds() -> None:
    """QualityScore must be in [0, 100] for any valid inputs."""
    q = compute_quality_score(
        epoch_drop_rate=0.1,
        bad_channel_fraction=0.05,
        line_noise_residual=0.1,
        snr_proxy=1.2,
    )
    assert 0.0 <= q["QualityScore"] <= 100.0


def test_perfect_quality() -> None:
    """Perfect data metrics yield high quality."""
    q = compute_quality_score(
        epoch_drop_rate=0.0,
        bad_channel_fraction=0.0,
        line_noise_residual=0.0,
        snr_proxy=2.0,  # high SNR
    )
    assert q["QualityScore"] >= 80.0


def test_bad_quality() -> None:
    """Very bad data metrics yield low quality."""
    q = compute_quality_score(
        epoch_drop_rate=0.9,
        bad_channel_fraction=0.5,
        line_noise_residual=0.8,
        snr_proxy=0.8,  # low SNR
    )
    assert q["QualityScore"] < 40.0


def test_quality_components_present() -> None:
    """All quality components should be in the result."""
    q = compute_quality_score(
        epoch_drop_rate=0.2,
        bad_channel_fraction=0.1,
        line_noise_residual=0.15,
        snr_proxy=1.1,
    )
    assert "components" in q
    comps = q["components"]
    assert "epoch_component" in comps
    assert "bad_channel_component" in comps
    assert "line_noise_component" in comps
    assert "snr_component" in comps
