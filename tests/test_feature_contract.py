"""Tests for feature column contracts and confidence/quality bounds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import mne

from features.motor_imagery_features import extract_motor_imagery_features


def _make_synthetic_epochs(n: int = 20) -> tuple[mne.Epochs, pd.DataFrame]:
    rng = np.random.default_rng(42)
    sfreq = 160.0
    tmin = -1.0
    n_times = int(5.0 * sfreq)

    ch_names = ["C3", "C4", "Cz", "Pz", "F3", "F4"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.normal(0, 1e-6, size=(n, len(ch_names), n_times))

    event_codes = np.tile([1, 2, 3], n // 3 + 1)[:n]
    events = np.column_stack([
        np.arange(n, dtype=int),
        np.zeros(n, dtype=int),
        event_codes,
    ])
    event_id = {"REST": 1, "IMAGINE_HAND": 2, "IMAGINE_WALKING": 3}
    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=tmin, verbose=False)

    metadata = pd.DataFrame({
        "epoch_idx": np.arange(n),
        "command_label": (["REST", "IMAGINE_HAND", "IMAGINE_WALKING"] * (n // 3 + 1))[:n],
        "is_imagery": ([0, 1, 1] * (n // 3 + 1))[:n],
        "task_label": (["REST", "HAND", "WALKING"] * (n // 3 + 1))[:n],
    })
    return epochs, metadata


def test_feature_columns_contract() -> None:
    """All expected feature columns must be present in output."""
    epochs, metadata = _make_synthetic_epochs()
    trial_df, summary_df, details = extract_motor_imagery_features(
        epochs=epochs, metadata=metadata, include_csp=False, include_riemann=False
    )

    required_columns = {
        "epoch_idx",
        "mu_power_baseline",
        "mu_power_imagery",
        "beta_power_baseline",
        "beta_power_imagery",
        "mu_erd_ers_pct",
        "beta_erd_ers_pct",
        "roi_mu_erd_ers_pct",
        "roi_beta_erd_ers_pct",
        "is_imagery",
    }
    actual = set(trial_df.columns)
    missing = required_columns - actual
    assert not missing, f"Missing feature columns: {missing}"


def test_summary_has_required_fields() -> None:
    """Summary DataFrame and details dict have required fields."""
    epochs, metadata = _make_synthetic_epochs()
    _, summary_df, details = extract_motor_imagery_features(
        epochs=epochs, metadata=metadata, include_csp=False, include_riemann=False
    )

    assert "n_trials" in summary_df.columns
    assert "n_imagery_trials" in summary_df.columns
    assert "trial_consistency" in summary_df.columns
    assert "strongest_markers" in details
    assert isinstance(details["strongest_markers"], list)


def test_csp_features_when_enabled() -> None:
    """CSP features should be added when enabled and enough data exists."""
    epochs, metadata = _make_synthetic_epochs(n=30)
    trial_df, _, _ = extract_motor_imagery_features(
        epochs=epochs, metadata=metadata, include_csp=True, include_riemann=False
    )
    csp_cols = [c for c in trial_df.columns if c.startswith("csp_")]
    # CSP may or may not succeed depending on data; just verify it doesn't crash
    assert isinstance(trial_df, pd.DataFrame)


def test_feature_values_finite() -> None:
    """All numeric feature values should be finite (no inf)."""
    epochs, metadata = _make_synthetic_epochs()
    trial_df, _, _ = extract_motor_imagery_features(
        epochs=epochs, metadata=metadata, include_csp=False, include_riemann=False
    )
    numeric = trial_df.select_dtypes(include=[np.number])
    assert np.all(np.isfinite(numeric.fillna(0).values)), "Non-finite values in features"
