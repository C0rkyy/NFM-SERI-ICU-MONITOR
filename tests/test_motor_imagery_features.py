from __future__ import annotations

import numpy as np
import pandas as pd
import mne

from features.motor_imagery_features import extract_motor_imagery_features


def _synthetic_epochs() -> tuple[mne.Epochs, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_epochs = 12
    sfreq = 160.0
    tmin = -1.0
    n_times = int(5.0 * sfreq)

    ch_names = ["C3", "C4", "Cz", "Pz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.normal(0, 1e-6, size=(n_epochs, len(ch_names), n_times))

    event_codes = np.array([1, 2, 3] * 4, dtype=int)
    events = np.column_stack([
        np.arange(n_epochs, dtype=int),
        np.zeros(n_epochs, dtype=int),
        event_codes,
    ])
    event_id = {"REST": 1, "IMAGINE_HAND": 2, "IMAGINE_WALKING": 3}

    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=tmin, verbose=False)

    metadata = pd.DataFrame(
        {
            "epoch_idx": np.arange(n_epochs),
            "command_label": ["REST", "IMAGINE_HAND", "IMAGINE_WALKING"] * 4,
            "is_imagery": [0, 1, 1] * 4,
            "task_label": ["REST", "HAND", "WALKING"] * 4,
        }
    )
    return epochs, metadata


def test_motor_imagery_feature_columns() -> None:
    epochs, metadata = _synthetic_epochs()
    trial_df, summary_df, summary_details = extract_motor_imagery_features(
        epochs=epochs,
        metadata=metadata,
        include_csp=False,
        include_riemann=False,
    )

    expected_cols = {
        "mu_power_baseline",
        "mu_power_imagery",
        "beta_power_baseline",
        "beta_power_imagery",
        "mu_erd_ers_pct",
        "beta_erd_ers_pct",
        "roi_mu_erd_ers_pct",
        "roi_beta_erd_ers_pct",
    }
    assert expected_cols.issubset(set(trial_df.columns))
    assert len(summary_df) == 1
    assert isinstance(summary_details.get("strongest_markers", []), list)
