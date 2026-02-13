from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ACTIVE_BASELINE_WINDOW,
    ACTIVE_IMAGERY_WINDOW,
    CACHE_DIR,
    DEFAULT_N_JOBS,
    FIGURES_DIR,
    LIS_DATASET_ROOT,
    LOCAL_DATASET_ROOTS,
    OUTPUT_DIR,
    PHYSIONET_RUNS,
    PHYSIONET_SUBJECTS,
    RANDOM_STATE,
    RESULTS_DIR,
    SESSION_STORE_DIR,
    SFREQ,
    STIMULUS_BASELINE_WINDOW,
    STIMULUS_RESPONSE_WINDOW,
    set_global_seed,
)
from data.downloader import discover_local_edf_files, load_eeg_data
from data.loader import load_subject
from features.connectivity import compute_average_coherence_matrix, extract_connectivity_features
from features.erp_features import compute_grand_average_erp, extract_erp_features
from features.feature_store import (
    ProcessingLog,
    load_cached_features,
    merge_features,
    save_cached_features,
    save_features,
)
from features.motor_imagery_features import extract_motor_imagery_features
from features.spectral_features import extract_spectral_features
from features.time_frequency import extract_tf_features, get_average_tfr
from models.command_following import train_command_following
from preprocessing.epochs import extract_command_epochs, extract_epochs, separate_baseline_stimulus
from preprocessing.pipeline import preprocess
from scoring.cfi import compute_cfi
from scoring.frs import compute_frs, subject_frs_summary
from scoring.quality import compute_quality_score, estimate_line_noise_residual, estimate_snr_proxy
from scoring.sensitivity import sensitivity_analysis
from schemas.session import (
    FeatureSummary,
    ModelOutputs,
    Patient,
    RecordingSession,
    Scores,
    StimulusProtocol as SessionStimulusProtocol,
    save_recording_session,
)
from statistics.group_comparison import full_group_comparison
from stimulus.protocols import get_protocol
from stimulus.session_plan import build_session_plan, save_session_plan

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _process_one_subject(
    subj_id: int,
    edf_paths: list[Path],
    protocol_name: str,
    skip_ica: bool,
    include_csp: bool,
    include_riemann: bool,
    random_state: int,
) -> dict[str, Any] | None:
    """Process a single subject: load → preprocess → features → quality.

    Designed to run in a separate process via ProcessPoolExecutor.
    Returns a serialisable dict with all results, or None if the subject
    should be skipped.
    """
    import mne
    mne.set_log_level("WARNING")
    set_global_seed(random_state)

    protocol = get_protocol(protocol_name)

    subject_data = load_subject(edf_paths, subj_id)
    raw = subject_data["raw"]
    events = subject_data["events"]
    event_id = subject_data["event_id"]
    raw_clean = preprocess(raw, skip_ica=skip_ica)

    session_plan = build_session_plan(protocol=protocol, subject_id=subj_id, seed=random_state)
    plan_path = save_session_plan(session_plan)
    session_id = str(session_plan["session_id"])

    try:
        command_epochs, metadata, epoch_stats = extract_command_epochs(
            raw=raw_clean,
            events=events,
            event_id=event_id,
            baseline_window=ACTIVE_BASELINE_WINDOW,
            imagery_window=ACTIVE_IMAGERY_WINDOW,
            marker_map=protocol.compatibility_event_markers,
        )
    except ValueError:
        return None

    if len(command_epochs) < 6:
        return None

    trial_df, summary_df, summary_details = extract_motor_imagery_features(
        epochs=command_epochs,
        metadata=metadata,
        baseline_window=ACTIVE_BASELINE_WINDOW,
        imagery_window=ACTIVE_IMAGERY_WINDOW,
        include_csp=include_csp,
        include_riemann=include_riemann,
    )
    trial_df.insert(0, "subject_id", subj_id)
    trial_df.insert(1, "session_id", session_id)

    line_noise_ratio = estimate_line_noise_residual(raw_clean)
    snr_proxy = estimate_snr_proxy(command_epochs, ACTIVE_BASELINE_WINDOW, ACTIVE_IMAGERY_WINDOW)
    quality = compute_quality_score(
        epoch_drop_rate=float(epoch_stats["epoch_drop_rate"]),
        bad_channel_fraction=float(epoch_stats["bad_channel_fraction"]),
        line_noise_residual=line_noise_ratio,
        snr_proxy=snr_proxy,
    )

    return {
        "subj_id": subj_id,
        "trial_df": trial_df,
        "session_id": session_id,
        "plan_path": str(plan_path),
        "summary": summary_df.iloc[0].to_dict(),
        "summary_details": summary_details,
        "quality": quality,
        "epoch_stats": epoch_stats,
    }


def _process_one_subject_passive(
    subj_id: int,
    edf_paths: list[Path],
    skip_ica: bool,
    include_epoch_arrays: bool,
    random_state: int,
) -> dict[str, Any] | None:
    """Process one subject for passive mode in a worker process."""
    import mne
    from scipy.signal import welch as _welch

    mne.set_log_level("WARNING")
    set_global_seed(random_state)

    subject_data = load_subject(edf_paths, subj_id)
    raw = subject_data["raw"]
    events = subject_data["events"]
    event_id = subject_data["event_id"]

    raw_clean = preprocess(raw, skip_ica=skip_ica)
    epochs = extract_epochs(raw_clean, events, event_id)
    if len(epochs) < 4:
        return None

    baseline_data, stimulus_data = separate_baseline_stimulus(epochs)

    erp_df = extract_erp_features(epochs)
    spectral_df = extract_spectral_features(epochs, baseline_data=baseline_data)
    tf_df = extract_tf_features(epochs)
    conn_df, coh_matrices = extract_connectivity_features(epochs)

    feat_df = merge_features(erp_df, spectral_df, tf_df, conn_df, subject_id=subj_id)

    ep_data: np.ndarray | None = None
    labels: np.ndarray | None = None
    if include_epoch_arrays:
        ep_data = epochs.get_data()
        median_amp = np.median(erp_df["p300_amplitude"].values)
        labels = (erp_df["p300_amplitude"].values >= median_amp).astype(int)

    erp_wave = compute_grand_average_erp(epochs)
    avg_tfr, tfr_freqs, tfr_times = get_average_tfr(epochs)
    avg_coh = compute_average_coherence_matrix(coh_matrices)

    nperseg_bl = min(baseline_data.shape[-1], int(SFREQ))
    f_bl, psd_bl = _welch(baseline_data.mean(axis=0), fs=SFREQ, nperseg=nperseg_bl, axis=-1)
    psd_bl_avg = psd_bl.mean(axis=0)

    nperseg_st = min(stimulus_data.shape[-1], int(SFREQ))
    _, psd_st = _welch(stimulus_data.mean(axis=0), fs=SFREQ, nperseg=nperseg_st, axis=-1)
    psd_st_avg = psd_st.mean(axis=0)

    return {
        "subj_id": subj_id,
        "features": feat_df,
        "epochs_data": ep_data,
        "labels": labels,
        "dashboard_cache": {
            "erp_times": epochs.times.tolist(),
            "erp_waveform": erp_wave.tolist(),
            "psd_freqs": f_bl.tolist(),
            "psd_baseline": psd_bl_avg.tolist(),
            "psd_stimulus": psd_st_avg.tolist(),
            "tfr_power": avg_tfr.tolist(),
            "tfr_freqs": tfr_freqs.tolist(),
            "tfr_times": tfr_times.tolist(),
            "coherence_matrix": avg_coh.tolist(),
            "ch_names": epochs.ch_names,
        },
    }
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NFM")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neuro Functional Monitor pipeline")
    p.add_argument("--subjects", nargs="+", type=int, default=None, help="Subject IDs to process")
    p.add_argument("--runs", nargs="+", type=int, default=None, help="Run numbers to process")
    p.add_argument("--skip-download", action="store_true", help="Use existing local EDF files")
    p.add_argument("--skip-cnn", action="store_true", help="Skip CNN training in passive mode")
    p.add_argument("--skip-ica", action="store_true", help="Skip ICA preprocessing")
    p.add_argument("--mode", choices=["passive", "active", "stimulus", "both"], default="passive")
    p.add_argument("--protocol", choices=["motor_imagery_basic"], default="motor_imagery_basic")
    p.add_argument("--skip-active", action="store_true", help="Skip active pipeline even if mode=both")
    p.add_argument(
        "--active-feature-profile",
        choices=["fast", "balanced", "full"],
        default="balanced",
        help="Feature depth for active mode: fast=no CSP/Riemann, balanced=CSP only, full=CSP+Riemann",
    )
    # New CLI flags
    p.add_argument("--dataset-root", type=str, default=None, help="Extra dataset root directory for EDF discovery")
    p.add_argument("--stimulus-dataset-root", type=str, default=None, help="LIS BrainVision dataset root directory")
    p.add_argument("--all-local-subjects", action="store_true", help="Process all subjects found locally (default in active+skip-download)")
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                   help=f"Number of parallel jobs (default: {DEFAULT_N_JOBS} = cpu_count-1)")
    p.add_argument("--cache-dir", type=str, default=None, help="Directory for cached intermediate results")
    p.add_argument("--resume", action="store_true", help="Skip subjects that already have results")
    # Clinical validation flags
    p.add_argument("--clinical-validation", choices=["none", "retrospective", "prospective"],
                   default="none", help="Clinical validation mode")
    p.add_argument("--clinical-labels", type=str, default=None,
                   help="Path to CRS-R clinical labels CSV")
    p.add_argument("--site-map", type=str, default=None,
                   help="Path to site mapping CSV")
    p.add_argument("--split-strategy", choices=["center", "time", "center_time"],
                   default="center_time", help="Validation split strategy")
    p.add_argument("--drift-baseline", type=str, default=None,
                   help="Path to baseline features CSV for drift monitoring")
    p.add_argument("--generate-clinical-report", action="store_true",
                   help="Generate clinical evidence report")
    return p.parse_args()


def _load_edf_map(args: argparse.Namespace) -> dict[int, list[Path]]:
    """Load EDF file map, preferring local discovery when --skip-download is set."""
    subjects_requested = args.subjects
    runs_requested = args.runs or PHYSIONET_RUNS
    extra_roots: list[Path] = []

    if args.dataset_root:
        extra_roots.append(Path(args.dataset_root))

    if args.skip_download:
        # Local mode: discover from all roots
        edf_map = discover_local_edf_files(
            extra_roots=extra_roots if extra_roots else None,
            subjects=subjects_requested,
            runs=runs_requested,
        )
        if not edf_map:
            logger.warning("No local EDF files found. Trying PhysioNet download fallback...")
            edf_map = load_eeg_data(
                dataset="physionet",
                subjects=subjects_requested or list(range(1, 11)),
                runs=runs_requested,
            )
        return edf_map

    return load_eeg_data(
        dataset="physionet",
        subjects=subjects_requested or PHYSIONET_SUBJECTS,
        runs=runs_requested,
    )


def run_passive_pipeline(args: argparse.Namespace) -> None:
    """Original passive FRS workflow."""
    set_global_seed(RANDOM_STATE)
    edf_map = _load_edf_map(args)
    if not edf_map:
        logger.error("No EDF files found. Exiting passive pipeline.")
        return

    all_features: list[pd.DataFrame] = []
    all_epochs_data: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    dashboard_cache: dict[int, dict[str, Any]] = {}
    skipped_subjects: list[int] = []
    include_epoch_arrays = not args.skip_cnn
    effective_workers = min(max(1, args.n_jobs), len(edf_map))

    logger.info(
        "Passive mode: %d subjects, n_jobs=%d, skip_cnn=%s",
        len(edf_map),
        effective_workers,
        args.skip_cnn,
    )

    if effective_workers > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures: dict[Any, int] = {}
            for subj_id, edf_paths in sorted(edf_map.items()):
                futures[
                    executor.submit(
                        _process_one_subject_passive,
                        subj_id,
                        edf_paths,
                        args.skip_ica,
                        include_epoch_arrays,
                        RANDOM_STATE,
                    )
                ] = subj_id

            for fut in as_completed(futures):
                subj_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning("Skipping S%03d in passive flow: worker error: %s", subj_id, exc)
                    skipped_subjects.append(subj_id)
                    continue

                if result is None:
                    logger.warning("Skipping S%03d in passive flow: too few epochs", subj_id)
                    skipped_subjects.append(subj_id)
                    continue

                all_features.append(result["features"])
                dashboard_cache[subj_id] = result["dashboard_cache"]
                if include_epoch_arrays and result["epochs_data"] is not None and result["labels"] is not None:
                    all_epochs_data.append(result["epochs_data"])
                    all_labels.append(result["labels"])
                logger.info("Passive subject done: S%03d (%d rows)", subj_id, len(result["features"]))
    else:
        for subj_id, edf_paths in sorted(edf_map.items()):
            logger.info("Passive pipeline subject S%03d", subj_id)
            try:
                result = _process_one_subject_passive(
                    subj_id=subj_id,
                    edf_paths=edf_paths,
                    skip_ica=args.skip_ica,
                    include_epoch_arrays=include_epoch_arrays,
                    random_state=RANDOM_STATE,
                )
            except Exception as exc:
                logger.warning("Skipping S%03d in passive flow: %s", subj_id, exc)
                skipped_subjects.append(subj_id)
                continue

            if result is None:
                logger.warning("Too few epochs for S%03d in passive flow", subj_id)
                skipped_subjects.append(subj_id)
                continue

            all_features.append(result["features"])
            dashboard_cache[subj_id] = result["dashboard_cache"]
            if include_epoch_arrays and result["epochs_data"] is not None and result["labels"] is not None:
                all_epochs_data.append(result["epochs_data"])
                all_labels.append(result["labels"])

    logger.info(
        "Passive feature extraction complete: %d subjects processed, %d skipped",
        len(all_features),
        len(skipped_subjects),
    )

    if not all_features:
        logger.error("No valid passive features generated.")
        return

    full_df = pd.concat(all_features, ignore_index=True)
    save_features(full_df, tag="all_subjects")

    full_epochs: np.ndarray | None = None
    full_labels: np.ndarray | None = None
    if include_epoch_arrays and all_epochs_data and all_labels:
        full_epochs = np.concatenate(all_epochs_data, axis=0)
        full_labels = np.concatenate(all_labels, axis=0)

    from models.random_forest import predict_proba, save_model as save_rf, train_random_forest

    rf_results = train_random_forest(full_df)
    save_rf(rf_results, tag="rf")
    rf_proba = predict_proba(rf_results["model"], rf_results["scaler"], full_df)

    cnn_results: dict[str, Any] | None = None
    cnn_proba: np.ndarray | None = None
    if not args.skip_cnn:
        try:
            from models.deep_learning import predict_proba_cnn, save_cnn, train_cnn

            if full_epochs is None or full_labels is None:
                raise ValueError("CNN arrays unavailable in passive flow")
            cnn_results = train_cnn(full_epochs, full_labels)
            save_cnn(cnn_results["model"], tag="cnn")
            cnn_proba = predict_proba_cnn(cnn_results["model"], full_epochs)
        except Exception as exc:
            logger.warning("CNN training skipped due to error: %s", exc)

    from models.evaluation import plot_dual_roc, plot_roc_curve, summarise_metrics

    if cnn_results:
        print(summarise_metrics(rf_results, cnn_results, names=["Random Forest", "CNN"]))
    else:
        print(summarise_metrics(rf_results, names=["Random Forest"]))

    model_proba = 0.5 * rf_proba + 0.5 * cnn_proba if cnn_proba is not None else rf_proba

    frs_df = compute_frs(full_df, model_proba=model_proba)
    frs_df.to_csv(RESULTS_DIR / "frs_results.csv", index=False)

    sens_df = sensitivity_analysis(full_df, model_proba=model_proba)
    sens_df.to_csv(RESULTS_DIR / "sensitivity_analysis.csv", index=False)

    subj_summary = subject_frs_summary(frs_df)
    subj_summary.to_csv(RESULTS_DIR / "subject_summary.csv", index=False)

    if len(subj_summary) >= 4:
        gc = full_group_comparison(subj_summary)
        gc["combined_df"].to_csv(RESULTS_DIR / "group_comparison.csv", index=False)
        group_stats = {
            "t_stat": gc["ttest"]["t_stat"],
            "p_value": gc["ttest"]["p_value"],
            "cohens_d": gc["cohens_d"],
            "roc_auc": gc["roc"]["roc_auc"],
        }
        (RESULTS_DIR / "group_stats.json").write_text(json.dumps(group_stats, indent=2), encoding="utf-8")

    plot_roc_curve(rf_results, title="Random Forest ROC", save_path=RESULTS_DIR / "roc_rf.png")
    if cnn_results:
        plot_roc_curve(cnn_results, title="CNN ROC", save_path=RESULTS_DIR / "roc_cnn.png")
        plot_dual_roc(rf_results, cnn_results, save_path=RESULTS_DIR / "roc_comparison.png")

    roc_data: dict[str, Any] = {
        "rf_fpr": rf_results["fpr"].tolist(),
        "rf_tpr": rf_results["tpr"].tolist(),
        "rf_auc": rf_results["roc_auc"],
    }
    if cnn_results:
        roc_data.update(
            {
                "cnn_fpr": cnn_results["fpr"].tolist(),
                "cnn_tpr": cnn_results["tpr"].tolist(),
                "cnn_auc": cnn_results["roc_auc"],
            }
        )
    (RESULTS_DIR / "roc_data.json").write_text(json.dumps(roc_data), encoding="utf-8")

    with (RESULTS_DIR / "dashboard_cache.pkl").open("wb") as f:
        pickle.dump(dashboard_cache, f)

    output_cols = [
        "subject_id",
        "session_id",
        "frs",
        "p300_amplitude",
        "norm_band_power_shift",
        "norm_model_probability",
    ]
    existing = [c for c in output_cols if c in frs_df.columns]
    frs_df[existing].to_csv(RESULTS_DIR / "nfm_output.csv", index=False)

    logger.info("Passive pipeline complete. Outputs: %s", RESULTS_DIR)


def _render_cfi_summary(results: pd.DataFrame, save_path: Path) -> None:
    if results.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d9534f" if v < 40 else "#f0ad4e" if v < 70 else "#5cb85c" for v in results["CFI"]]
    ax.bar(results["subject_id"].astype(str), results["CFI"], color=colors)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Command-Following Index")
    ax.set_title("Command-Following Indicator Summary")
    for idx, value in enumerate(results["CFI"]):
        ax.text(idx, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_active_pipeline(args: argparse.Namespace) -> None:
    """Active mental imagery pipeline producing command-following indicators."""
    t_pipeline_start = time.perf_counter()
    set_global_seed(RANDOM_STATE)

    # Propagate n_jobs to submodules via env var
    n_jobs = max(1, args.n_jobs)
    os.environ["NFM_N_JOBS"] = str(n_jobs)

    protocol = get_protocol(args.protocol)
    edf_map = _load_edf_map(args)
    if not edf_map:
        logger.error("No EDF files found. Exiting active pipeline.")
        return

    logger.info("=" * 60)
    logger.info("ACTIVE PIPELINE: %d subjects, profile=%s", len(edf_map), args.active_feature_profile)
    logger.info("=" * 60)

    subject_context: dict[int, dict[str, Any]] = {}
    trial_frames: list[pd.DataFrame] = []
    proc_log_active = ProcessingLog(output_dir=RESULTS_DIR)

    include_csp = args.active_feature_profile in {"balanced", "full"}
    include_riemann = args.active_feature_profile == "full"
    logger.info(
        "Active feature profile=%s (CSP=%s, Riemann=%s)",
        args.active_feature_profile,
        include_csp,
        include_riemann,
    )

    skipped_subjects: list[int] = []
    t_feature_start = time.perf_counter()

    # Parallel subject-level feature extraction
    effective_workers = min(n_jobs, len(edf_map))
    if effective_workers > 1:
        logger.info("Parallel feature extraction: %d workers", effective_workers)
        futures: dict[Any, int] = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for subj_id, edf_paths in sorted(edf_map.items()):
                # Resume support: skip if already processed
                if args.resume:
                    session_files = list(SESSION_STORE_DIR.glob(f"session_s{subj_id:03d}_*.json"))
                    if session_files:
                        logger.info("  Skipping S%03d (resume: session file exists)", subj_id)
                        continue
                fut = executor.submit(
                    _process_one_subject,
                    subj_id,
                    edf_paths,
                    args.protocol,
                    args.skip_ica,
                    include_csp,
                    include_riemann,
                    RANDOM_STATE,
                )
                futures[fut] = subj_id

            for fut in as_completed(futures):
                subj_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning("Skipping S%03d: worker error: %s", subj_id, exc)
                    skipped_subjects.append(subj_id)
                    proc_log_active.record_error(f"S{subj_id:03d}", str(exc))
                    continue
                if result is None:
                    skipped_subjects.append(subj_id)
                    proc_log_active.record_skip(f"S{subj_id:03d}", "returned None")
                    continue
                trial_frames.append(result["trial_df"])
                subject_context[result["subj_id"]] = {
                    "session_id": result["session_id"],
                    "plan_path": result["plan_path"],
                    "summary": result["summary"],
                    "summary_details": result["summary_details"],
                    "quality": result["quality"],
                    "epoch_stats": result["epoch_stats"],
                }
                proc_log_active.record_success(
                    f"S{result['subj_id']:03d}", len(result["trial_df"]),
                    result["quality"]["QualityScore"], 0.0,
                )
                logger.info("  S%03d done: %d trials, quality=%.1f",
                            result["subj_id"], len(result["trial_df"]),
                            result["quality"]["QualityScore"])
    else:
        # Sequential fallback
        for subj_id, edf_paths in sorted(edf_map.items()):
            t_subj = time.perf_counter()
            logger.info("Active pipeline subject S%03d (%d files)", subj_id, len(edf_paths))

            # Resume support: skip if already processed
            if args.resume:
                session_files = list(SESSION_STORE_DIR.glob(f"session_s{subj_id:03d}_*.json"))
                if session_files:
                    logger.info("  Skipping S%03d (resume: session file exists)", subj_id)
                    continue

            result = _process_one_subject(
                subj_id, edf_paths, args.protocol,
                args.skip_ica, include_csp, include_riemann, RANDOM_STATE,
            )
            if result is None:
                skipped_subjects.append(subj_id)
                proc_log_active.record_skip(f"S{subj_id:03d}", "returned None")
                continue

            trial_frames.append(result["trial_df"])
            subject_context[result["subj_id"]] = {
                "session_id": result["session_id"],
                "plan_path": result["plan_path"],
                "summary": result["summary"],
                "summary_details": result["summary_details"],
                "quality": result["quality"],
                "epoch_stats": result["epoch_stats"],
            }
            proc_log_active.record_success(
                f"S{subj_id:03d}", len(result["trial_df"]),
                result["quality"]["QualityScore"], time.perf_counter() - t_subj,
            )

            logger.info("  S%03d done: %d trials, quality=%.1f (%.1fs)",
                         subj_id, len(result["trial_df"]), result["quality"]["QualityScore"],
                         time.perf_counter() - t_subj)

    t_feature_end = time.perf_counter()
    proc_log_active.save(tag="active")
    logger.info("Feature extraction: %.1fs for %d subjects (%d skipped)",
                t_feature_end - t_feature_start, len(trial_frames), len(skipped_subjects))

    if not trial_frames:
        logger.error("Active pipeline produced no usable trials.")
        return

    trials_all = pd.concat(trial_frames, ignore_index=True)
    logger.info("Total trials: %d from %d subjects", len(trials_all), trials_all["subject_id"].nunique())

    t_model_start = time.perf_counter()
    model_results = train_command_following(trials_all)
    t_model_end = time.perf_counter()
    logger.info("Model training: %.1fs (n_jobs=%d)", t_model_end - t_model_start, n_jobs)

    trials_all["imagery_probability"] = model_results["probabilities"]

    # Extract calibration diagnostics
    brier_score = model_results["metrics"].get("brier_score")
    ece_score = model_results["metrics"].get("ece")

    reliability_candidates = [
        model_results["metrics"].get("across_subject_auc"),
        model_results["metrics"].get("within_subject_auc"),
        model_results["metrics"].get("train_auc"),
    ]
    reliability_values = [float(v) for v in reliability_candidates if v is not None]
    model_reliability = float(np.clip(np.mean(reliability_values), 0.0, 1.0)) if reliability_values else 0.75

    t_scoring_start = time.perf_counter()
    subject_rows: list[dict[str, Any]] = []

    for subj_id, grp in trials_all.groupby("subject_id"):
        context = subject_context.get(int(subj_id), {})
        quality_score = float(context.get("quality", {}).get("QualityScore", 0.0))
        cfi_result = compute_cfi(
            imagery_probabilities=grp["imagery_probability"].to_numpy(dtype=float),
            quality_score=quality_score,
            model_reliability=model_reliability,
            brier_score=brier_score,
            ece=ece_score,
        )

        trials_all.loc[grp.index, "CFI"] = cfi_result["CFI"]
        trials_all.loc[grp.index, "Confidence"] = cfi_result["Confidence"]
        trials_all.loc[grp.index, "QualityScore"] = cfi_result["QualityScore"]
        trials_all.loc[grp.index, "consistency"] = cfi_result["consistency"]
        trials_all.loc[grp.index, "Evidence"] = cfi_result["evidence_label"]
        trials_all.loc[grp.index, "ConfidenceLevel"] = cfi_result["confidence_label"]
        trials_all.loc[grp.index, "QualityLevel"] = cfi_result["quality_label"]
        trials_all.loc[grp.index, "ClinicalSummary"] = cfi_result["clinical_summary"]

        subject_rows.append(
            {
                "subject_id": int(subj_id),
                "session_id": str(context.get("session_id", "active_session")),
                "CFI": float(cfi_result["CFI"]),
                "Confidence": float(cfi_result["Confidence"]),
                "QualityScore": float(cfi_result["QualityScore"]),
                "Evidence": str(cfi_result["evidence_label"]),
                "ConfidenceLevel": str(cfi_result["confidence_label"]),
                "QualityLevel": str(cfi_result["quality_label"]),
                "ClinicalSummary": str(cfi_result["clinical_summary"]),
                "mean_imagery_probability": float(cfi_result["mean_imagery_probability"]),
                "consistency": float(cfi_result["consistency"]),
                "ci_lower": float(cfi_result.get("ci_lower", 0.0)),
                "ci_upper": float(cfi_result.get("ci_upper", 1.0)),
                "n_trials": int(len(grp)),
                # Confidence component breakdown for dashboard
                "sharpness": float(cfi_result.get("sharpness", 0.0)),
                "usable_trial_score": float(cfi_result.get("usable_trial_score", 0.0)),
                "reliability_score": float(cfi_result.get("reliability_score", 0.0)),
                "ci_tightness": float(cfi_result.get("ci_tightness", 0.0)),
            }
        )

        summary = context.get("summary", {})
        summary_details = context.get("summary_details", {})
        protocol_schema = SessionStimulusProtocol(
            name=protocol.name,
            commands=list(protocol.commands),
            cue_duration_s=protocol.cue_duration_s,
            imagery_duration_s=protocol.imagery_duration_s,
            rest_duration_s=protocol.rest_duration_s,
            n_trials=protocol.n_trials,
            randomized_order=protocol.randomized_order,
            inter_trial_interval=protocol.inter_trial_interval,
        )

        feature_summary = FeatureSummary(
            n_trials=int(summary.get("n_trials", len(grp))),
            n_imagery_trials=int(summary.get("n_imagery_trials", int(grp["is_imagery"].sum()))),
            roi_channels_available=list(summary_details.get("roi_channels_available", [])),
            strongest_markers=list(summary_details.get("strongest_markers", [])),
            trial_consistency=float(summary_details.get("trial_consistency", cfi_result["consistency"])),
            mu_erd_mean=float(summary.get("mu_erd_mean", grp["roi_mu_erd_ers_pct"].mean() if "roi_mu_erd_ers_pct" in grp.columns else 0.0)),
            beta_erd_mean=float(summary.get("beta_erd_mean", grp["roi_beta_erd_ers_pct"].mean() if "roi_beta_erd_ers_pct" in grp.columns else 0.0)),
        )

        model_outputs = ModelOutputs(
            model_name="ensemble_lr_hgb_calibrated",
            calibration_method=str(model_results["calibration_method"]),
            cv_within_subject_auc=model_results["metrics"].get("within_subject_auc"),
            cv_across_subject_auc=model_results["metrics"].get("across_subject_auc"),
            artifact_paths=model_results["artifact_paths"],
        )

        session_payload = RecordingSession(
            session_id=str(context.get("session_id", "active_session")),
            patient=Patient(patient_id=f"S{subj_id:03d}", subject_id=int(subj_id)),
            protocol=protocol_schema,
            feature_summary=feature_summary,
            model_outputs=model_outputs,
            scores=Scores(
                FRS=None,
                CFI=float(cfi_result["CFI"]),
                Quality=float(cfi_result["QualityScore"]),
                Confidence=float(cfi_result["Confidence"]),
            ),
        )
        save_recording_session(session_payload, SESSION_STORE_DIR)

    result_df = pd.DataFrame(subject_rows).sort_values("subject_id")
    detailed_path = OUTPUT_DIR / "results_command_following.csv"
    result_df.to_csv(detailed_path, index=False)

    trials_all.to_csv(RESULTS_DIR / "command_following_trials.csv", index=False)

    cfi_figure_path = FIGURES_DIR / "cfi_summary.png"
    _render_cfi_summary(result_df, cfi_figure_path)

    t_scoring_end = time.perf_counter()
    t_pipeline_end = time.perf_counter()
    total_runtime = t_pipeline_end - t_pipeline_start

    # ── AGGREGATE STATS REPORT ──────────────────────────────────
    n_subjects_processed = result_df["subject_id"].nunique()
    median_confidence = float(result_df["Confidence"].median()) if not result_df.empty else 0.0
    pct_conf_70 = float((result_df["Confidence"] >= 70).mean() * 100) if not result_df.empty else 0.0
    median_quality = float(result_df["QualityScore"].median()) if not result_df.empty else 0.0
    across_auc = model_results["metrics"].get("across_subject_auc", "N/A")

    # High quality subset
    hq = result_df[result_df["QualityScore"] >= 70] if not result_df.empty else pd.DataFrame()
    hq_median_conf = float(hq["Confidence"].median()) if not hq.empty else 0.0
    hq_pct_conf_70 = float((hq["Confidence"] >= 70).mean() * 100) if not hq.empty else 0.0

    # Save runtime metrics to model metrics file
    runtime_metrics = model_results["metrics"].copy()
    runtime_metrics.update({
        "n_subjects_processed": n_subjects_processed,
        "n_subjects_skipped": len(skipped_subjects),
        "median_confidence": median_confidence,
        "pct_sessions_confidence_ge_70": pct_conf_70,
        "hq_median_confidence": hq_median_conf,
        "hq_pct_sessions_confidence_ge_70": hq_pct_conf_70,
        "median_quality": median_quality,
        "total_runtime_s": round(total_runtime, 2),
        "feature_extraction_time_s": round(t_feature_end - t_feature_start, 2),
        "model_training_time_s": round(t_model_end - t_model_start, 2),
        "scoring_time_s": round(t_scoring_end - t_scoring_start, 2),
        "n_jobs": n_jobs,
    })
    metrics_path = OUTPUT_DIR / "models" / "command_following_metrics.json"
    metrics_path.write_text(json.dumps(runtime_metrics, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("ACTIVE PIPELINE COMPLETE")
    logger.info("  Subjects processed: %d", n_subjects_processed)
    logger.info("  Subjects skipped: %d", len(skipped_subjects))
    logger.info("  Median Confidence: %.1f", median_confidence)
    logger.info("  %% sessions Confidence >= 70: %.1f%%", pct_conf_70)
    logger.info("  Median QualityScore: %.1f", median_quality)
    logger.info("  Across-subject AUC: %s", across_auc)
    logger.info("  Brier score: %s", brier_score)
    logger.info("  ECE: %s", ece_score)
    if not hq.empty:
        logger.info("  [High-quality subset QS>=70]")
        logger.info("    Median Confidence: %.1f", hq_median_conf)
        logger.info("    %% Confidence >= 70: %.1f%%", hq_pct_conf_70)
    logger.info("  ── Runtime breakdown ──")
    logger.info("    Feature extraction:  %.1fs", t_feature_end - t_feature_start)
    logger.info("    Model training:      %.1fs (n_jobs=%d)", t_model_end - t_model_start, n_jobs)
    logger.info("    Scoring + session I/O: %.1fs", t_scoring_end - t_scoring_start)
    logger.info("    Total:               %.1fs", total_runtime)
    logger.info("  Results: %s", detailed_path)
    logger.info("  Session store: %s", SESSION_STORE_DIR)
    logger.info("=" * 60)

    # Bottleneck analysis if target not met
    if hq_median_conf < 70 and not hq.empty:
        logger.warning("=" * 60)
        logger.warning("BOTTLENECK ANALYSIS: High-quality median confidence %.1f < 70", hq_median_conf)
        logger.warning("Potential causes:")
        if across_auc and float(across_auc) < 0.70:
            logger.warning("  - Across-subject AUC is %.3f (low generalization)", float(across_auc))
        if brier_score and float(brier_score) > 0.20:
            logger.warning("  - Brier score is %.4f (calibration could improve)", float(brier_score))
        if ece_score and float(ece_score) > 0.10:
            logger.warning("  - ECE is %.4f (predicted probabilities are miscalibrated)", float(ece_score))
        if median_quality < 75:
            logger.warning("  - Median quality is %.1f (data quality limits confidence)", median_quality)
        logger.warning("Remediation steps:")
        logger.warning("  1. Use --active-feature-profile full for CSP+Riemann features")
        logger.warning("  2. Ensure all 109 subjects are included for training diversity")
        logger.warning("  3. Consider artifact rejection threshold tuning")
        logger.warning("  4. Add more runs (beyond 3,7,11) for additional training data")
        logger.warning("=" * 60)


# ═══════════════════════════════════════════════════════════════
# Stimulus-evoked response pipeline worker
# ═══════════════════════════════════════════════════════════════
def _process_one_stimulus_session(
    session_info: dict[str, Any],
    skip_ica: bool,
    random_state: int,
) -> dict[str, Any] | None:
    """Process a single BrainVision stimulus session in a worker process."""
    import mne
    mne.set_log_level("WARNING")
    set_global_seed(random_state)

    from data.brainvision_loader import (
        extract_brainvision_events,
        load_brainvision_raw,
    )
    from features.stimulus_features import (
        compute_grand_average_erp as stim_grand_avg,
        extract_stimulus_response_features,
    )
    from preprocessing.stimulus_epochs import (
        extract_stimulus_epochs,
        separate_baseline_response,
    )
    from stimulus.event_extraction import (
        extract_stimulus_events,
        get_stimulus_onset_events,
        validate_stimulus_session,
    )

    vhdr_path = Path(session_info["file_path"])
    session_dir = vhdr_path.parent

    try:
        raw = load_brainvision_raw(vhdr_path)
    except Exception as exc:
        return {"error": f"Load failed: {exc}", "session_info": session_info}

    # Lightweight preprocessing: bandpass filter only (BrainVision data)
    from preprocessing.pipeline import bandpass_filter, notch_filter
    try:
        raw = bandpass_filter(raw, l_freq=0.5, h_freq=40.0)
        raw = notch_filter(raw, freqs=[50.0])
    except Exception:
        pass  # Proceed with raw data if filtering fails

    # Extract events
    events, event_id = extract_brainvision_events(raw)
    sfreq = float(raw.info["sfreq"])

    # Parse and classify stimulus events
    stim_events_df = extract_stimulus_events(
        events=events,
        event_id=event_id,
        sfreq=sfreq,
        session_dir=session_dir,
    )

    # Validate session
    validation = validate_stimulus_session(stim_events_df, sfreq)
    if not validation["valid"]:
        return {
            "error": f"Validation failed: {'; '.join(validation['issues'])}",
            "session_info": session_info,
        }

    # Get stimulus onset events for epoching
    onset_events, onset_event_id = get_stimulus_onset_events(stim_events_df, event_id)
    if onset_events.shape[0] == 0:
        # Fallback: use all stimulus events
        onset_events = events
        onset_event_id = event_id

    # Extract epochs
    try:
        epochs, metadata, epoch_stats = extract_stimulus_epochs(
            raw=raw,
            events=onset_events,
            event_id=onset_event_id,
            baseline_window=STIMULUS_BASELINE_WINDOW,
            response_window=STIMULUS_RESPONSE_WINDOW,
        )
    except Exception as exc:
        return {"error": f"Epoching failed: {exc}", "session_info": session_info}

    if len(epochs) < 3:
        return {"error": "Too few epochs after rejection", "session_info": session_info}

    # Extract features
    trial_df, summary_df, summary_details = extract_stimulus_response_features(
        epochs=epochs,
        metadata=metadata,
        baseline_window=STIMULUS_BASELINE_WINDOW,
        response_window=STIMULUS_RESPONSE_WINDOW,
    )

    # Add session identifiers
    trial_df.insert(0, "subject_id", session_info["subject_id"])
    trial_df.insert(1, "session_id", f"{session_info['visit_id']}_{session_info['day_id']}_{session_info['session_type']}")

    # Quality computation
    from scoring.quality import compute_quality_score as cqs
    # Estimate SNR from epochs
    baseline_data, response_data = separate_baseline_response(epochs)
    bl_rms = float(np.sqrt(np.mean(np.square(baseline_data)))) if baseline_data.size > 0 else 1e-12
    resp_rms = float(np.sqrt(np.mean(np.square(response_data)))) if response_data.size > 0 else 0.0
    snr_proxy = resp_rms / (bl_rms + 1e-12)

    line_noise_residual = estimate_line_noise_residual(raw)
    quality = cqs(
        epoch_drop_rate=float(epoch_stats["epoch_drop_rate"]),
        bad_channel_fraction=float(epoch_stats["bad_channel_fraction"]),
        line_noise_residual=line_noise_residual,
        snr_proxy=snr_proxy,
    )

    # Compute ERP waveform for caching
    erp_waveform = stim_grand_avg(epochs)

    return {
        "session_info": session_info,
        "trial_df": trial_df,
        "summary": summary_df.iloc[0].to_dict() if len(summary_df) > 0 else {},
        "summary_details": summary_details,
        "quality": quality,
        "epoch_stats": epoch_stats,
        "erp_waveform": erp_waveform.tolist(),
        "erp_times": epochs.times.tolist(),
        "n_trials": len(epochs),
        "snr_proxy": snr_proxy,
    }


def _render_seri_summary(results: pd.DataFrame, save_path: Path) -> None:
    """Render SERI summary bar chart."""
    if results.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = results["subject_id"].astype(str) + "/" + results["session_id"].astype(str)
    if len(labels) > 20:
        labels = results["subject_id"].astype(str)
    colors = ["#d9534f" if v < 40 else "#f0ad4e" if v < 70 else "#5cb85c" for v in results["SERI"]]
    ax.bar(range(len(results)), results["SERI"], color=colors, tick_label=labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Subject / Session")
    ax.set_ylabel("SERI (Stimulus-Evoked Response Index)")
    ax.set_title("Stimulus-Evoked Response Indicator Summary")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    for idx, value in enumerate(results["SERI"]):
        ax.text(idx, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _derive_stimulus_response_labels(trials_all: pd.DataFrame) -> tuple[pd.Series, str]:
    """Derive response labels for stimulus-locked modulation detection.

    Priority:
    1) Use explicit binary stimulus labels from sidecar metadata (if available).
    2) Fill remaining/absent labels with a rule-based ERP+spectral response score.
    """
    # Developer note:
    # this is intentionally conservative. If curated labels are missing, we still
    # need a reproducible fallback to keep pipeline runs comparable.
    label_series = pd.Series(np.nan, index=trials_all.index, dtype=float)
    label_source = "rule_based_erp_spectral"

    if "stimulus_label" in trials_all.columns:
        # Trust sidecar labels first when they are truly binary and sufficiently present.
        raw = pd.to_numeric(trials_all["stimulus_label"], errors="coerce")
        mask_binary = raw.isin([0.0, 1.0])
        if int(mask_binary.sum()) >= 20 and raw[mask_binary].nunique() == 2:
            label_series.loc[mask_binary] = raw.loc[mask_binary]
            label_source = "stimulus_label"

    # Rule score fallback: combine robust ERP + spectral cues into one ranking value.
    score = pd.Series(0.0, index=trials_all.index, dtype=float)
    weights = {
        "p300_amplitude": 0.35,
        "n100_amplitude": 0.20,
        "peak_to_peak": 0.20,
        "alpha_pct_change": 0.15,
        "beta_pct_change": 0.10,
    }
    used_weight = 0.0
    for col, weight in weights.items():
        if col not in trials_all.columns:
            continue
        vals = pd.to_numeric(trials_all[col], errors="coerce")
        vals = vals.fillna(vals.median() if vals.notna().any() else 0.0)
        vals = vals.abs()
        std = float(vals.std())
        if std <= 1e-12:
            zvals = pd.Series(0.0, index=vals.index, dtype=float)
        else:
            zvals = (vals - float(vals.mean())) / std
        score = score + weight * zvals
        used_weight += weight

    if used_weight > 0:
        score = score / used_weight
    # Slightly-above-median threshold keeps positive class from dominating.
    threshold = float(score.quantile(0.55)) if score.notna().any() else 0.0
    fallback_labels = (score >= threshold).astype(int)

    missing_mask = label_series.isna()
    label_series.loc[missing_mask] = fallback_labels.loc[missing_mask]
    if label_source == "stimulus_label" and missing_mask.any():
        label_source = "hybrid_stimulus_label_plus_rule"

    return label_series.astype(int), label_source


def run_stimulus_pipeline(args: argparse.Namespace) -> None:
    """Run the stimulus-evoked response analysis pipeline on LIS BrainVision data."""
    t_pipeline_start = time.perf_counter()
    set_global_seed(RANDOM_STATE)

    n_jobs = max(1, args.n_jobs)
    os.environ["NFM_N_JOBS"] = str(n_jobs)

    # Developer note:
    # keep dataset root explicit in logs; this avoids silent confusion when
    # multiple local copies exist.
    # Determine dataset root
    stimulus_root = Path(args.stimulus_dataset_root) if args.stimulus_dataset_root else LIS_DATASET_ROOT
    if not stimulus_root.exists():
        logger.error("Stimulus dataset root not found: %s", stimulus_root)
        return

    logger.info("=" * 60)
    logger.info("STIMULUS-EVOKED RESPONSE PIPELINE")
    logger.info("  Dataset root: %s", stimulus_root)
    logger.info("  n_jobs: %d", n_jobs)
    logger.info("=" * 60)

    # Discover sessions
    from data.brainvision_loader import build_stimulus_dataset_registry, discover_brainvision_sessions
    sessions = discover_brainvision_sessions(stimulus_root)

    if not sessions:
        logger.error("No BrainVision sessions found under %s", stimulus_root)
        return

    # Registry pass is expensive but worth it: it gives us auditability on what
    # was discovered vs. what was actually usable.
    # Build and save registry
    t_registry_start = time.perf_counter()
    registry_df = build_stimulus_dataset_registry(stimulus_root)
    t_registry_end = time.perf_counter()
    logger.info("Registry: %d sessions discovered (%.1fs)", len(registry_df), t_registry_end - t_registry_start)

    # ── Cache & resume setup ────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    proc_log = ProcessingLog(output_dir=RESULTS_DIR)
    cache_params = {"skip_ica": args.skip_ica, "sfreq": SFREQ}

    # Session processing is the true bottleneck. We parallelize here and keep
    # per-session context for downstream scoring/reporting.
    # Process sessions in parallel
    t_feature_start = time.perf_counter()
    trial_frames: list[pd.DataFrame] = []
    session_context: dict[str, dict[str, Any]] = {}
    session_context_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: list[str] = []
    errors: list[str] = []

    def _handle_stimulus_result(key: str, si: dict, result: dict | None, t0: float) -> None:
        """Shared bookkeeping for a processed stimulus session."""
        elapsed = time.perf_counter() - t0
        if result is None:
            skipped.append(key)
            proc_log.record_skip(key, "returned None")
            return
        if "error" in result:
            logger.warning("Skipping %s: %s", key, result["error"])
            errors.append(f"{key}: {result['error']}")
            proc_log.record_error(key, result["error"], elapsed)
            return
        trial_frames.append(result["trial_df"])
        session_context[key] = result
        subj_key = str(result["session_info"]["subject_id"])
        sess_key = str(result["trial_df"]["session_id"].iloc[0])
        session_context_by_pair[(subj_key, sess_key)] = result
        quality_val = result["quality"]["QualityScore"] if isinstance(result.get("quality"), dict) else 0.0
        # Cache features for future runs
        source = Path(si["file_path"])
        save_cached_features(result["trial_df"], key, source, cache_params, cache_dir)
        proc_log.record_success(key, result["n_trials"], quality_val, elapsed)
        logger.info("  %s done: %d trials, quality=%.1f (%.1fs)",
                    key, result["n_trials"], quality_val, elapsed)

    effective_workers = min(n_jobs, len(sessions))

    if effective_workers > 1:
        logger.info("Parallel stimulus processing: %d workers for %d sessions", effective_workers, len(sessions))
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for si in sessions:
                key = f"{si['subject_id']}_{si['visit_id']}_{si['day_id']}_{si['session_type']}"
                source = Path(si["file_path"])

                # ── Cache / resume check ──────────────
                if args.resume:
                    session_files = list(SESSION_STORE_DIR.glob(f"session_{si['subject_id']}_*.json"))
                    if session_files:
                        logger.info("  Skipping %s (resume: session file exists)", key)
                        skipped.append(key)
                        proc_log.record_skip(key, "resume: session file exists")
                        continue

                cached_df = load_cached_features(key, source, cache_params, cache_dir)
                if cached_df is not None:
                    # Reconstruct minimal result from cache
                    trial_frames.append(cached_df)
                    proc_log.record_success(key, len(cached_df), 0.0, 0.0, cached=True)
                    logger.info("  %s loaded from cache (%d trials)", key, len(cached_df))
                    continue

                futures[executor.submit(
                    _process_one_stimulus_session, si, args.skip_ica, RANDOM_STATE
                )] = (key, si)

            for fut in as_completed(futures):
                key, si = futures[fut]
                t0 = time.perf_counter()
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning("Worker error for %s: %s", key, exc)
                    errors.append(f"{key}: {exc}")
                    proc_log.record_error(key, str(exc))
                    continue
                _handle_stimulus_result(key, si, result, t0)
    else:
        for si in sessions:
            key = f"{si['subject_id']}_{si['visit_id']}_{si['day_id']}_{si['session_type']}"
            source = Path(si["file_path"])

            # ── Cache / resume check ──────────────
            if args.resume:
                session_files = list(SESSION_STORE_DIR.glob(f"session_{si['subject_id']}_*.json"))
                if session_files:
                    logger.info("  Skipping %s (resume: session file exists)", key)
                    skipped.append(key)
                    proc_log.record_skip(key, "resume: session file exists")
                    continue

            cached_df = load_cached_features(key, source, cache_params, cache_dir)
            if cached_df is not None:
                trial_frames.append(cached_df)
                proc_log.record_success(key, len(cached_df), 0.0, 0.0, cached=True)
                logger.info("  %s loaded from cache (%d trials)", key, len(cached_df))
                continue

            logger.info("Processing stimulus session: %s", key)
            t0 = time.perf_counter()
            try:
                result = _process_one_stimulus_session(si, args.skip_ica, RANDOM_STATE)
            except Exception as exc:
                logger.warning("Error processing %s: %s", key, exc)
                errors.append(f"{key}: {exc}")
                proc_log.record_error(key, str(exc), time.perf_counter() - t0)
                continue
            _handle_stimulus_result(key, si, result, t0)

    t_feature_end = time.perf_counter()
    # Save structured processing log
    proc_log.save(tag="stimulus")
    logger.info("Feature extraction: %.1fs for %d sessions (%d errors, %d skipped, %d cached)",
                t_feature_end - t_feature_start, len(trial_frames), len(errors), len(skipped), proc_log.n_cached)

    if not trial_frames:
        logger.error("Stimulus pipeline produced no usable trials.")
        if errors:
            logger.error("Errors encountered: %s", "; ".join(errors[:10]))
        return

    # Concatenate all trials
    trials_all = pd.concat(trial_frames, ignore_index=True)
    logger.info("Total stimulus trials: %d from %d sessions",
                len(trials_all), trials_all.groupby(["subject_id", "session_id"]).ngroups)

    # Train stimulus response model
    t_model_start = time.perf_counter()
    from models.stimulus_response import train_stimulus_response

    # Generate response labels aligned with stimulus-evoked modulation (not command intent).
    response_labels, label_source = _derive_stimulus_response_labels(trials_all)
    trials_all["is_response"] = response_labels
    logger.info(
        "Stimulus response labels: source=%s, positives=%d/%d (%.1f%%)",
        label_source,
        int(trials_all["is_response"].sum()),
        len(trials_all),
        100.0 * float(trials_all["is_response"].mean()) if len(trials_all) else 0.0,
    )

    model_results = train_stimulus_response(trials_all, label_col="is_response")
    t_model_end = time.perf_counter()
    logger.info("Model training: %.1fs", t_model_end - t_model_start)

    # Use out-of-fold probabilities for SERI scoring to avoid optimistic in-sample inflation.
    scoring_probs = model_results.get("scoring_probabilities", model_results["probabilities"])
    scoring_probs_arr = np.asarray(scoring_probs, dtype=float)
    if scoring_probs_arr.shape[0] != len(trials_all):
        scoring_probs_arr = np.asarray(model_results["probabilities"], dtype=float)
    if np.isnan(scoring_probs_arr).any():
        fallback_probs = np.asarray(model_results["probabilities"], dtype=float)
        scoring_probs_arr = np.where(np.isnan(scoring_probs_arr), fallback_probs, scoring_probs_arr)
    trials_all["response_probability"] = scoring_probs_arr

    # Compute SERI per session (patient-facing summary is session-level, not trial-level).
    from scoring.seri import compute_seri
    t_scoring_start = time.perf_counter()
    subject_rows: list[dict[str, Any]] = []

    for (subj_id, sess_id), grp in trials_all.groupby(["subject_id", "session_id"]):
        key = f"{subj_id}_{sess_id}"
        ctx = session_context_by_pair.get((str(subj_id), str(sess_id)))

        quality_score = float(ctx["quality"]["QualityScore"]) if ctx else 50.0
        mean_snr = float(grp["evoked_snr"].mean()) if "evoked_snr" in grp.columns else 1.0
        consistency = float(ctx["summary_details"].get("trial_consistency", 0.5)) if ctx else 0.5

        model_reliability_candidates = [
            model_results["metrics"].get("across_subject_auc"),
            model_results["metrics"].get("within_subject_auc"),
            model_results["metrics"].get("train_auc"),
        ]
        model_reliability = float(np.mean([v for v in model_reliability_candidates if v is not None])) if any(v is not None for v in model_reliability_candidates) else 0.75

        seri_result = compute_seri(
            response_probabilities=grp["response_probability"].to_numpy(dtype=float),
            quality_score=quality_score,
            evoked_snr=mean_snr,
            response_consistency=consistency,
            model_reliability=model_reliability,
        )

        # Assign to trials
        for col_name in ["SERI", "Confidence", "QualityScore"]:
            trials_all.loc[grp.index, col_name] = seri_result[col_name]
        trials_all.loc[grp.index, "Evidence"] = seri_result["evidence_label"]

        summary = ctx.get("summary", {}) if ctx else {}
        strongest = ctx["summary_details"].get("strongest_markers", []) if ctx else []

        subject_rows.append({
            "subject_id": str(subj_id),
            "session_id": str(sess_id),
            "SERI": float(seri_result["SERI"]),
            "Confidence": float(seri_result["Confidence"]),
            "QualityScore": float(seri_result["QualityScore"]),
            "Evidence": str(seri_result["evidence_label"]),
            "ConfidenceLevel": str(seri_result["confidence_label"]),
            "QualityLevel": str(seri_result["quality_label"]),
            "ClinicalSummary": str(seri_result["clinical_summary"]),
            "mean_response_probability": float(seri_result["mean_response_probability"]),
            "consistency": float(seri_result["consistency"]),
            "evoked_snr": mean_snr,
            "n_trials": len(grp),
            "label_source": label_source,
            "mean_p300_amp": float(summary.get("mean_p300_amp", 0.0)),
            "mean_n100_amp": float(summary.get("mean_n100_amp", 0.0)),
        })

        # Save session JSON
        try:
            # Parse subject_id as int for patient
            try:
                subj_int = int(str(subj_id).replace("P", ""))
            except ValueError:
                subj_int = hash(str(subj_id)) % 10000

            stim_protocol = SessionStimulusProtocol(
                name="stimulus_evoked_response",
                commands=["STIMULUS_ONSET"],
                cue_duration_s=0.5,
                imagery_duration_s=0.8,
                rest_duration_s=1.0,
                n_trials=len(grp),
                randomized_order=False,
                inter_trial_interval=2.0,
            )

            feature_summary = FeatureSummary(
                n_trials=len(grp),
                n_response_trials=int(grp["is_response"].sum()),
                strongest_markers=strongest[:5],
                trial_consistency=float(consistency),
                mean_evoked_snr=float(mean_snr),
                mean_p300_amp=float(summary.get("mean_p300_amp", 0.0)),
            )

            model_outputs = ModelOutputs(
                model_name="stimulus_response_hgb",
                calibration_method=str(model_results["calibration_method"]),
                cv_within_subject_auc=model_results["metrics"].get("within_subject_auc"),
                cv_across_subject_auc=model_results["metrics"].get("across_subject_auc"),
                artifact_paths=model_results["artifact_paths"],
            )

            session_payload = RecordingSession(
                session_id=str(sess_id),
                patient=Patient(patient_id=f"S{subj_int:03d}", subject_id=subj_int),
                protocol=stim_protocol,
                feature_summary=feature_summary,
                model_outputs=model_outputs,
                scores=Scores(
                    SERI=float(seri_result["SERI"]),
                    Quality=float(seri_result["QualityScore"]),
                    Confidence=float(seri_result["Confidence"]),
                ),
                mode="stimulus",
            )
            save_recording_session(session_payload, SESSION_STORE_DIR)
        except Exception as exc:
            logger.warning("Failed to save session JSON for %s: %s", key, exc)

    # Save results
    result_df = pd.DataFrame(subject_rows)
    stimulus_output_path = OUTPUT_DIR / "results_stimulus_response.csv"
    result_df.to_csv(stimulus_output_path, index=False)

    trials_all.to_csv(RESULTS_DIR / "stimulus_response_trials.csv", index=False)

    # Save SERI summary figure
    seri_figure_path = FIGURES_DIR / "seri_summary.png"
    _render_seri_summary(result_df, seri_figure_path)

    t_scoring_end = time.perf_counter()
    t_pipeline_end = time.perf_counter()

    # ── Summary metrics ───────────────────────────────────────
    n_sessions = len(result_df)
    median_seri = float(result_df["SERI"].median()) if not result_df.empty else 0.0
    median_conf = float(result_df["Confidence"].median()) if not result_df.empty else 0.0
    median_quality = float(result_df["QualityScore"].median()) if not result_df.empty else 0.0

    runtime_metrics = {
        "n_sessions_processed": n_sessions,
        "n_sessions_skipped": len(skipped),
        "n_errors": len(errors),
        "label_source": label_source,
        "median_seri": median_seri,
        "median_confidence": median_conf,
        "median_quality": median_quality,
        "total_runtime_s": round(t_pipeline_end - t_pipeline_start, 2),
        "registry_time_s": round(t_registry_end - t_registry_start, 2),
        "feature_time_s": round(t_feature_end - t_feature_start, 2),
        "model_time_s": round(t_model_end - t_model_start, 2),
        "scoring_time_s": round(t_scoring_end - t_scoring_start, 2),
        "n_jobs": n_jobs,
    }
    runtime_metrics.update({k: v for k, v in model_results["metrics"].items() if k != "feature_columns"})
    metrics_path = OUTPUT_DIR / "models" / "stimulus_response_pipeline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(runtime_metrics, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("STIMULUS-EVOKED RESPONSE PIPELINE COMPLETE")
    logger.info("  Sessions processed: %d", n_sessions)
    logger.info("  Sessions skipped/errored: %d / %d", len(skipped), len(errors))
    logger.info("  Median SERI: %.1f", median_seri)
    logger.info("  Median Confidence: %.1f", median_conf)
    logger.info("  Median QualityScore: %.1f", median_quality)
    logger.info("  ── Runtime breakdown ──")
    logger.info("    Registry discovery:  %.1fs", t_registry_end - t_registry_start)
    logger.info("    Feature extraction:  %.1fs (n_jobs=%d)", t_feature_end - t_feature_start, n_jobs)
    logger.info("    Model training:      %.1fs", t_model_end - t_model_start)
    logger.info("    Scoring + I/O:       %.1fs", t_scoring_end - t_scoring_start)
    logger.info("    Total:               %.1fs", t_pipeline_end - t_pipeline_start)
    logger.info("  Results: %s", stimulus_output_path)
    logger.info("  Session store: %s", SESSION_STORE_DIR)
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()

    if args.mode in {"passive", "both"}:
        run_passive_pipeline(args)

    if args.mode in {"active", "both"}:
        if args.skip_active:
            logger.info("Active mode skipped due to --skip-active")
        else:
            run_active_pipeline(args)

    if args.mode in {"stimulus", "both"}:
        run_stimulus_pipeline(args)

    # Clinical validation post-processing
    if args.clinical_validation != "none":
        run_clinical_validation(args)


def run_clinical_validation(args: argparse.Namespace) -> None:
    """Run clinical validation, calibration, drift, and prospective workflow."""
    from data.clinical_loader import (
        derive_reference_labels,
        harmonize_assessments,
        join_assessments_to_sessions,
        load_clinical_assessments,
    )
    from monitoring.drift import build_drift_report, plot_drift_dashboard, save_drift_report
    from reports.clinical_report import build_run_manifest, generate_clinical_report
    from studies.prospective import save_prospective_outputs, simulate_prospective_run
    from validation.calibration_monitor import (
        build_calibration_report,
        plot_reliability_curve,
        save_calibration_report,
    )
    from validation.external_validation import run_external_validation, save_validation_outputs
    from validation.splits import auto_split

    CLINICAL_OUT = OUTPUT_DIR / "clinical"
    CLINICAL_OUT.mkdir(parents=True, exist_ok=True)

    # Load stimulus results
    stim_csv = OUTPUT_DIR / "results_stimulus_response.csv"
    if not stim_csv.exists():
        logger.warning("No stimulus results found at %s; skipping clinical validation", stim_csv)
        return
    results_df = pd.read_csv(stim_csv)
    logger.info("Clinical validation: loaded %d result rows from %s", len(results_df), stim_csv)

    # Load trials for drift monitoring
    trials_csv = RESULTS_DIR / "stimulus_response_trials.csv"
    trials_df = pd.read_csv(trials_csv) if trials_csv.exists() else pd.DataFrame()

    # ── 1. Clinical labels + join ──────────────────────────
    assessments_df = pd.DataFrame()
    validation_metrics_dict: dict | None = None

    if args.clinical_labels:
        labels_path = Path(args.clinical_labels)
        assessments_df = load_clinical_assessments(labels_path)
        if not assessments_df.empty:
            validated = harmonize_assessments(assessments_df)
            labels = derive_reference_labels(validated)
            logger.info("Derived %d reference labels from %d assessments", len(labels), len(validated))

            # Join to results
            results_df = join_assessments_to_sessions(results_df, assessments_df)

            # Add probability column if missing
            if "response_probability" not in results_df.columns and "SERI" in results_df.columns:
                results_df["response_probability"] = results_df["SERI"] / 100.0

            # ── 2. External validation ─────────────────────
            if args.clinical_validation == "retrospective" and "reference_label" in results_df.columns:
                labelled = results_df.dropna(subset=["reference_label"])
                if len(labelled) >= 5:
                    try:
                        train_df, test_df, split_spec = auto_split(
                            labelled, args.split_strategy,
                        )
                        val_result = run_external_validation(
                            results_df, split_spec,
                            prob_col="response_probability",
                            label_col="reference_label",
                        )
                        save_validation_outputs(val_result, CLINICAL_OUT)
                        if val_result.get("metrics"):
                            validation_metrics_dict = val_result["metrics"].model_dump()
                            logger.info("External validation AUC: %.3f", val_result["metrics"].auc)
                    except Exception as exc:
                        logger.warning("External validation failed: %s", exc)
                else:
                    logger.warning("Too few labelled samples (%d) for external validation", len(labelled))
        else:
            logger.warning("No clinical assessments loaded from %s", labels_path)
    else:
        logger.info("No --clinical-labels provided; skipping reference-based validation")

    # ── 3. Calibration ─────────────────────────────────────
    calibration_dict: dict | None = None
    if "reference_label" in results_df.columns and "response_probability" in results_df.columns:
        labelled = results_df.dropna(subset=["reference_label", "response_probability"])
        if len(labelled) >= 5:
            cal_report = build_calibration_report(
                labelled,
                prob_col="response_probability",
                label_col="reference_label",
            )
            save_calibration_report(cal_report, CLINICAL_OUT)
            plot_reliability_curve(cal_report, FIGURES_DIR / "calibration_reliability.png")
            calibration_dict = cal_report
            logger.info("Calibration report saved (ECE=%.4f)", cal_report.get("overall", {}).get("ece", 0))

    # ── 4. Drift monitoring ────────────────────────────────
    drift_dict: dict | None = None
    if args.drift_baseline and Path(args.drift_baseline).exists():
        baseline_df = pd.read_csv(Path(args.drift_baseline))
        snapshot = build_drift_report(
            baseline_df=baseline_df,
            current_df=trials_df if not trials_df.empty else results_df,
        )
        save_drift_report(snapshot, CLINICAL_OUT)
        plot_drift_dashboard(snapshot, FIGURES_DIR / "drift_dashboard.png")
        drift_dict = json.loads(snapshot.model_dump_json())
        logger.info("Drift report: alert=%s", snapshot.alert_level.value)
    elif not trials_df.empty:
        # Self-drift baseline (compare first half vs second half)
        mid = len(trials_df) // 2
        if mid >= 10:
            snapshot = build_drift_report(
                baseline_df=trials_df.iloc[:mid],
                current_df=trials_df.iloc[mid:],
            )
            save_drift_report(snapshot, CLINICAL_OUT)
            plot_drift_dashboard(snapshot, FIGURES_DIR / "drift_dashboard.png")
            drift_dict = json.loads(snapshot.model_dump_json())
            logger.info("Self-drift report: alert=%s", snapshot.alert_level.value)

    # ── 5. Prospective workflow ────────────────────────────
    prospective_metrics: dict | None = None
    if args.clinical_validation == "prospective" or True:  # always run prospective
        prosp_result = simulate_prospective_run(results_df, score_col="SERI")
        save_prospective_outputs(prosp_result, CLINICAL_OUT)
        prospective_metrics = prosp_result["metrics"]
        logger.info("Prospective workflow: %d sessions, invalid_rate=%.1f%%",
                    prospective_metrics["n_sessions"],
                    prospective_metrics["invalid_rate"] * 100)

    # ── 6. Run manifest ────────────────────────────────────
    manifest = build_run_manifest(
        seed=RANDOM_STATE,
        data_paths=[str(stim_csv)],
        args={
            "mode": args.mode,
            "clinical_validation": args.clinical_validation,
            "split_strategy": args.split_strategy,
            "skip_ica": args.skip_ica,
            "n_jobs": args.n_jobs,
        },
    )
    manifest_path = CLINICAL_OUT / "run_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Run manifest saved: %s", manifest_path)

    # ── 7. Clinical report ─────────────────────────────────
    if args.generate_clinical_report:
        cohort_info = {
            "dataset": "LIS BrainVision",
            "n_subjects": int(results_df["subject_id"].nunique()) if "subject_id" in results_df.columns else 0,
            "n_sessions": len(results_df),
            "n_trials": len(trials_df) if not trials_df.empty else 0,
        }
        generate_clinical_report(
            output_dir=CLINICAL_OUT,
            validation_metrics=validation_metrics_dict,
            calibration_report=calibration_dict,
            drift_report=drift_dict,
            prospective_metrics=prospective_metrics,
            run_manifest=manifest,
            cohort_info=cohort_info,
        )
        logger.info("Clinical evidence report generated in %s", CLINICAL_OUT)

    logger.info("Clinical validation pipeline complete.")


if __name__ == "__main__":
    main()
