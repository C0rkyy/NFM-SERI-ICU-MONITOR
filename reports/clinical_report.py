"""
Clinical evidence report generator.

Produces a structured Markdown and JSON report covering:
- Data sources and cohort description
- Model version + config hash
- Validation design
- Performance and calibration
- Drift status
- Operational readiness
- Known limitations

All outputs: decision-support only, not a diagnosis.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from schemas.clinical_validation import RunManifest

logger = logging.getLogger(__name__)


def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _config_hash() -> str:
    """Hash key configuration values for reproducibility tracking."""
    try:
        from config import (
            RANDOM_STATE, SFREQ, STIMULUS_BASELINE_WINDOW,
            STIMULUS_RESPONSE_WINDOW, SERI_LOW_MAX, SERI_MODERATE_MAX,
        )
        blob = json.dumps({
            "seed": RANDOM_STATE,
            "sfreq": SFREQ,
            "bl_window": list(STIMULUS_BASELINE_WINDOW),
            "resp_window": list(STIMULUS_RESPONSE_WINDOW),
            "seri_low": SERI_LOW_MAX,
            "seri_mod": SERI_MODERATE_MAX,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def build_run_manifest(
    seed: int,
    data_paths: list[str],
    args: dict[str, Any],
    model_version: str = "stimulus_response_hgb_v1",
) -> RunManifest:
    """Build a deterministic run manifest."""
    return RunManifest(
        seed=seed,
        python_version=platform.python_version(),
        git_hash=_get_git_hash(),
        model_version=model_version,
        data_paths=data_paths,
        args=args,
        config_hash=_config_hash(),
    )


def generate_clinical_report(
    output_dir: Path,
    *,
    validation_metrics: dict[str, Any] | None = None,
    calibration_report: dict[str, Any] | None = None,
    drift_report: dict[str, Any] | None = None,
    prospective_metrics: dict[str, Any] | None = None,
    run_manifest: RunManifest | None = None,
    cohort_info: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate clinical evidence report in Markdown and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    json_report: dict[str, Any] = {}

    # Header
    sections.append("# Clinical Evidence Report — NFM Stimulus-Evoked Response Monitor")
    sections.append("")
    sections.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    sections.append("**Disclaimer:** Decision-support only, not a diagnosis. "
                     "This is a stimulus-evoked response indicator measuring "
                     "functional cortical responsiveness to external stimuli.")
    sections.append("")

    # 1. Data sources & cohort
    sections.append("## 1. Data Sources and Cohort")
    if cohort_info:
        sections.append(f"- **Dataset:** {cohort_info.get('dataset', 'LIS BrainVision')}")
        sections.append(f"- **Subjects:** {cohort_info.get('n_subjects', 'N/A')}")
        sections.append(f"- **Sessions:** {cohort_info.get('n_sessions', 'N/A')}")
        sections.append(f"- **Trials:** {cohort_info.get('n_trials', 'N/A')}")
        json_report["cohort"] = cohort_info
    else:
        sections.append("No cohort information available.")
    sections.append("")

    # 2. Locked model version
    sections.append("## 2. Model Version and Configuration")
    if run_manifest:
        sections.append(f"- **Model:** {run_manifest.model_version}")
        sections.append(f"- **Config hash:** `{run_manifest.config_hash}`")
        sections.append(f"- **Seed:** {run_manifest.seed}")
        sections.append(f"- **Python:** {run_manifest.python_version}")
        sections.append(f"- **Git hash:** {run_manifest.git_hash or 'N/A'}")
        json_report["run_manifest"] = json.loads(run_manifest.model_dump_json())
    else:
        sections.append("No run manifest available.")
    sections.append("")

    # 3. Validation design
    sections.append("## 3. Validation Design")
    if validation_metrics:
        split_type = validation_metrics.get("split_type", "N/A")
        sections.append(f"- **Split strategy:** {split_type}")
        sections.append(f"- **AUC:** {validation_metrics.get('auc', 'N/A')}")
        sections.append(f"- **PR-AUC:** {validation_metrics.get('pr_auc', 'N/A')}")
        sections.append(f"- **Sensitivity:** {validation_metrics.get('sensitivity', 'N/A')}")
        sections.append(f"- **Specificity:** {validation_metrics.get('specificity', 'N/A')}")
        sections.append(f"- **PPV:** {validation_metrics.get('ppv', 'N/A')}")
        sections.append(f"- **NPV:** {validation_metrics.get('npv', 'N/A')}")
        sections.append(f"- **N (test):** {validation_metrics.get('n_samples', 'N/A')}")
        json_report["validation"] = validation_metrics
    else:
        sections.append("No external validation metrics available. "
                        "External validation requires clinical reference labels (CRS-R).")
    sections.append("")

    # 4. Performance + calibration
    sections.append("## 4. Calibration")
    if calibration_report and "error" not in calibration_report:
        overall = calibration_report.get("overall", {})
        sections.append(f"- **ECE:** {overall.get('ece', 'N/A')}")
        sections.append(f"- **MCE:** {overall.get('mce', 'N/A')}")
        sections.append(f"- **Brier:** {overall.get('brier', 'N/A')}")
        interp = calibration_report.get("interpretation", "")
        if interp:
            sections.append(f"- **Interpretation:** {interp}")
        json_report["calibration"] = calibration_report
    else:
        sections.append("Calibration analysis requires clinical reference labels.")
    sections.append("")

    # 5. Drift status
    sections.append("## 5. Drift Status")
    if drift_report:
        alert = drift_report.get("alert_level", "N/A")
        if isinstance(alert, dict):
            alert = alert.get("value", alert) if isinstance(alert, dict) else alert
        sections.append(f"- **Alert level:** {alert}")
        sections.append(f"- **Calibration drift:** {drift_report.get('calibration_shift', 'N/A')}")
        details = drift_report.get("details", "")
        if details:
            sections.append(f"- **Details:** {details}")
        json_report["drift"] = drift_report
    else:
        sections.append("No drift baseline available for comparison.")
    sections.append("")

    # 6. Operational readiness
    sections.append("## 6. Operational Readiness (Prospective)")
    if prospective_metrics:
        sections.append(f"- **Sessions evaluated:** {prospective_metrics.get('n_sessions', 0)}")
        sections.append(f"- **Valid sessions:** {prospective_metrics.get('n_valid', 0)}")
        sections.append(f"- **Invalid rate:** {prospective_metrics.get('invalid_rate', 0):.2%}")
        sections.append(f"- **Failure rate:** {prospective_metrics.get('failure_rate', 0):.2%}")
        sections.append(f"- **Repeatability (ICC proxy):** {prospective_metrics.get('repeatability_icc_proxy', 0):.3f}")
        sections.append(f"- **Processing latency:** {prospective_metrics.get('latency_s', 0):.4f}s")
        json_report["prospective"] = prospective_metrics
    else:
        sections.append("No prospective workflow data available.")
    sections.append("")

    # 7. Limitations
    sections.append("## 7. Known Limitations and Next Steps")
    sections.append("1. **Research proof-of-concept only** — not validated for clinical deployment.")
    sections.append("2. **Small dataset** — external validation is limited by cohort size.")
    sections.append("3. **Reference labels** — CRS-R template provides simulated data; "
                     "real multi-rater CRS-R assessments are needed.")
    sections.append("4. **Single-site data** — multi-center validation is required.")
    sections.append("5. **No regulatory submission** — this tool does not meet device requirements.")
    sections.append("")
    sections.append("### Recommended Next Steps")
    sections.append("1. Collect multi-center CRS-R assessments with ≥2 raters per session.")
    sections.append("2. Run prospective silent-mode study for ≥30 days.")
    sections.append("3. Perform formal sample-size calculation for pivotal validation.")
    sections.append("")
    sections.append("---")
    sections.append("*This report is auto-generated. Decision-support only, not a diagnosis.*")

    # Write Markdown
    md_path = output_dir / "clinical_evidence_report.md"
    md_path.write_text("\n".join(sections), encoding="utf-8")

    # Write JSON
    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

    json_path = output_dir / "clinical_evidence_report.json"
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()
    json_report["disclaimer"] = ("Decision-support only, not a diagnosis. "
                                  "Stimulus-evoked response indicator for "
                                  "functional cortical responsiveness.")
    json_path.write_text(json.dumps(json_report, indent=2, default=_default), encoding="utf-8")

    logger.info("Generated clinical evidence report: %s, %s", md_path, json_path)
    return {"markdown": md_path, "json": json_path}
