# Neuro Functional Monitor (NFM)

**Research-grade EEG processing system for stimulus-evoked cortical response monitoring.**

Computes a **Stimulus-Evoked Response Index (SERI, 0–100)** from stimulus-locked EEG recordings, providing quantitative evidence of functional cortical responsiveness to external stimuli.

> **Disclaimer:** This system is a research proof-of-concept and a decision-support tool only. SERI values do not constitute a clinical diagnosis. All outputs must be interpreted by qualified clinicians in context.

---

## Recommended GitHub Repository

- **Repository name:** `nfm-seri-icu-monitor`
- **GitHub short description:** `Stimulus-evoked EEG response monitoring (SERI) for ICU decision-support using LIS BrainVision data. Research PoC, not a diagnosis tool.`
- **Suggested GitHub topics:** `eeg`, `brainvision`, `icu`, `neuro`, `streamlit`, `machine-learning`, `mne-python`, `clinical-ai`, `decision-support`

---

## Overview

NFM processes BrainVision-format EEG data (LIS Zenodo dataset) and/or PhysioNet EDF data through a complete neuroscientific pipeline:

### Stimulus-Evoked Response Pipeline (Primary)

1. **Data Ingestion** — Discover and load BrainVision (.vhdr/.vmrk/.eeg) sessions from the LIS dataset
2. **Event Extraction** — Parse stimulus markers (S1–S15), classify onset/response/block events, read senlist sidecar files
3. **Preprocessing** — Band-pass filter (0.5–40 Hz), 50 Hz notch filter, artefact rejection
4. **Epoch Extraction** — Stimulus-locked epochs: baseline (−200 to 0 ms), response (0 to +800 ms)
5. **Feature Extraction** — ERP components (N100, P200, P300), spectral band power changes (delta/theta/alpha/beta), evoked SNR, trial consistency
6. **Model Training** — HistGradientBoosting classifier with calibrated probabilities, within-subject and across-subject (LOGO) cross-validation
7. **SERI Scoring** — Composite score [0, 100] from response probability, consistency, SNR, and data quality
8. **Dashboard** — Interactive Streamlit app with SERI visualization, trend analysis, and clinical glossary

### Legacy Pipelines (Preserved)

- **Passive FRS Pipeline** — Functional Responsiveness Score from PhysioNet motor imagery data
- **Active Command Following** — CFI-based command-following intent detection

---

## Project Structure

```
almaha/
├── config.py                    # Global parameters, paths, SERI thresholds
├── main.py                      # End-to-end pipeline orchestrator (passive/active/stimulus)
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── brainvision_loader.py    # BrainVision session discovery and loading
│   ├── downloader.py            # PhysioNet EDF downloader
│   └── loader.py                # EDF reader, event extraction
│
├── preprocessing/
│   ├── pipeline.py              # Band-pass, notch, ICA, re-ref, z-score
│   ├── epochs.py                # Stimulus-locked epoch extraction (PhysioNet)
│   └── stimulus_epochs.py       # Stimulus-locked epoching (BrainVision/LIS)
│
├── stimulus/
│   ├── event_extraction.py      # Parse/classify BrainVision stimulus markers
│   ├── protocols.py             # Protocol definitions
│   └── session_plan.py          # Session planning
│
├── features/
│   ├── stimulus_features.py     # ERP components, band power, SNR, consistency
│   ├── erp_features.py          # P300 amplitude, latency, AUC
│   ├── spectral_features.py     # PSD, band power, relative changes
│   ├── motor_imagery_features.py# Motor imagery features (legacy)
│   ├── time_frequency.py        # Morlet wavelet TFR
│   ├── connectivity.py          # Pairwise coherence
│   └── feature_store.py         # Merge & persist feature DataFrames
│
├── models/
│   ├── stimulus_response.py     # Stimulus response detection model
│   ├── command_following.py     # Command-following classifier (legacy)
│   ├── random_forest.py         # RF classifier + cross-validation
│   ├── deep_learning.py         # 1-D CNN (TensorFlow/Keras)
│   └── evaluation.py            # Metrics, ROC plotting
│
├── scoring/
│   ├── seri.py                  # SERI formula, evidence/confidence labels
│   ├── cfi.py                   # CFI scoring (legacy)
│   ├── frs.py                   # FRS formula (legacy)
│   ├── quality.py               # Data quality assessment
│   └── sensitivity.py           # Weight perturbation analysis
│
├── schemas/
│   └── session.py               # Pydantic models (supports SERI + CFI)
│
├── statistics/
│   └── group_comparison.py      # t-test, Cohen's d, group ROC
│
├── dashboard/
│   ├── app.py                   # Streamlit main app (3 tabs)
│   ├── stimulus_response.py     # Stimulus Response tab (SERI)
│   ├── command_following.py     # Command Following tab (CFI)
│   ├── clinical_summary.py      # Clinical summary panels
│   ├── interpretation_engine.py # Evidence interpretation logic
│   └── trend_analysis.py        # Longitudinal trend charts
│
├── tests/
│   ├── test_seri.py             # SERI bounds, labels, monotonicity
│   ├── test_stimulus_pipeline.py# Ingestion, epochs, features, schema tests
│   ├── test_cfi.py              # CFI scoring tests (legacy)
│   └── test_dashboard_smoke.py  # Dashboard import smoke tests
│
└── outputs/
    ├── models/                  # Saved model artefacts (.pkl)
    ├── results/                 # CSV, JSON, PNG artefacts
    ├── figures/                 # SERI summary and diagnostic charts
    └── session_store/           # Per-session JSON records
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Place the LIS BrainVision dataset under:
```
data/external/lis_zenodo/Raw_Files/
    P11/V01/D01/Training/...
    P13/V01/D01/Training/...
    P15/...
    P16/...
```

Each session folder contains `.vhdr`, `.vmrk`, `.eeg` triplets and optional `Block_x_senlist.txt` sidecar files.

### 3. Run the stimulus pipeline

```bash
python main.py --mode stimulus --skip-ica --n-jobs 4
```

This will:
- Discover all BrainVision sessions in the LIS dataset
- Preprocess, extract stimulus-locked features, train response model
- Compute SERI scores per session
- Save results to `outputs/`

### 4. Run the full pipeline (all modes)

```bash
python main.py --mode both --n-jobs 4
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

### 6. Run tests

```bash
python -m pytest tests/ -q --tb=short
```

---

## CLI Options

| Flag | Description |
|---|---|
| `--mode {passive,active,stimulus,both}` | Pipeline mode (default: `passive`; use `stimulus` for primary workflow) |
| `--stimulus-dataset-root PATH` | Override LIS dataset root directory |
| `--subjects 1 2 3` | Process only specified subjects (passive/active) |
| `--n-jobs N` | Number of parallel workers |
| `--skip-ica` | Skip ICA step |
| `--skip-cnn` | Skip CNN training |
| `--skip-active` | Skip active pipeline even in `both` mode |

---

## SERI Formula

$$
\text{SERI}_{\text{raw}} = 0.50 \cdot S_{\text{base}} + 0.20 \cdot S_{\text{consistency}} + 0.15 \cdot S_{\text{SNR}} + 0.15 \cdot S_{\text{quality}}
$$

$$
\text{SERI} = \text{clamp}(100 \times \text{SERI}_{\text{raw}}, \ 0, \ 100)
$$

| Component | Weight | Description |
|---|---|---|
| Base signal ($S_\text{base}$) | 0.50 | Mean response probability from calibrated model |
| Consistency ($S_\text{consistency}$) | 0.20 | Trial-to-trial response consistency ($1 - \text{CV}$) |
| SNR ($S_\text{SNR}$) | 0.15 | Evoked signal-to-noise ratio (saturates via tanh) |
| Quality ($S_\text{quality}$) | 0.15 | Data quality score (epoch rejection rate, bad channels) |

### Evidence Bands

| SERI Range | Label | Interpretation |
|---|---|---|
| 0–39 | **Low** | Little evidence of stimulus-evoked cortical response |
| 40–69 | **Moderate** | Some evidence of stimulus-evoked cortical response |
| 70–100 | **High** | Strong evidence of stimulus-evoked cortical response |

### Confidence Score

Confidence (0–100) reflects data adequacy and model calibration, computed from:
- Response probability sharpness (15%)
- Trial consistency (20%)
- Trial count sufficiency (15%)
- Data quality (15%)
- Model reliability / calibration (20%)
- Bootstrap CI tightness (15%)

---

## Outputs

| File | Contents |
|---|---|
| `results_stimulus_response.csv` | Per-session SERI, Confidence, Quality, Evidence, trial counts |
| `stimulus_response_trials.csv` | Per-trial features with response probabilities |
| `stimulus_response_pipeline_metrics.json` | Runtime metrics, model AUC, session counts |
| `session_store/*.json` | Individual session Pydantic records |
| `figures/seri_summary.png` | SERI bar chart across sessions |
| `frs_results.csv` | Per-subject FRS (legacy passive pipeline) |
| `results_command_following.csv` | Per-subject CFI (legacy active pipeline) |

---

## Dataset

### Primary: LIS BrainVision Dataset (Zenodo)

- 4 subjects (P11, P13, P15, P16), multiple visits/days/sessions
- 10 channels: R1, R2, L1, L2, Cz, C1, C2, EOGD, EOGL, EOGR (500 Hz)
- Stimulus markers: S1–S15 (auditory stimuli with binary response labels)
- BrainVision format: `.vhdr` (header), `.vmrk` (markers), `.eeg` (binary)

### Legacy: PhysioNet EEG Motor Movement / Imagery Dataset

- 109 subjects, 64-channel EEG, 160 Hz
- Runs 3, 7, 11: motor-imagery events (T0=rest, T1=left, T2=right)

---

## License

Research prototype — not for clinical diagnosis. Decision-support only.

---

## Publish to GitHub

If this folder is not a Git repository yet:

```bash
cd c:\Users\omar\almaha
git init
git branch -M main
git add .
git commit -m "Initial commit: LIS stimulus-evoked SERI pipeline and dashboard"
```

Create a new empty repository on GitHub named `nfm-seri-icu-monitor`, then connect and push:

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/nfm-seri-icu-monitor.git
git push -u origin main
```

If a remote already exists:

```bash
git remote -v
git push
```

Optional (GitHub CLI):

```bash
gh repo create nfm-seri-icu-monitor --public --source . --remote origin --push --description "Stimulus-evoked EEG response monitoring (SERI) for ICU decision-support using LIS BrainVision data. Research PoC, not a diagnosis tool."
```
