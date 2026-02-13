You are an expert Streamlit + EEG clinical dashboard engineer + ML pipeline engineer.

Project root: C:\Users\omar\almaha

==================================================
CONTEXT — WHAT THIS PROJECT IS
==================================================
"Neuro Functional Monitor" (NFM) — a clinical decision-support dashboard built on Streamlit that:
1. Processes EEG from PhysioNet Motor Movement/Imagery dataset (109 subjects, S001–S109)
2. Runs two analysis modes:
   - PASSIVE mode: computes FRS (Functional Responsiveness Score) from ERPs, spectral features, connectivity, ML classifier
   - ACTIVE mode: command-following paradigm (motor imagery detection), computes CFI, Confidence, QualityScore
3. Dashboard (`streamlit run app.py`) displays results for clinicians

Dataset is fully local at:
  C:\Users\omar\almaha\eeg-motor-movementimagery-dataset-1.0.0\files\S001..S109
  AND C:\Users\omar\almaha\files\S001..S109

Model outputs go to:
  C:\Users\omar\almaha\outputs\results\       (CSVs: frs_results.csv, features_all_subjects.csv, subject_summary.csv, command_following_trials.csv)
  C:\Users\omar\almaha\outputs\session_store\  (JSON session files per subject)
  C:\Users\omar\almaha\outputs\models\         (trained models)
  C:\Users\omar\almaha\outputs\results_command_following.csv

==================================================
CURRENT STATE / WHAT HAS BEEN DONE
==================================================
- Active mode has been run successfully for all 109 subjects
- Passive mode has old/stale data — Model Confidence showing ~42% for S001 which may be wrong
- Dashboard has been partially fixed for crash bugs (width="stretch" → use_container_width, widget keys, try/except guards)
- Clinical explanation text was blank (markdown inside HTML div); fixed by converting **bold** → <strong> and \n\n → <br>
- Refresh Data button added to sidebar

==================================================
PROBLEMS STILL TO SOLVE (YOUR TASKS)
==================================================

TASK 1: RE-RUN FULL PIPELINE TO REFRESH ALL DATA
─────────────────────────────────────────────────
The outputs/ directory has stale data from old pipeline runs. You must:

A) Re-run PASSIVE mode for all 109 subjects:
   python main.py --mode passive --skip-download --all-local-subjects --skip-ica --n-jobs 15

B) Re-run ACTIVE mode for all 109 subjects:
   python main.py --mode active --skip-download --all-local-subjects --active-feature-profile full --skip-ica --n-jobs 15

C) Verify outputs exist and are complete:
   - outputs/results/frs_results.csv should have rows for subjects 1–109
   - outputs/results/features_all_subjects.csv should exist
   - outputs/results/subject_summary.csv should exist
   - outputs/results_command_following.csv should have rows for 109 subjects
   - outputs/session_store/ should have session_s*.json for 109 subjects

D) If any subject fails during pipeline, log it but continue with remaining subjects (the pipeline has --resume support)

TASK 2: FIX SUBJECT SWITCHING RENDERING ISSUES
─────────────────────────────────────────────────
When switching patients in the Streamlit sidebar, the dashboard sometimes:
- Shows a blank/dark screen (frontend crash)
- Doesn't update the charts for the new subject
- Shows stale data from previous subject

Root causes to check and fix:

A) @st.cache_data on load_dashboard_data() — this is correct for raw CSV loading BUT:
   - Verify cached data doesn't hold stale references
   - Subject-specific filtering must always use .copy() to avoid cross-contamination
   - All widget keys must include subject_id to prevent Streamlit component key collisions

B) Plotly chart keys — every st.plotly_chart() call needs a unique key that changes with subject:
   - key=f"frs_gauge_{subject_id}"  
   - key=f"trend_chart_{subject_id}"
   - key=f"cfi_trend_{selected_subject}"
   - key=f"conf_breakdown_{selected_subject}"
   Check ALL dashboard/*.py files for any st.plotly_chart() or st.dataframe() calls missing dynamic keys.

C) Widget key conflicts — st.selectbox, st.slider, st.download_button all need stable unique keys.
   The patient_table.py has keys like "pt_cat_filter" and "pt_frs_filter" which are fine (they're global filters).
   But if any widget key collides during tab switching, Streamlit throws DuplicateWidgetID and shows blank screen.

D) DataFrame safety:
   - Every .iloc[-1] must be guarded with an emptiness check
   - Every float() conversion from DataFrame values must handle NaN: `float(x) if pd.notna(x) else 0.0`
   - Every filtered DataFrame should use .copy() to prevent SettingWithCopyWarning

TASK 3: FIX THE MODEL CONFIDENCE DISPLAY
─────────────────────────────────────────
In the dashboard screenshot, Model Confidence shows 42% for S001. This comes from:
  frs_df["norm_model_probability"].mean()

This might be correct (if the classifier probability for S001 is genuinely 0.42), OR it might be stale.
After re-running the pipeline (Task 1), verify:
- Check outputs/results/frs_results.csv for subject_id=1, column norm_model_probability
- The value should reflect the actual trained model's prediction probability
- If the model needs retraining, the passive pipeline handles it automatically

TASK 4: ENSURE CLINICAL EXPLANATION BOX RENDERS PROPERLY
────────────────────────────────────────────────────────
The clinical explanation box (below the KPI metrics in the Passive FRS tab) was blank.
Fixed by converting markdown to HTML in interpretation_engine.py's generate_clinical_explanation().
Verify:
- The explanation text appears with proper bold text, line breaks
- The text color is dark (#212529) on the light gray (#f8f9fa) background
- The explanation updates correctly when switching subjects

TASK 5: FULL VALIDATION
───────────────────────
After all fixes:

A) Run all tests:
   python -m pytest tests/ -q --tb=short
   All must pass, including test_dashboard_smoke.py (72 parametrized tests)

B) Launch dashboard and test subject switching:
   streamlit run app.py
   - Switch rapidly between subjects: 1, 10, 20, 50, 90, 109
   - Verify no blank screens
   - Verify charts update for each subject
   - Both Passive FRS and Command Following tabs must work
   - Clinical explanation box must show text (not blank)

C) Verify data freshness:
   - After pipeline re-run, click "Refresh Data" button in sidebar
   - Values should reflect the latest pipeline outputs

==================================================
RELEVANT FILES
==================================================
Main pipeline:
  main.py                          — Entry point (--mode passive/active)
  config.py                        — All paths, constants, hyperparameters

Dashboard:
  app.py                           — Root wrapper (imports dashboard.app)
  dashboard/app.py                 — Main Streamlit app, data loading, tab routing
  dashboard/clinical_summary.py    — FRS gauge, KPIs, clinical explanation
  dashboard/command_following.py   — Command following tab (CFI, confidence, trend)
  dashboard/trend_analysis.py      — FRS trend charts
  dashboard/patient_table.py       — Full cohort table
  dashboard/export_module.py       — CSV/JSON export buttons
  dashboard/interpretation_engine.py — Clinical text generation

Scoring/Models:
  scoring/cfi.py                   — CFI evidence labels, confidence/quality labels
  scoring/frs.py                   — FRS computation
  models/                          — Random forest, deep learning, evaluation

Tests:
  tests/test_dashboard_smoke.py    — 72 parametrized smoke tests for subject switching
  tests/test_*.py                  — Other existing tests

==================================================
ARCHITECTURE CONSTRAINTS
==================================================
- Python 3.12, Streamlit (latest), Plotly, Pandas, scikit-learn, MNE
- Dataset: PhysioNet EEG Motor Movement/Imagery, 109 subjects, 64 EEG channels, 160 Hz
- Must use --skip-ica flag (ICA is slow and not essential for this dataset)
- Must use --skip-download (data is already local)
- Must use --all-local-subjects to process all available subjects
- Pipeline supports --resume to continue from where it left off
- Never claim diagnosis or consciousness measurement in UI text

==================================================
NON-NEGOTIABLE CLINICAL LANGUAGE RULE
==================================================
Never claim diagnosis or consciousness measurement. Use:
- "command-following indicator"
- "functional responsiveness to mental commands"
- "decision-support only, not a diagnosis"

==================================================
EXECUTION ORDER
==================================================
1. First, run passive pipeline for all 109 subjects
2. Then, run active pipeline for all 109 subjects  
3. Then, fix any remaining dashboard rendering bugs
4. Then, run pytest to verify all tests pass
5. Then, launch streamlit run app.py and verify visually
6. Report results

==================================================
OUTPUT FORMAT
==================================================
Report:
1. Pipeline run results (how many subjects succeeded, any failures)
2. Files changed with reason
3. Test results (all must pass)
4. Dashboard verification results
5. Any remaining issues
