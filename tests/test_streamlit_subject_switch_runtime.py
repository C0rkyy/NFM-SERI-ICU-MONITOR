from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import OUTPUT_DIR, RESULTS_DIR


@pytest.mark.skipif(not (RESULTS_DIR / "frs_results.csv").exists(), reason="frs_results.csv not found")
def test_streamlit_subject_switch_no_blank_screen() -> None:
    """Subject changes should not produce exceptions/blank render state."""
    app_path = Path("app.py")
    assert app_path.exists(), "app.py missing"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=180)
    assert len(at.exception) == 0, f"Initial render exception(s): {[e.value for e in at.exception]}"
    assert any(t.value == "Neuro Functional Monitor" for t in at.title), "Dashboard title missing on initial render"

    assert len(at.sidebar.selectbox) >= 1, "Subject selector not found"
    selector = at.sidebar.selectbox[0]
    options = list(selector.options)
    assert len(options) >= 2, "Need at least 2 subjects to test switching"

    candidate_indices = sorted({0, min(9, len(options) - 1), min(19, len(options) - 1), len(options) - 1})
    for idx in candidate_indices:
        selector.set_value(options[idx])
        at.run(timeout=180)
        assert len(at.exception) == 0, (
            f"Render exception after switching to option index {idx} ({options[idx]}): "
            f"{[e.value for e in at.exception]}"
        )
        assert any(t.value == "Neuro Functional Monitor" for t in at.title), (
            f"Dashboard title missing after switching to {options[idx]}"
        )


@pytest.mark.skipif(not (OUTPUT_DIR / "results_command_following.csv").exists(), reason="results_command_following.csv not found")
def test_streamlit_subject_selector_accepts_string_option_values() -> None:
    """Regression test for format_func crash when widget state carries string labels."""
    at = AppTest.from_file("app.py")
    at.run(timeout=180)
    assert len(at.exception) == 0

    selector = at.sidebar.selectbox[0]
    options = list(selector.options)
    label_option = next((opt for opt in options if isinstance(opt, str) and "Patient" in opt), options[0])
    selector.set_value(label_option)
    at.run(timeout=180)

    assert len(at.exception) == 0, f"String option value caused rerun failure: {[e.value for e in at.exception]}"
