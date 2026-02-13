"""
NFM — Automated Clinical Interpretation Engine
=================================================
Rule-based natural-language explanation generator that converts
raw numeric features into plain-English clinical summaries
suitable for ICU physicians with no EEG expertise.

Every function is pure (no side-effects) and returns strings
that can be rendered directly in the dashboard.
"""

from __future__ import annotations
import re
from typing import Dict, Optional


# ───────────────────────────────────────────────────────────────
# Clinical category thresholds
# ───────────────────────────────────────────────────────────────
CATEGORIES = [
    (0,  20, "Minimal Response",  "#d32f2f",  "red"),
    (21, 40, "Weak Response",     "#f57c00",  "orange"),
    (41, 70, "Moderate Response", "#fbc02d",  "yellow"),
    (71, 100,"Strong Response",   "#388e3c",  "green"),
]


def clinical_category(frs: float) -> Dict[str, str]:
    """
    Map an FRS value to its clinical category.

    Returns dict with keys: label, color_hex, color_name, range_str
    """
    for lo, hi, label, hex_color, name in CATEGORIES:
        if lo <= frs <= hi:
            return {
                "label": label,
                "color_hex": hex_color,
                "color_name": name,
                "range_str": f"{lo}–{hi}",
            }
    return {"label": "Unknown", "color_hex": "#9e9e9e",
            "color_name": "gray", "range_str": "—"}


# ───────────────────────────────────────────────────────────────
# Sub-component interpretation helpers
# ───────────────────────────────────────────────────────────────
def _erp_text(norm_erp: float) -> str:
    """Plain-language description of ERP amplitude finding."""
    if norm_erp >= 0.70:
        return ("Stimulus-evoked cortical potentials are **large and clearly "
                "defined**, indicating robust cortical processing of the stimulus.")
    elif norm_erp >= 0.40:
        return ("Stimulus-evoked cortical potentials are **present at moderate "
                "amplitude**, suggesting preserved but attenuated cortical processing.")
    elif norm_erp >= 0.15:
        return ("Stimulus-evoked cortical potentials are **detectable but "
                "reduced**, indicating minimal residual cortical activation.")
    else:
        return ("Stimulus-evoked cortical potentials are **very low or absent**, "
                "suggesting markedly diminished cortical responsiveness to the stimulus.")


def _spectral_text(norm_psd: float) -> str:
    """Plain-language description of spectral power shift."""
    if norm_psd >= 0.60:
        return ("Brain-wave frequency patterns show a **significant shift** in "
                "response to stimulation, consistent with active neural engagement.")
    elif norm_psd >= 0.30:
        return ("Brain-wave frequency patterns show a **moderate shift**, "
                "suggesting partial neural engagement.")
    else:
        return ("Brain-wave frequency patterns show **little to no change** "
                "from baseline, indicating limited neural modulation.")


def _model_text(prob: float) -> str:
    """Plain-language interpretation of classifier probability."""
    if prob >= 0.70:
        return ("The multivariate pattern classifier identifies this recording "
                "as **highly consistent** with functional responsiveness (probability "
                f"{prob:.0%}).")
    elif prob >= 0.40:
        return ("The multivariate pattern classifier identifies this recording "
                "as **partially consistent** with functional responsiveness "
                f"(probability {prob:.0%}).")
    else:
        return ("The multivariate pattern classifier indicates **low likelihood** "
                f"of functional responsiveness (probability {prob:.0%}).")


def _connectivity_text(norm_gci: float) -> str:
    """Plain-language description of connectivity index."""
    if norm_gci >= 0.60:
        return ("Inter-regional brain connectivity is **preserved**, suggesting "
                "intact large-scale cortical networks.")
    elif norm_gci >= 0.30:
        return ("Inter-regional brain connectivity is **partially preserved**, "
                "with some network degradation.")
    else:
        return ("Inter-regional brain connectivity is **reduced**, "
                "suggesting disrupted cortical network communication.")


# ───────────────────────────────────────────────────────────────
# Public API — full clinical explanation
# ───────────────────────────────────────────────────────────────
def generate_clinical_explanation(
    frs: float,
    norm_erp: float = 0.5,
    norm_psd: float = 0.5,
    model_prob: float = 0.5,
    norm_gci: float = 0.5,
) -> str:
    """
    Generate a multi-sentence clinical summary suitable for display
    in the main dashboard panel.

    Parameters
    ----------
    frs        : Functional Responsiveness Score (0–100)
    norm_erp   : normalised ERP amplitude        (0–1)
    norm_psd   : normalised band-power shift      (0–1)
    model_prob : classifier probability           (0–1)
    norm_gci   : normalised connectivity index    (0–1)

    Returns
    -------
    str — Markdown-formatted paragraph.
    """
    cat = clinical_category(frs)

    # Headline sentence
    if frs >= 71:
        headline = (
            f"**Assessment: {cat['label']}** — Stimulus-evoked cortical "
            f"activity is clearly present and functionally significant."
        )
    elif frs >= 41:
        headline = (
            f"**Assessment: {cat['label']}** — Stimulus-evoked cortical "
            f"activity is detectable, suggesting preserved but reduced "
            f"functional responsiveness."
        )
    elif frs >= 21:
        headline = (
            f"**Assessment: {cat['label']}** — Limited stimulus-evoked "
            f"cortical activity detected. Functional responsiveness is "
            f"present but substantially diminished."
        )
    else:
        headline = (
            f"**Assessment: {cat['label']}** — No significant "
            f"stimulus-evoked cortical activity detected. Functional "
            f"responsiveness is minimal or absent."
        )

    # Component sentences
    parts = [
        headline,
        "",
        _erp_text(norm_erp),
        _spectral_text(norm_psd),
        _model_text(model_prob),
        _connectivity_text(norm_gci),
    ]

    text = "\n\n".join(parts)
    return _md_to_html(text)


def _md_to_html(text: str) -> str:
    """Convert basic markdown bold/italic to HTML for use inside raw HTML divs."""
    # **bold** -> <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # *italic* -> <em>italic</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # \n\n -> <br><br> for paragraph breaks
    text = text.replace("\n\n", "<br><br>")
    # single \n -> <br>
    text = text.replace("\n", "<br>")
    return text


def generate_short_summary(frs: float) -> str:
    """One-liner for table cells and compact views."""
    cat = clinical_category(frs)
    return f"{cat['label']} (FRS {frs:.0f})"


# ───────────────────────────────────────────────────────────────
# Trend interpretation
# ───────────────────────────────────────────────────────────────
def trend_label(pct_change: float) -> Dict[str, str]:
    """
    Classify a percentage change between sessions.

    Returns dict: label, arrow, color
    """
    if pct_change > 10:
        return {"label": "Improving", "arrow": "↑", "color": "#388e3c"}
    elif pct_change < -10:
        return {"label": "Declining", "arrow": "↓", "color": "#d32f2f"}
    else:
        return {"label": "Stable", "arrow": "→", "color": "#1565c0"}
