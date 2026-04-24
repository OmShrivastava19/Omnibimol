from __future__ import annotations

from academic_model_hub.ui_helpers import ACADEMIC_STATUS_COLORS, contrast_ratio, error_hint, status_color


def test_status_colors_meet_wcag_aa_against_white_text() -> None:
    for _, color in ACADEMIC_STATUS_COLORS.items():
        assert contrast_ratio(color, "#FFFFFF") >= 4.5


def test_error_hints_expose_fixability() -> None:
    for code in ["INVALID_INPUT_SCHEMA", "MISSING_WEIGHTS", "UPSTREAM_SCRIPT_FAILURE", "BACKEND_UNREACHABLE"]:
        hint = error_hint(code)
        assert hint["fixability"] in {"user-fixable", "infra/runtime", "depends"}


def test_status_color_defaults_to_unknown() -> None:
    assert status_color("nonsense") == ACADEMIC_STATUS_COLORS["unknown"]
