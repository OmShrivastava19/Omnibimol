from __future__ import annotations

from academic_model_hub.ui_helpers import available_downloads, error_hint, push_run_history


def test_error_hint_maps_known_code() -> None:
    hint = error_hint("INVALID_INPUT_SCHEMA")
    assert hint["category"] == "Input issue"
    assert "required fields" in hint["hint"]


def test_push_run_history_prepends_and_caps() -> None:
    history: list[dict] = []
    result = {"status": "success", "provenance": {"request_hash": "abc123", "request_id": "req-1"}}
    updated = push_run_history(history, result, "flexpose", "native")
    assert len(updated) == 1
    assert updated[0]["model"] == "flexpose"
    assert updated[0]["request_hash"] == "abc123"
    assert updated[0]["request_id"] == "req-1"


def test_available_downloads_handles_empty_artifacts() -> None:
    result = {"artifacts": {}}
    assert available_downloads(result) == []
