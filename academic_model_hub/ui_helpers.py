"""UI helper utilities for academic model hub pages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACADEMIC_STATUS_COLORS: dict[str, str] = {
    "ok": "#1B5E20",
    "success": "#1B5E20",
    "degraded": "#8A6D00",
    "warning": "#8A6D00",
    "error": "#8B0000",
    "failed": "#8B0000",
    "unknown": "#37474F",
}

ERROR_HINTS: dict[str, dict[str, str]] = {
    "INVALID_INPUT_SCHEMA": {
        "category": "Input issue",
        "hint": "Review required fields and file paths before resubmitting.",
        "fixability": "user-fixable",
    },
    "MISSING_WEIGHTS": {
        "category": "Runtime setup",
        "hint": "Mount model checkpoints and confirm payload paths point to readable files.",
        "fixability": "infra/runtime",
    },
    "INCOMPATIBLE_RUNTIME": {
        "category": "Infrastructure",
        "hint": "Switch runtime mode or fix Python/Torch/Docker/repository configuration.",
        "fixability": "infra/runtime",
    },
    "GPU_UNAVAILABLE": {
        "category": "Infrastructure",
        "hint": "Select CPU-compatible settings or run on a host with CUDA configured.",
        "fixability": "infra/runtime",
    },
    "UNSUPPORTED_TASK": {
        "category": "Input issue",
        "hint": "Choose one of the supported tasks listed on the model card.",
        "fixability": "user-fixable",
    },
    "MALFORMED_BIOLOGICAL_INPUT": {
        "category": "Input issue",
        "hint": "Check sequence alphabet, PAM format, and biological field syntax.",
        "fixability": "user-fixable",
    },
    "UPSTREAM_SCRIPT_FAILURE": {
        "category": "Execution issue",
        "hint": "Check adapter runtime health and upstream script compatibility for the selected commit.",
        "fixability": "infra/runtime",
    },
    "OUTPUT_PARSE_FAILURE": {
        "category": "Execution issue",
        "hint": "The model ran but output format was unexpected; review upstream version and output files.",
        "fixability": "infra/runtime",
    },
    "UPSTREAM_TIMEOUT": {
        "category": "Execution issue",
        "hint": "Increase timeout only after confirming backend health and model runtime readiness.",
        "fixability": "infra/runtime",
    },
    "BACKEND_UNREACHABLE": {
        "category": "Infrastructure",
        "hint": "Confirm BACKEND_API_URL and ensure the backend service is reachable from Streamlit host.",
        "fixability": "infra/runtime",
    },
    "ACADEMIC_HTTP_ERROR": {
        "category": "Execution issue",
        "hint": "Review backend status and request payload; retry only after correcting actionable issues.",
        "fixability": "depends",
    },
}


def error_hint(error_code: str) -> dict[str, str]:
    return ERROR_HINTS.get(
        error_code,
        {
            "category": "Unknown",
            "hint": "Review the error payload details and retry after validating runtime configuration.",
            "fixability": "depends",
        },
    )


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    val = color.lstrip("#")
    if len(val) != 6:
        raise ValueError(f"Invalid hex color: {color}")
    return (int(val[0:2], 16) / 255.0, int(val[2:4], 16) / 255.0, int(val[4:6], 16) / 255.0)


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    def _linearize(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    ar, ag, ab = hex_to_rgb(hex_a)
    br, bg, bb = hex_to_rgb(hex_b)
    l1 = 0.2126 * _linearize(ar) + 0.7152 * _linearize(ag) + 0.0722 * _linearize(ab)
    l2 = 0.2126 * _linearize(br) + 0.7152 * _linearize(bg) + 0.0722 * _linearize(bb)
    light = max(l1, l2)
    dark = min(l1, l2)
    return (light + 0.05) / (dark + 0.05)


def status_color(status: str) -> str:
    key = str(status or "unknown").lower().strip()
    return ACADEMIC_STATUS_COLORS.get(key, ACADEMIC_STATUS_COLORS["unknown"])


def push_run_history(history: list[dict[str, Any]], result: dict[str, Any], model_name: str, runtime_mode: str) -> list[dict[str, Any]]:
    provenance = result.get("provenance", {}) if isinstance(result, dict) else {}
    client_meta = result.get("_client_meta", {}) if isinstance(result, dict) else {}
    item = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "status": result.get("status"),
        "request_hash": provenance.get("request_hash", ""),
        "request_id": provenance.get("request_id") or client_meta.get("request_id", ""),
        "runtime_mode": runtime_mode,
        "result": result,
    }
    return [item] + history[:29]


def available_downloads(result: dict[str, Any]) -> list[tuple[str, str]]:
    artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
    outputs: list[tuple[str, str]] = []
    if artifacts.get("provenance_manifest"):
        outputs.append(("Provenance Manifest", str(artifacts["provenance_manifest"])))
    out_dir = artifacts.get("output_dir")
    if out_dir:
        summary_path = Path(str(out_dir)) / "run_summary.json"
        if summary_path.exists():
            outputs.append(("Run Summary", str(summary_path)))
    for key, value in artifacts.items():
        if key in {"output_dir", "provenance_manifest"}:
            continue
        if isinstance(value, str) and (value.endswith(".json") or value.endswith(".csv") or value.endswith(".pdb")):
            outputs.append((f"Artifact: {key}", value))
    return outputs
