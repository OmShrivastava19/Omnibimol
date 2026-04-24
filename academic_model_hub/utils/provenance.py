"""Provenance and request hashing utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from academic_model_hub.schemas.common import RuntimeDeclaration


VOLATILE_KEYS = {"output_dir", "temp_dir", "timestamp", "request_id"}


def canonicalize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(payload):
            if key in VOLATILE_KEYS or key.endswith("_dir"):
                continue
            normalized[key] = canonicalize_payload(payload[key])
        return normalized
    if isinstance(payload, list):
        return [canonicalize_payload(item) for item in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def request_hash(model_name: str, payload: dict[str, Any]) -> str:
    material = {"model_name": model_name, "payload": canonicalize_payload(payload)}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    *,
    paper_title: str,
    paper_year: int | None,
    repo_url: str,
    commit_or_release: str,
    runtime: RuntimeDeclaration,
    device: str,
    req_hash: str,
) -> dict[str, Any]:
    return {
        "paper_title": paper_title,
        "paper_year": paper_year,
        "repo_url": repo_url,
        "commit_or_release": commit_or_release,
        "runtime_mode": runtime.mode,
        "python_version": runtime.python_version,
        "torch_version": runtime.torch_version,
        "device": device,
        "request_hash": req_hash,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def write_provenance_manifest(output_dir: str | Path, provenance: dict[str, Any]) -> str:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "provenance_manifest.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return str(path)
