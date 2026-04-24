"""File and subprocess helpers used by adapters."""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from academic_model_hub.errors import AcademicModelError, ErrorCode

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_records(path: str | Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_tsv_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(
    command: list[str],
    *,
    cwd: str | None = None,
    timeout_seconds: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    if result.returncode != 0:
        logger.error("Upstream command failed rc=%s cmd=%s stderr=%s", result.returncode, command, result.stderr[:400])
        raise AcademicModelError(
            code=ErrorCode.UPSTREAM_SCRIPT_FAILURE,
            message=f"Upstream command failed: {' '.join(command)}",
            details={"stderr": result.stderr[:1000], "returncode": result.returncode},
            safe_details={"returncode": result.returncode, "stderr_preview": result.stderr[:220]},
        )
    return result


def run_command_allow_timeout(
    command: list[str],
    *,
    cwd: str | None = None,
    timeout_seconds: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_command(command, cwd=cwd, timeout_seconds=timeout_seconds, env=env)
    except subprocess.TimeoutExpired as exc:
        raise AcademicModelError(
            code=ErrorCode.UPSTREAM_SCRIPT_FAILURE,
            message=f"Upstream command timed out after {timeout_seconds}s",
            safe_details={"timeout_seconds": timeout_seconds},
        ) from exc


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


