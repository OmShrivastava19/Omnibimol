"""Runtime compatibility helpers for adapter execution modes."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from academic_model_hub.errors import AcademicModelError, ErrorCode

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class RuntimeCheckResult:
    mode: str
    python_ok: bool
    torch_ok: bool
    gpu_ok: bool
    repo_ok: bool
    details: dict[str, Any]


def parse_min_python(raw: str) -> tuple[int, int]:
    cleaned = raw.replace("+", "").strip()
    major, minor, *_ = cleaned.split(".")
    return int(major), int(minor)


def require_repo_path(repo_path: str | None, *, var_name: str) -> Path:
    if not repo_path:
        raise AcademicModelError(
            code=ErrorCode.INCOMPATIBLE_RUNTIME,
            message=f"Missing upstream repository path for {var_name}",
            details={"env_var": var_name},
            safe_details={"env_var": var_name},
        )
    path = Path(repo_path)
    if not path.exists():
        raise AcademicModelError(
            code=ErrorCode.INCOMPATIBLE_RUNTIME,
            message=f"Configured upstream repository path not found for {var_name}",
            details={"env_var": var_name, "path": repo_path},
            safe_details={"env_var": var_name},
        )
    return path


def check_runtime(
    *,
    mode: str,
    min_python: str,
    torch_version: str | None,
    gpu_required: bool,
    repo_path: str | None,
    repo_env_var: str,
) -> RuntimeCheckResult:
    min_major, min_minor = parse_min_python(min_python)
    py_ok = sys.version_info >= (min_major, min_minor)
    torch_ok = True
    gpu_ok = True
    details: dict[str, Any] = {
        "python_current": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_required": min_python,
        "mode": mode,
    }
    if torch_version:
        torch_ok = bool(torch is not None)
        details["torch_required"] = torch_version
        details["torch_current"] = getattr(torch, "__version__", None)
    if gpu_required:
        gpu_ok = bool(torch is not None and torch.cuda.is_available())
        details["gpu_required"] = True
        details["gpu_available"] = bool(torch is not None and torch.cuda.is_available())
    repo_ok = bool(repo_path and Path(repo_path).exists())
    details["repo_env_var"] = repo_env_var
    details["repo_path"] = repo_path
    return RuntimeCheckResult(
        mode=mode,
        python_ok=py_ok,
        torch_ok=torch_ok,
        gpu_ok=gpu_ok,
        repo_ok=repo_ok,
        details=details,
    )


def require_runtime_compatible(result: RuntimeCheckResult) -> None:
    failed: list[str] = []
    if not result.python_ok:
        failed.append("python")
    if not result.torch_ok:
        failed.append("torch")
    if not result.gpu_ok:
        failed.append("gpu")
    if not result.repo_ok:
        failed.append("repo")
    if failed:
        raise AcademicModelError(
            code=ErrorCode.INCOMPATIBLE_RUNTIME,
            message=f"Runtime precheck failed for {', '.join(failed)}",
            details=result.details,
            safe_details={"failed_checks": failed, "mode": result.mode},
        )


def ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        raise AcademicModelError(
            code=ErrorCode.INCOMPATIBLE_RUNTIME,
            message="Container mode requested but Docker executable is unavailable",
            safe_details={"hint": "Install Docker or use native mode"},
        )


def pick_mode(payload_mode: str | None, default_mode: str) -> str:
    mode = (payload_mode or default_mode or "native").strip().lower()
    if mode not in {"native", "container"}:
        raise AcademicModelError(
            code=ErrorCode.INVALID_INPUT_SCHEMA,
            message="runtime.mode must be native or container",
            safe_details={"provided_mode": payload_mode},
        )
    return mode


def resolve_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
