"""Error types and helpers for academic model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT_SCHEMA = "INVALID_INPUT_SCHEMA"
    MISSING_WEIGHTS = "MISSING_WEIGHTS"
    INCOMPATIBLE_RUNTIME = "INCOMPATIBLE_RUNTIME"
    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    MALFORMED_BIOLOGICAL_INPUT = "MALFORMED_BIOLOGICAL_INPUT"
    UPSTREAM_SCRIPT_FAILURE = "UPSTREAM_SCRIPT_FAILURE"
    OUTPUT_PARSE_FAILURE = "OUTPUT_PARSE_FAILURE"


@dataclass(slots=True)
class AcademicModelError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    safe_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.safe_details or self.details,
        }
