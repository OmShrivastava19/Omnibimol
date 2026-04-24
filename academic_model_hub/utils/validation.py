"""Validation helpers shared by model adapters."""

from __future__ import annotations

import re
from pathlib import Path

from academic_model_hub.errors import AcademicModelError, ErrorCode

GENE_OMIC_RE = re.compile(r"^[A-Za-z0-9\-]+_[A-Za-z0-9]+$")
DNA_SEQ_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def require_existing_file(path: str, *, expected_suffix: str | None = None, field_name: str = "path") -> None:
    p = Path(path)
    if not p.exists():
        raise AcademicModelError(
            code=ErrorCode.INVALID_INPUT_SCHEMA,
            message=f"{field_name} does not exist",
            details={"field": field_name, "path": path},
        )
    if expected_suffix and p.suffix.lower() != expected_suffix.lower():
        raise AcademicModelError(
            code=ErrorCode.INVALID_INPUT_SCHEMA,
            message=f"{field_name} must be {expected_suffix}",
            details={"field": field_name, "path": path},
        )


def validate_dna_sequence(sequence: str, *, field_name: str) -> None:
    if not sequence or not DNA_SEQ_RE.fullmatch(sequence):
        raise AcademicModelError(
            code=ErrorCode.MALFORMED_BIOLOGICAL_INPUT,
            message=f"{field_name} must contain DNA bases only",
            details={"field": field_name},
        )


def validate_gene_omic_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for col in columns:
        if GENE_OMIC_RE.fullmatch(col):
            valid.append(col)
        else:
            invalid.append(col)
    return valid, invalid
