"""Core schemas for payload/result normalization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeDeclaration(BaseModel):
    mode: Literal["native", "container"] = "native"
    python_version: str
    torch_version: str | None = None
    gpu_optional: bool = True
    gpu_required: bool = False


class ErrorItem(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class NormalizedResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["success", "error"]
    model: str
    model_version: str | None = None
    source_paper: str | None = None
    input_schema_version: str = "v1"
    prediction: dict[str, Any] = Field(default_factory=dict)
    explanations: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    confidence: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    errors: list[ErrorItem] = Field(default_factory=list)


class ModelDescriptor(BaseModel):
    model_name: str
    paper_title: str
    repo_url: str
    runtime_mode: str
    input_schema_name: str
    supported_tasks: list[str]
    explanation_support: bool
    artifact_support: bool
