"""Base abstraction for all academic model adapters."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

from academic_model_hub.errors import AcademicModelError, ErrorCode
from academic_model_hub.schemas.common import ModelDescriptor, NormalizedResult, RuntimeDeclaration
from academic_model_hub.utils.io import write_json
from academic_model_hub.utils.provenance import (
    build_provenance,
    request_hash,
    write_provenance_manifest,
)

logger = logging.getLogger(__name__)

class BaseAcademicModelAdapter(abc.ABC):
    model_name: str
    model_version: str = "unknown"
    paper_title: str
    paper_year: int | None = None
    source_url: str
    repo_url: str
    commit_or_release: str = "unknown"
    runtime: RuntimeDeclaration
    input_schema_name: str = "v1"
    supported_tasks: list[str]
    explanation_support: bool = False
    artifact_support: bool = False

    @abc.abstractmethod
    def validate_payload(self, payload: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def preprocess(self, payload: dict[str, Any]) -> Any: ...

    @abc.abstractmethod
    def run_inference(self, prepared: Any) -> Any: ...

    @abc.abstractmethod
    def postprocess(self, raw_output: Any) -> dict[str, Any]: ...

    @abc.abstractmethod
    def healthcheck(self, *, depth: str = "shallow") -> dict[str, Any]: ...

    @abc.abstractmethod
    def runtime_precheck(self, payload: dict[str, Any], *, mode: str) -> dict[str, Any]: ...

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(
            model_name=self.model_name,
            paper_title=self.paper_title,
            repo_url=self.repo_url,
            runtime_mode=self.runtime.mode,
            input_schema_name=self.input_schema_name,
            supported_tasks=self.supported_tasks,
            explanation_support=self.explanation_support,
            artifact_support=self.artifact_support,
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        working_payload = dict(payload)
        req_hash = request_hash(self.model_name, working_payload)
        working_payload["_request_hash"] = req_hash
        device = working_payload.get("device", "cpu")
        try:
            mode = str(working_payload.get("runtime_mode", self.runtime.mode))
            logger.info("academic_model_predict model=%s mode=%s request_hash=%s", self.model_name, mode, req_hash[:12])
            self.validate_payload(working_payload)
            self.runtime_precheck(working_payload, mode=mode)
            prepared = self.preprocess(working_payload)
            raw = self.run_inference(prepared)
            normalized = self.postprocess(raw)
            provenance = build_provenance(
                paper_title=self.paper_title,
                paper_year=self.paper_year,
                repo_url=self.repo_url,
                commit_or_release=self.commit_or_release,
                runtime=self.runtime,
                device=device,
                req_hash=req_hash,
            )
            artifacts = normalized.get("artifacts", {})
            manifest_target = working_payload.get("output_dir") or artifacts.get("output_dir")
            if manifest_target:
                manifest_path = write_provenance_manifest(manifest_target, provenance)
                artifacts["provenance_manifest"] = manifest_path
                write_json(
                    Path(manifest_target) / "run_summary.json",
                    {
                        "model": self.model_name,
                        "status": "success",
                        "request_hash": req_hash,
                        "runtime_mode": mode,
                        "artifacts": {k: v for k, v in artifacts.items() if isinstance(v, (str, int, float, bool))},
                    },
                )
            result = NormalizedResult(
                status="success",
                model=self.model_name,
                model_version=self.model_version,
                source_paper=self.paper_title,
                input_schema_version=self.input_schema_name,
                prediction=normalized.get("prediction", {}),
                explanations=normalized.get("explanations", {}),
                artifacts=artifacts,
                confidence=normalized.get("confidence", {}),
                provenance=provenance,
                errors=[],
            )
            return result.model_dump()
        except AcademicModelError as exc:
            return self._error_response(exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            wrapped = AcademicModelError(
                code=ErrorCode.UPSTREAM_SCRIPT_FAILURE,
                message=str(exc),
                details={},
                safe_details={"error_type": type(exc).__name__},
            )
            return self._error_response(wrapped)

    def deterministic_request_dir(self, output_dir: str | Path, req_hash: str) -> str:
        base = Path(output_dir) / self.model_name / req_hash[:12]
        out = base
        suffix = 0
        while out.exists() and any(out.iterdir()):
            suffix += 1
            out = Path(f"{base}_{suffix}")
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _error_response(self, exc: AcademicModelError) -> dict[str, Any]:
        result = NormalizedResult(
            status="error",
            model=self.model_name,
            model_version=self.model_version,
            source_paper=self.paper_title,
            input_schema_version=self.input_schema_name,
            prediction={},
            explanations={},
            artifacts={},
            confidence={},
            provenance={},
            errors=[exc.to_dict()],
        )
        return result.model_dump()
