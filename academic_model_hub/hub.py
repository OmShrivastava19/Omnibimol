"""Public hub API for model predict/health/list operations."""

from __future__ import annotations

from typing import Any

from academic_model_hub.adapters.crispr_dipoff_adapter import CrisprDipoffAdapter
from academic_model_hub.adapters.deepathnet_adapter import DeePathNetAdapter
from academic_model_hub.adapters.deepdtagen_adapter import DeepDTAGenAdapter
from academic_model_hub.adapters.flexpose_adapter import FlexPoseAdapter
from academic_model_hub.registry import AdapterRegistry


class AcademicModelHub:
    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or AdapterRegistry()
        if not registry:
            self._register_defaults()

    def _register_defaults(self) -> None:
        self.registry.register(FlexPoseAdapter(), aliases=["flex-pose"])
        self.registry.register(DeePathNetAdapter(), aliases=["dee_path_net"])
        self.registry.register(CrisprDipoffAdapter(), aliases=["crispr-dipoff"])
        self.registry.register(DeepDTAGenAdapter(), aliases=["deep_dta_gen"])

    def predict(self, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self.registry.get(model_name)
            return adapter.predict(payload)
        except KeyError:
            return {
                "status": "error",
                "model": model_name,
                "model_version": None,
                "source_paper": None,
                "input_schema_version": "v1",
                "prediction": {},
                "explanations": {},
                "artifacts": {},
                "confidence": {},
                "provenance": {},
                "errors": [
                    {
                        "code": "INVALID_INPUT_SCHEMA",
                        "message": f"Unknown model '{model_name}'",
                        "details": {"available_models": [m["model_name"] for m in self.list_models()]},
                    }
                ],
            }

    def healthcheck(self, model_name: str | None = None, *, depth: str = "shallow") -> dict[str, Any]:
        if model_name:
            adapter = self.registry.get(model_name)
            return {adapter.model_name: adapter.healthcheck(depth=depth)}
        return {adapter.model_name: adapter.healthcheck(depth=depth) for adapter in self.registry.list_unique()}

    def list_models(self) -> list[dict[str, Any]]:
        return [adapter.describe().model_dump() for adapter in self.registry.list_unique()]
