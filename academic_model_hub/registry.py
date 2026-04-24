"""Registry for academic model adapters."""

from __future__ import annotations

from academic_model_hub.base import BaseAcademicModelAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BaseAcademicModelAdapter] = {}

    def register(self, adapter: BaseAcademicModelAdapter, aliases: list[str] | None = None) -> None:
        canonical = adapter.model_name.lower().strip()
        self._adapters[canonical] = adapter
        for alias in aliases or []:
            self._adapters[alias.lower().strip()] = adapter

    def get(self, model_name: str) -> BaseAcademicModelAdapter:
        key = model_name.lower().strip()
        if key not in self._adapters:
            available = sorted({a.model_name for a in self._adapters.values()})
            raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
        return self._adapters[key]

    def list_unique(self) -> list[BaseAcademicModelAdapter]:
        unique: dict[str, BaseAcademicModelAdapter] = {}
        for adapter in self._adapters.values():
            unique[adapter.model_name] = adapter
        return [unique[name] for name in sorted(unique)]
