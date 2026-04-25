"""ChemProt interaction scoring endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.services.chemprot import ChemProtInteractionService, get_chemprot_service

router = APIRouter(prefix="/chemprot", tags=["chemprot"])


class ChemProtAbstractInput(BaseModel):
    pmid: str | None = Field(default=None, max_length=32)
    title: str | None = Field(default=None, max_length=1000)
    abstract: str = Field(..., min_length=1, max_length=20000)


class ChemProtScoreRequest(BaseModel):
    drug_name: str | None = Field(default=None, max_length=500)
    chemical: str | None = Field(default=None, max_length=500)
    disease_context: str | None = Field(default=None, max_length=1000)
    candidate_proteins: list[str] | None = Field(default=None)
    abstracts: list[ChemProtAbstractInput] | None = Field(default=None)
    pmids: list[str] | None = Field(default=None)
    max_results: int = Field(default=10, ge=1, le=100)
    runtime_mode: Literal["native", "simulation"] = Field(default="native")

    @field_validator("candidate_proteins")
    @classmethod
    def _clean_candidates(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip().upper() for item in value if isinstance(item, str) and item.strip()]
        return cleaned or None

    @field_validator("pmids")
    @classmethod
    def _clean_pmids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return cleaned or None

    @model_validator(mode="after")
    def _ensure_chemical_present(self) -> "ChemProtScoreRequest":
        if not (self.chemical or self.drug_name):
            raise ValueError("Either chemical or drug_name must be provided")
        return self


class ChemProtScoreResponse(BaseModel):
    ranked_targets: list[dict[str, Any]]
    interaction_probability: float
    final_score: float
    reranker_used: bool
    evidence: list[dict[str, Any]]
    model_metadata: dict[str, Any]
    resolved_abstracts: list[dict[str, Any]]
    resolved_candidate_proteins: list[str]
    degraded_mode: bool
    latency_ms: float


@router.get("/health")
def health(service: ChemProtInteractionService = Depends(get_chemprot_service)) -> dict[str, Any]:
    return service.health_snapshot()


@router.post("/score", response_model=ChemProtScoreResponse)
async def score(
    body: ChemProtScoreRequest,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    __: AuthPrincipal = Depends(get_current_principal),
    service: ChemProtInteractionService = Depends(get_chemprot_service),
) -> dict[str, Any]:
    if not service.settings.chemprot_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChemProt interaction scoring is disabled",
        )

    try:
        return await service.score_request(
            chemical=body.chemical or body.drug_name or "",
            disease_context=body.disease_context,
            candidate_proteins=body.candidate_proteins,
            abstracts=[row.model_dump() for row in body.abstracts] if body.abstracts else None,
            pmids=body.pmids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
