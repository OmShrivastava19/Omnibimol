"""Protein localization inference endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.core.config import get_settings
from backend.services.protein_localization import (
    ProteinLocalizationService,
    clean_protein_sequence,
    get_protein_localization_service,
)


router = APIRouter(prefix="/protein-localization", tags=["protein-localization"])


class ProteinLocalizationPredictRequest(BaseModel):
    sequence: str = Field(..., min_length=20, max_length=20000, description="Amino acid sequence")
    confidence_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold gate used for evidence filtering",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        return clean_protein_sequence(value)


class ProteinLocalizationPredictResponse(BaseModel):
    localization: str
    confidence: float
    membrane_risk: float
    wetlab_prioritization_score: float
    recommended_assay: str
    evidence_passed: bool
    sequence_length: int
    non_membrane_probability: float
    confidence_threshold: float
    all_probabilities: dict[str, float]
    model_metadata: dict[str, Any]


@router.get("/health")
def health(service: ProteinLocalizationService = Depends(get_protein_localization_service)) -> dict[str, Any]:
    return service.health_snapshot()


@router.post("/predict", response_model=ProteinLocalizationPredictResponse)
def predict(
    body: ProteinLocalizationPredictRequest,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    __: AuthPrincipal = Depends(get_current_principal),
    service: ProteinLocalizationService = Depends(get_protein_localization_service),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.localizer_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Protein localization inference is disabled",
        )

    try:
        return service.predict(
            body.sequence,
            confidence_threshold=body.confidence_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc