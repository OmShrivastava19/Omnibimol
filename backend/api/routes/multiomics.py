"""Multi-omics fusion prediction endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.core.config import get_settings
from backend.services.multiomics_fusion import FusionSettings, MultiOmicsFusionService

router = APIRouter(prefix="/multiomics", tags=["multiomics"])


class DrugInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    smiles: str | None = None
    descriptors: dict[str, float] = Field(default_factory=dict)


class GenomicsInput(BaseModel):
    mutations: dict[str, float] = Field(default_factory=dict)
    cnv: dict[str, float] = Field(default_factory=dict)


class OmicsSampleInput(BaseModel):
    genomics: GenomicsInput
    transcriptomics: dict[str, float] = Field(default_factory=dict)
    proteomics: dict[str, float] = Field(default_factory=dict)


class MultiOmicsPredictRequest(BaseModel):
    drug: DrugInput
    sample_omics: OmicsSampleInput | None = None
    cohort_omics: list[OmicsSampleInput] | None = None


class TopPathway(BaseModel):
    pathway: str
    aggregate_contribution: float


class TopFeature(BaseModel):
    pathway: str
    feature: str
    value: float


class ModalityUsageSummary(BaseModel):
    samples_evaluated: int
    used_modalities: list[str]
    missing_modalities: list[str]


class MultiOmicsPredictResponse(BaseModel):
    predicted_response_probability: float
    predicted_sensitivity_score: float
    uncertainty: float
    top_pathways: list[TopPathway]
    top_features: list[TopFeature]
    explanation_text: str
    modality_usage_summary: ModalityUsageSummary


@router.post("/predict", response_model=MultiOmicsPredictResponse)
def predict_multiomics_response(
    body: MultiOmicsPredictRequest,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    __: AuthPrincipal = Depends(get_current_principal),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.multiomics_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-omics fusion engine is disabled",
        )

    fusion_service = MultiOmicsFusionService(
        settings=FusionSettings(
            calibration_a=settings.multiomics_calibration_a,
            calibration_b=settings.multiomics_calibration_b,
            transcriptomics_weight=settings.multiomics_transcriptomics_weight,
            genomics_weight=settings.multiomics_genomics_weight,
            proteomics_weight=settings.multiomics_proteomics_weight,
        )
    )
    try:
        return fusion_service.predict(
            drug_name=body.drug.name,
            smiles=body.drug.smiles,
            drug_descriptors=body.drug.descriptors,
            sample_omics=body.sample_omics.model_dump() if body.sample_omics else None,
            cohort_omics=[row.model_dump() for row in body.cohort_omics] if body.cohort_omics else None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
