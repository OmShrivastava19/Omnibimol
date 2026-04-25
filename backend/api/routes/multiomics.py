"""Multi-omics fusion prediction endpoints."""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.core.config import get_settings
from backend.services.multiomics_fusion import FusionSettings, MultiOmicsFusionService

router = APIRouter(prefix="/multiomics", tags=["multiomics"])

# SMILES pattern validation (basic check for common SMILES characters)
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9\[\]\(\)\\=@#+\-]+$')


class DrugInput(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Drug name or identifier"
    )
    smiles: str | None = Field(
        None,
        description="Ligand SMILES string (format validation: alphanumeric with brackets, parentheses, and special SMILES characters)"
    )
    descriptors: dict[str, float] = Field(
        default_factory=dict,
        description="Optional molecular descriptors (keys: descriptor names, values: float values)"
    )

    @field_validator('name')
    @classmethod
    def validate_drug_name(cls, v: str) -> str:
        """Validate drug name format: alphanumeric, spaces, hyphens, underscores allowed."""
        if not re.match(r'^[A-Za-z0-9\s\-_()]+$', v):
            raise ValueError("Drug name must contain only alphanumeric characters, spaces, hyphens, underscores, and parentheses")
        return v.strip()

    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v: str | None) -> str | None:
        """Validate SMILES format."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not SMILES_PATTERN.match(v):
            raise ValueError(
                "Invalid SMILES string. SMILES must contain only valid chemical notation "
                "(alphanumeric, brackets, parentheses, =, @, #, +, -, backslash)"
            )
        if len(v) > 2000:
            raise ValueError("SMILES string is too long (max 2000 characters)")
        return v


class GenomicsInput(BaseModel):
    mutations: dict[str, float] = Field(
        default_factory=dict,
        description="Mutation data (keys: gene names, values: mutation scores or frequencies)"
    )
    cnv: dict[str, float] = Field(
        default_factory=dict,
        description="Copy number variation data (keys: chromosomal regions, values: CNV values)"
    )

    @field_validator('mutations', 'cnv')
    @classmethod
    def validate_omics_values(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all values in omics data are valid floats."""
        if not isinstance(v, dict):
            raise ValueError("Mutations and CNV data must be dictionaries")
        for key, value in v.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Invalid omics value for key '{key}': expected numeric value, got {type(value).__name__}")
            if not (-1e6 < value < 1e6):  # Reasonable bounds for biological data
                raise ValueError(f"Omics value for key '{key}' is out of reasonable range: {value}")
        return v


class OmicsSampleInput(BaseModel):
    genomics: GenomicsInput = Field(
        ...,
        description="Genomic data including mutations and copy number variations"
    )
    transcriptomics: dict[str, float] = Field(
        default_factory=dict,
        description="Gene expression data (keys: gene IDs, values: expression levels)"
    )
    proteomics: dict[str, float] = Field(
        default_factory=dict,
        description="Protein abundance data (keys: protein IDs, values: abundance levels)"
    )

    @field_validator('transcriptomics', 'proteomics')
    @classmethod
    def validate_omics_values(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all values in omics data are valid floats."""
        if not isinstance(v, dict):
            raise ValueError("Transcriptomics and proteomics data must be dictionaries")
        for key, value in v.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Invalid omics value for key '{key}': expected numeric value, got {type(value).__name__}")
            if not (-1e6 < value < 1e6):  # Reasonable bounds for biological data
                raise ValueError(f"Omics value for key '{key}' is out of reasonable range: {value}")
        return v


class MultiOmicsPredictRequest(BaseModel):
    drug: DrugInput = Field(
        ...,
        description="Drug information including name, SMILES, and molecular descriptors"
    )
    sample_omics: OmicsSampleInput | None = Field(
        None,
        description="Single sample omics data for patient-specific prediction"
    )
    cohort_omics: list[OmicsSampleInput] | None = Field(
        None,
        description="Cohort-level omics data for statistical analysis"
    )
    runtime_mode: Literal["native", "simulation"] = Field(
        "native",
        description="Execution mode: 'native' for production inference, 'simulation' for testing"
    )

    @field_validator('cohort_omics')
    @classmethod
    def validate_cohort_size(cls, v: list[OmicsSampleInput] | None) -> list[OmicsSampleInput] | None:
        """Validate cohort size is reasonable."""
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("Cohort omics list must not be empty if provided")
        if len(v) > 10000:
            raise ValueError("Cohort size exceeds maximum of 10,000 samples")
        return v


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
    """Predict multi-omics response using drug and patient/cohort genomic data.
    
    Args:
        body: MultiOmicsPredictRequest with drug info and omics data
        
    Returns:
        MultiOmicsPredictResponse with predictions, uncertainty, and explanations
    """
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
            runtime_mode=body.runtime_mode,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
