"""API endpoints for academic model hub."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, field_validator

from academic_model_hub import AcademicModelHub

router = APIRouter(prefix="/academic-models", tags=["academic-models"])
hub = AcademicModelHub()
logger = logging.getLogger(__name__)

# Model name validation pattern
MODEL_NAME_PATTERN = re.compile(r'^[a-z0-9_\-]+$')
VALID_MODELS = {
    'deepathnet', 'flexpose', 'deepdtagen', 'crispr_dipoff',
    'protein_mpnn', 'esmfold', 'alphafold2'
}


class PredictRequest(BaseModel):
    """Request model for academic model predictions."""
    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the academic model to use for prediction (e.g., 'deepathnet', 'flexpose', 'deepdtagen')"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific input data payload (structure depends on model_name)"
    )
    runtime_mode: Literal["native", "simulation"] = Field(
        "native",
        description="Execution mode: 'native' for production inference, 'simulation' for testing/demo"
    )
    timeout_seconds: int = Field(
        300,
        ge=10,
        le=3600,
        description="Maximum runtime for model inference in seconds (10-3600 seconds)"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format and existence."""
        v = v.strip().lower()
        
        # Check format
        if not MODEL_NAME_PATTERN.match(v):
            raise ValueError(
                "Model name must contain only lowercase alphanumeric characters, hyphens, and underscores"
            )
        
        # Note: We log a warning for unknown models but don't fail hard to allow flexibility
        if v not in VALID_MODELS:
            logger.warning(f"Requested model '{v}' is not in standard model registry")
        
        return v

    @field_validator('payload')
    @classmethod
    def validate_payload_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate payload is a proper dictionary with reasonable size."""
        if not isinstance(v, dict):
            raise ValueError("Payload must be a dictionary")
        
        # Validate payload size (rough JSON serialization check)
        import json
        try:
            serialized = json.dumps(v)
            if len(serialized) > 50 * 1024 * 1024:  # 50MB limit for model inputs
                raise ValueError("Payload exceeds maximum size of 50MB")
        except TypeError as e:
            raise ValueError(f"Payload contains non-serializable objects: {e}")
        
        return v


class ModelHealthRequest(BaseModel):
    """Request model for model health checks."""
    model_name: str | None = Field(
        None,
        max_length=100,
        description="Specific model to check. If omitted, checks all models."
    )
    depth: Literal["shallow", "deep"] = Field(
        "shallow",
        description="Health check depth: 'shallow' for quick status, 'deep' for comprehensive testing"
    )
    include_performance_metrics: bool = Field(
        False,
        description="Whether to include performance metrics in the health check response"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name_if_provided(cls, v: str | None) -> str | None:
        """Validate model name format if provided."""
        if v is None:
            return v
        
        v = v.strip().lower()
        if not MODEL_NAME_PATTERN.match(v):
            raise ValueError(
                "Model name must contain only lowercase alphanumeric characters, hyphens, and underscores"
            )
        
        if v not in VALID_MODELS:
            logger.warning(f"Health check requested for unknown model '{v}'")
        
        return v


@router.get("/models")
def list_models() -> list[dict[str, Any]]:
    """List all available academic models with their metadata."""
    return hub.list_models()


@router.get("/health")
def health(
    model_name: str | None = Query(
        None,
        max_length=100,
        description="Specific model to check. If omitted, checks all models."
    ),
    depth: Literal["shallow", "deep"] = Query(
        "shallow",
        description="Health check depth: 'shallow' for quick status, 'deep' for comprehensive testing"
    ),
) -> dict[str, Any]:
    """Check health status of academic models."""
    # Validate model_name if provided
    if model_name:
        model_name = model_name.strip().lower()
        if not MODEL_NAME_PATTERN.match(model_name):
            raise ValueError(
                "Model name must contain only lowercase alphanumeric characters, hyphens, and underscores"
            )
        if model_name not in VALID_MODELS:
            logger.warning(f"Health check requested for unknown model '{model_name}'")
    
    return hub.healthcheck(model_name, depth=depth)


@router.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    """Run prediction using the specified academic model.
    
    Args:
        body: PredictRequest containing model_name, payload, runtime_mode, and timeout_seconds
        
    Returns:
        Prediction results from the specified model
    """
    logger.info(
        "academic_model_api_predict model=%s runtime_mode=%s timeout=%s",
        body.model_name,
        body.runtime_mode,
        body.timeout_seconds
    )
    return hub.predict(
        model_name=body.model_name,
        payload=body.payload,
        runtime_mode=body.runtime_mode,
        timeout_seconds=body.timeout_seconds
    )
