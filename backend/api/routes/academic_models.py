"""API endpoints for academic model hub."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from academic_model_hub import AcademicModelHub

router = APIRouter(prefix="/academic-models", tags=["academic-models"])
hub = AcademicModelHub()
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_name: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


@router.get("/models")
def list_models() -> list[dict[str, Any]]:
    return hub.list_models()


@router.get("/health")
def health(model_name: str | None = None, depth: str = "shallow") -> dict[str, Any]:
    return hub.healthcheck(model_name, depth=depth)


@router.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    logger.info("academic_model_api_predict model=%s", body.model_name)
    return hub.predict(model_name=body.model_name, payload=body.payload)
