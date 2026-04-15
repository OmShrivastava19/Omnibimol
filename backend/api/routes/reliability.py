"""Reliability and validation endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.reference_data_service import ReferenceDataService
from backend.services.upload_validation import validate_upload_payload

router = APIRouter(prefix="/reliability", tags=["reliability"])


class ValidateUploadRequest(BaseModel):
    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_type: str | None = None


@router.get("/upstream-status")
def upstream_status(url: str) -> dict[str, object]:
    return ReferenceDataService().fetch_upstream_status(url)


@router.post("/validate-upload")
def validate_upload(body: ValidateUploadRequest) -> dict[str, object]:
    result = validate_upload_payload(
        filename=body.filename,
        content=body.content,
        content_type=body.content_type,
    )
    return {"status": "valid", **result}
