"""Async job endpoints for heavy workloads."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.audit.service import AuditService
from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import get_tenant_context, require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.db.job_repository import JobRepository
from backend.db.session import get_db
from backend.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobEnqueueRequest(BaseModel):
    job_type: str = Field(min_length=1, max_length=100)
    payload: dict = Field(default_factory=dict)


def _serialize_job(job) -> dict[str, object]:
    return {
        "id": job.id,
        "tenant_id": job.tenant_id,
        "requested_by_user_id": job.requested_by_user_id,
        "job_type": job.job_type,
        "status": job.status,
        "input_payload": job.input_payload,
        "result_payload": job.result_payload,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("")
def enqueue_job(
    body: JobEnqueueRequest,
    _: AuthPrincipal = Depends(require_permission("project.write")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = JobRepository(db)
    job = JobService(repo).enqueue(
        tenant_id=tenant_context.tenant_id,
        requested_by_user_id=tenant_context.user_id,
        job_type=body.job_type,
        input_payload=body.payload,
    )
    AuditService(db).log_event(
        tenant_id=tenant_context.tenant_id,
        actor_id=tenant_context.user_id,
        action="job.enqueued",
        resource_type="job",
        resource_id=str(job.id),
        ip_address=None,
        user_agent=None,
        details={"job_type": body.job_type},
        commit=True,
    )
    return _serialize_job(job)


@router.get("/{job_id}")
def get_job_status(
    job_id: int,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = JobRepository(db)
    job = repo.get_job_for_tenant(tenant_id=tenant_context.tenant_id, job_id=job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    previous_status = job.status
    job = JobService(repo).advance(job)
    if previous_status != job.status and job.status in {"completed", "failed"}:
        action = "job.completed" if job.status == "completed" else "job.failed"
        AuditService(db).log_event(
            tenant_id=tenant_context.tenant_id,
            actor_id=tenant_context.user_id,
            action=action,
            resource_type="job",
            resource_id=str(job.id),
            ip_address=None,
            user_agent=None,
            details={"job_type": job.job_type, "status": job.status},
            commit=True,
        )
    return _serialize_job(job)
