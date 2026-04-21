"""Tenant-scoped persistence helpers for async jobs."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import JobRun


class JobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_job(
        self,
        *,
        tenant_id: int,
        requested_by_user_id: int,
        job_type: str,
        input_payload: dict,
    ) -> JobRun:
        job = JobRun(
            tenant_id=tenant_id,
            requested_by_user_id=requested_by_user_id,
            job_type=job_type,
            status="queued",
            input_payload=input_payload,
            result_payload={},
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job_for_tenant(self, *, tenant_id: int, job_id: int) -> JobRun | None:
        stmt = select(JobRun).where(JobRun.id == job_id, JobRun.tenant_id == tenant_id)
        return self.db.scalar(stmt)

    def get_job_by_id(self, job_id: int) -> JobRun | None:
        return self.db.get(JobRun, job_id)

    def mark_running(self, job: JobRun) -> JobRun:
        if job.status == "queued":
            job.status = "running"
            job.started_at = datetime.now(UTC)
            self.db.commit()
            self.db.refresh(job)
        return job

    def claim_next_job(self, *, job_types: set[str] | None = None) -> JobRun | None:
        stmt = select(JobRun).where(JobRun.status == "queued")
        if job_types:
            stmt = stmt.where(JobRun.job_type.in_(sorted(job_types)))
        stmt = stmt.order_by(JobRun.created_at.asc()).with_for_update(skip_locked=True)

        job = self.db.scalar(stmt)
        if job is None:
            return None
        return self.mark_running(job)

    def mark_completed(self, *, job: JobRun, result_payload: dict) -> JobRun:
        job.status = "completed"
        job.result_payload = result_payload
        job.error_message = None
        job.completed_at = datetime.now(UTC)
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_failed(self, *, job: JobRun, error_message: str) -> JobRun:
        job.status = "failed"
        job.result_payload = {}
        job.error_message = error_message
        job.completed_at = datetime.now(UTC)
        self.db.commit()
        self.db.refresh(job)
        return job
