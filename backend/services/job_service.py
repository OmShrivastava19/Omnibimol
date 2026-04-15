"""Async job state handling service."""

from backend.db.job_repository import JobRepository
from backend.db.models import JobRun


class JobService:
    """Polling-driven job progression for tenant-scoped workloads."""

    def __init__(self, repo: JobRepository):
        self.repo = repo

    def enqueue(
        self,
        *,
        tenant_id: int,
        requested_by_user_id: int,
        job_type: str,
        input_payload: dict,
    ) -> JobRun:
        return self.repo.create_job(
            tenant_id=tenant_id,
            requested_by_user_id=requested_by_user_id,
            job_type=job_type,
            input_payload=input_payload,
        )

    def advance(self, job: JobRun) -> JobRun:
        if job.status in {"completed", "failed"}:
            return job
        if job.status == "queued":
            return self.repo.mark_running(job)

        # running -> terminal (simulated worker execution)
        try:
            if job.job_type == "fail_demo":
                raise RuntimeError("Simulated worker failure")

            result = {
                "job_type": job.job_type,
                "processed_items": len(job.input_payload.get("items", [])),
                "message": "Job completed successfully",
            }
            return self.repo.mark_completed(job=job, result_payload=result)
        except Exception as exc:
            return self.repo.mark_failed(job=job, error_message=str(exc))
