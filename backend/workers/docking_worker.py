"""Polling worker for real docking jobs."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from backend.core.config import get_settings
from backend.db.job_repository import JobRepository
from backend.db.session import get_session_local
from backend.services.docking import DockingExecutionError, DockingProcessor

logger = logging.getLogger(__name__)


class DockingWorker:
    """Process queued docking jobs from the shared database."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.processor = DockingProcessor(self.settings)

    def _claim_next_job(self) -> dict[str, Any] | None:
        session_local = get_session_local()
        db = session_local()
        try:
            repo = JobRepository(db)
            job = repo.claim_next_job(job_types={"docking.vina"})
            if job is None:
                return None
            return {
                "id": job.id,
                "tenant_id": job.tenant_id,
                "requested_by_user_id": job.requested_by_user_id,
                "job_type": job.job_type,
                "input_payload": job.input_payload,
            }
        finally:
            db.close()

    def _complete_job(self, *, job_id: int, result_payload: dict[str, Any]) -> None:
        session_local = get_session_local()
        db = session_local()
        try:
            repo = JobRepository(db)
            job = repo.get_job_by_id(job_id)
            if job is None:
                raise DockingExecutionError(f"Docking job {job_id} disappeared before completion")
            repo.mark_completed(job=job, result_payload=result_payload)
        finally:
            db.close()

    def _fail_job(self, *, job_id: int, error_message: str) -> None:
        session_local = get_session_local()
        db = session_local()
        try:
            repo = JobRepository(db)
            job = repo.get_job_by_id(job_id)
            if job is None:
                logger.error("Failed docking job %s vanished before failure could be recorded: %s", job_id, error_message)
                return
            repo.mark_failed(job=job, error_message=error_message)
        finally:
            db.close()

    def run_once(self) -> bool:
        job = self._claim_next_job()
        if job is None:
            return False

        job_id = int(job["id"])
        payload = dict(job["input_payload"] or {})

        def execute() -> dict[str, Any]:
            return self.processor.process_job_payload(payload)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(execute)
            result_payload = future.result(timeout=self.settings.docking_timeout_seconds)
            result_payload["job_id"] = job_id
            self._complete_job(job_id=job_id, result_payload=result_payload)
            logger.info("Docking job %s completed", job_id)
            return True
        except FuturesTimeoutError:
            error_message = f"Docking job timed out after {self.settings.docking_timeout_seconds} seconds"
        except Exception as exc:  # pragma: no cover - exercised via tests
            error_message = str(exc)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        self._fail_job(job_id=job_id, error_message=error_message)
        logger.error("Docking job %s failed: %s", job_id, error_message)
        return True

    def run_forever(self, poll_interval_seconds: float = 2.0) -> None:
        logger.info("Docking worker started with concurrency=%s", self.settings.docking_worker_concurrency)
        while True:
            worked = False
            for _ in range(self.settings.docking_worker_concurrency):
                worked = self.run_once() or worked
            if not worked:
                time.sleep(poll_interval_seconds)


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    DockingWorker().run_forever()


if __name__ == "__main__":
    main()