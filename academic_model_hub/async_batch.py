"""Minimal async batch queue for model hub predictions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from academic_model_hub.hub import AcademicModelHub


@dataclass(slots=True)
class BatchJob:
    job_id: str
    model_name: str
    payload: dict[str, Any]
    status: str = "queued"
    result: dict[str, Any] | None = None


class AsyncBatchQueue:
    def __init__(self, hub: AcademicModelHub | None = None) -> None:
        self.hub = hub or AcademicModelHub()
        self._jobs: dict[str, BatchJob] = {}

    def enqueue(self, model_name: str, payload: dict[str, Any]) -> str:
        job_id = str(uuid4())
        self._jobs[job_id] = BatchJob(job_id=job_id, model_name=model_name, payload=payload)
        return job_id

    async def run_job(self, job_id: str) -> dict[str, Any]:
        job = self._jobs[job_id]
        job.status = "running"
        await asyncio.sleep(0)
        job.result = self.hub.predict(job.model_name, job.payload)
        job.status = "finished"
        return job.result

    def get_job(self, job_id: str) -> BatchJob:
        return self._jobs[job_id]
