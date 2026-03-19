from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from .models import JobArtifacts, JobEvent, JobRecord, JobState
from .carla_runner.models import SimulationRunRequest, SimulationStreamMessage

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    # Jobs are ephemeral in-memory. Aurora DB in simcloud-platform is the authoritative record.

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue_order: list[str] = []

    def create(self, job_id: str, request: SimulationRunRequest, artifacts: JobArtifacts) -> JobRecord:
        with self._lock:
            now = utc_now()
            self._queue_order.append(job_id)
            job = JobRecord(
                job_id=job_id,
                state=JobState.queued,
                created_at=now,
                updated_at=now,
                request=request,
                queue_position=len(self._queue_order),
                artifacts=artifacts,
            )
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.model_copy(deep=True)

    def list(self) -> list[JobRecord]:
        with self._lock:
            ordered = [self._jobs[job_id] for job_id in self._queue_order if job_id in self._jobs]
            return [job.model_copy(deep=True) for job in ordered]

    def queued_job_ids(self) -> list[str]:
        with self._lock:
            return [job_id for job_id in self._queue_order if self._jobs[job_id].state == JobState.queued]

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.state == JobState.queued)

    def latest(self) -> JobRecord | None:
        with self._lock:
            if not self._queue_order:
                return None
            for job_id in reversed(self._queue_order):
                job = self._jobs.get(job_id)
                if job is not None:
                    return job.model_copy(deep=True)
            return None

    def latest_running(self) -> JobRecord | None:
        with self._lock:
            for job_id in reversed(self._queue_order):
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.state in {JobState.starting, JobState.running}:
                    return job.model_copy(deep=True)
            return None

    def update_queue_positions(self) -> None:
        queued_ids = self.queued_job_ids()
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.state == JobState.queued:
                    job.queue_position = queued_ids.index(job_id) + 1
                else:
                    job.queue_position = 0
                job.updated_at = utc_now()

    def update(self, job_id: str, **updates) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job = job.model_copy(update={**updates, "updated_at": utc_now()})
            self._jobs[job_id] = job
            return job.model_copy(deep=True)

    def append_event(self, job_id: str, payload: SimulationStreamMessage) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            events = [*job.events, JobEvent(created_at=utc_now(), payload=payload)]
            job = job.model_copy(update={"events": events, "updated_at": utc_now()})
            self._jobs[job_id] = job
            return job.model_copy(deep=True)
