from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

from .models import JobArtifacts, JobEvent, JobRecord, JobState
from .carla_runner.models import SimulationRunRequest, SimulationStreamMessage

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue_order: list[str] = []
        self._persist_path = persist_path
        if persist_path is not None:
            self._load_persisted()

    # ── persistence ──────────────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """Load previously persisted jobs from disk on startup."""
        if self._persist_path is None or not self._persist_path.is_dir():
            return
        loaded = 0
        for job_file in sorted(self._persist_path.glob("*.json")):
            try:
                data = json.loads(job_file.read_text(encoding="utf-8"))
                job = JobRecord.model_validate(data)
                self._jobs[job.job_id] = job
                self._queue_order.append(job.job_id)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load persisted job %s: %s", job_file.name, exc)
        if loaded:
            logger.info("Restored %d persisted jobs from %s", loaded, self._persist_path)

    def _persist_job(self, job: JobRecord) -> None:
        """Write a single job record to disk as JSON (no events to keep files small)."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            # Persist without events to keep files small and avoid serialization churn
            data = job.model_dump(mode="json", exclude={"events"})
            path = self._persist_path / f"{job.job_id}.json"
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist job %s: %s", job.job_id, exc)

    # ── public API (unchanged signatures) ────────────────────────────────

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
            self._persist_job(job)
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
            self._persist_job(job)
            return job.model_copy(deep=True)

    def append_event(self, job_id: str, payload: SimulationStreamMessage) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            events = [*job.events, JobEvent(created_at=utc_now(), payload=payload)]
            job = job.model_copy(update={"events": events, "updated_at": utc_now()})
            self._jobs[job_id] = job
            return job.model_copy(deep=True)
