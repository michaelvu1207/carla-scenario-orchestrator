from __future__ import annotations

import json
import logging
import requests as _requests_lib
import threading
import uuid
from pathlib import Path

from .artifact_storage import ArtifactStorage, NullArtifactStorage, S3ArtifactStorage
from .config import Settings
from .models import (
    CancelJobResponse,
    HealthResponse,
    JobArtifacts,
    JobListResponse,
    JobRecord,
    JobState,
    JobSubmissionResponse,
    RuntimeLaunchSpec,
)
from .carla_metadata import CarlaMetadataService
from .llm import BedrockScenarioLLM, BedrockSceneAssistant
from .llm.langchain_support import LANGCHAIN_AVAILABLE, LANGSMITH_AVAILABLE, langsmith_tracing_enabled
from .runtime_backend import DockerRuntimeBackend, RuntimeBackend
from .scheduler import GpuScheduler
from .store import JobStore
from .carla_runner.dataset_repository import list_supported_maps
from .carla_runner.models import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    RecordingInfo,
    SceneAssistantRequest,
    SceneAssistantResponse,
    SimulationRunDiagnostics,
    SimulationRunRequest,
    SimulationStreamMessage,
)


logger = logging.getLogger(__name__)


class OrchestratorService:
    def __init__(
        self,
        settings: Settings,
        scheduler: GpuScheduler | None = None,
        store: JobStore | None = None,
        runtime_backend: RuntimeBackend | None = None,
        artifact_storage: ArtifactStorage | None = None,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler or GpuScheduler(settings)
        self.store = store or JobStore()
        self.runtime_backend = runtime_backend or DockerRuntimeBackend(settings)
        if artifact_storage is not None:
            self.artifact_storage = artifact_storage
        elif settings.storage_bucket:
            self.artifact_storage = S3ArtifactStorage(settings)
        else:
            self.artifact_storage = NullArtifactStorage()
        metadata_slot = self.scheduler.metadata_slot()
        self.carla_metadata = CarlaMetadataService(
            host=settings.carla_metadata_host,
            port=metadata_slot.carla_rpc_port,
            timeout=settings.carla_metadata_timeout,
        )
        self._metadata_slot_index = metadata_slot.slot_index
        self.llm = BedrockScenarioLLM()
        self.scene_assistant = BedrockSceneAssistant(carla_metadata=self.carla_metadata)
        self._cancel_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._runtime_pool_ready = False

    def startup(self) -> None:
        self._ensure_runtime_pool_started()
        if self.settings.warm_metadata_cache_on_startup:
            threading.Thread(target=self.carla_metadata.warm_cache, daemon=True).start()

    def _ensure_runtime_pool_started(self) -> None:
        with self._lock:
            if self._runtime_pool_ready:
                return
            self.runtime_backend.initialize_pool(self.scheduler)
            self._runtime_pool_ready = True

    def submit_job(self, request: SimulationRunRequest) -> JobSubmissionResponse:
        self._ensure_runtime_pool_started()
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.settings.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        artifacts = JobArtifacts(
            output_dir=str(job_dir),
            request_file=str(job_dir / "request.json"),
            runtime_settings_file=str(job_dir / "runtime_settings.json"),
        )
        job = self.store.create(job_id, request, artifacts)
        cancel_event = threading.Event()
        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        with self._lock:
            self._cancel_events[job_id] = cancel_event
            self._threads[job_id] = worker
        self.store.update_queue_positions()
        worker.start()
        current = self.store.get(job_id)
        assert current is not None
        return JobSubmissionResponse(job_id=job_id, state=current.state, queue_position=current.queue_position)


    def get_job(self, job_id: str) -> JobRecord | None:
        return self.store.get(job_id)

    def list_jobs(self) -> JobListResponse:
        return JobListResponse(items=self.store.list())

    def cancel_job(self, job_id: str) -> CancelJobResponse:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
        if cancel_event is not None:
            cancel_event.set()
        if job.state == JobState.queued:
            self.store.update(job_id, state=JobState.cancelled, error="Job cancelled before start.")
            self.store.update_queue_positions()
        current = self.store.get(job_id)
        assert current is not None
        return CancelJobResponse(job_id=job_id, state=current.state)

    def supported_maps(self) -> list[str]:
        return sorted(list_supported_maps())

    def capacity(self):
        return self.scheduler.snapshot()

    def health(self) -> HealthResponse:
        capacity = self.scheduler.snapshot()
        metadata_connected = capacity.metadata_ready
        overall_status = "healthy" if capacity.unavailable_slots == 0 and metadata_connected else "degraded"
        return HealthResponse(
            status=overall_status,
            total_slots=capacity.total_slots,
            busy_slots=capacity.busy_slots,
            queued_jobs=self.store.queued_count(),
            carla_connected=metadata_connected,
            metadata_connected=metadata_connected,
            metadata_slot_index=capacity.metadata_slot_index,
            running=capacity.busy_slots > 0,
            simulation_running=capacity.busy_slots > 0,
            connections=0,
            langchain_available=LANGCHAIN_AVAILABLE,
            langsmith_available=LANGSMITH_AVAILABLE,
            langsmith_tracing=langsmith_tracing_enabled(),
        )

    def carla_status(self):
        return self.carla_metadata.get_status()

    def carla_load_map(self, map_name: str):
        return self.carla_metadata.load_map(map_name)

    def runtime_map(self):
        return self.carla_metadata.get_runtime_map()

    def map_xodr(self) -> str:
        return self.carla_metadata.get_map_xodr()

    def map_generated(self) -> dict[str, object]:
        return self.carla_metadata.get_generated_map_with_runtime()

    def actor_blueprints(self) -> dict[str, list[str]]:
        return self.carla_metadata.list_blueprints()

    def llm_generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        return self.llm.generate(request)

    def llm_scene_assistant_chat(self, request: SceneAssistantRequest) -> SceneAssistantResponse:
        return self.scene_assistant.chat(request)

    def latest_job(self) -> JobRecord | None:
        return self.store.latest()

    def latest_running_job(self) -> JobRecord | None:
        return self.store.latest_running()

    def map_info(self) -> dict[str, object]:
        runtime_map = self.runtime_map()
        waypoints: list[dict[str, float]] = []
        for segment in runtime_map.road_segments:
            waypoints.extend(segment.centerline)
        return {
            "map_name": runtime_map.map_name,
            "waypoints": waypoints,
        }

    def list_recordings(self, source_run_id: str | None = None) -> list[RecordingInfo]:
        items: list[RecordingInfo] = []
        for job in self.store.list():
            if not job.run_id:
                continue
            if not job.artifacts.recording_path:
                continue
            if source_run_id and getattr(job.request, 'source_run_id', None) != source_run_id:
                continue
            # Find S3 URL for the MP4 if uploaded
            s3_url: str | None = None
            for artifact in job.artifacts.uploaded_artifacts:
                if artifact.kind == "MP4" and artifact.s3_key:
                    s3_url = f"https://{artifact.s3_bucket}.s3.amazonaws.com/{artifact.s3_key}"
                    break
            items.append(
                RecordingInfo(
                    run_id=job.run_id,
                    label=f"{job.request.map_name} ({job.job_id})",
                    mp4_path=job.artifacts.recording_path,
                    frames_path=None,
                    created_at=job.updated_at.isoformat(),
                    source_run_id=getattr(job.request, 'source_run_id', None),
                    s3_url=s3_url,
                )
            )
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items

    def latest_run_diagnostics(self) -> SimulationRunDiagnostics | None:
        for job in reversed(self.store.list()):
            diagnostics = self.job_diagnostics(job.job_id)
            if diagnostics is not None:
                return diagnostics
        return None

    def job_diagnostics(self, identifier: str) -> SimulationRunDiagnostics | None:
        jobs = self.store.list()
        for job in jobs:
            if job.job_id == identifier or job.run_id == identifier:
                manifest_path = job.artifacts.manifest_path
                if not manifest_path:
                    return None
                return self._read_run_diagnostics(Path(manifest_path))
        return None


    def get_job_log(self, job_id: str) -> str | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        # Try the debug_log_path from artifacts first (set after run completes)
        if job.artifacts.debug_log_path:
            p = Path(job.artifacts.debug_log_path)
            if p.is_file():
                return p.read_text(encoding='utf-8', errors='replace')
        # During a running job, search for run.log in the output directory
        job_dir = Path(job.artifacts.output_dir)
        for log_path in job_dir.rglob('run.log'):
            return log_path.read_text(encoding='utf-8', errors='replace')
        return ''

    def _read_run_diagnostics(self, manifest_path: Path) -> SimulationRunDiagnostics:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest_path.parent.name
        debug_log_path = data.get("debug_log")
        log_excerpt = ""
        if debug_log_path:
            debug_path = Path(debug_log_path)
            if debug_path.is_file():
                try:
                    lines = debug_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    log_excerpt = "\n".join(lines[-80:])
                except Exception:
                    log_excerpt = ""
        return SimulationRunDiagnostics(
            run_id=run_id,
            map_name=str(data.get("map_name") or ""),
            created_at=str(data.get("created_at") or ""),
            selected_roads=list(data.get("selected_roads") or []),
            actors=list(data.get("actors") or []),
            recording_path=data.get("recording_path"),
            scenario_log_path=data.get("scenario_log"),
            debug_log_path=debug_log_path,
            worker_error=data.get("worker_error"),
            saved_frame_count=int(data.get("saved_frame_count") or 0),
            sensor_timeout_count=int(data.get("sensor_timeout_count") or 0),
            last_sensor_frame=data.get("last_sensor_frame"),
            skipped_actors=list(data.get("skipped_actors") or []),
            log_excerpt=log_excerpt,
        )

    def _run_job(self, job_id: str) -> None:
        self._ensure_runtime_pool_started()
        job = self.store.get(job_id)
        if job is None:
            return
        with self._lock:
            cancel_event = self._cancel_events[job_id]
        try:
            lease = self.scheduler.acquire(job_id, cancel_event)
        except RuntimeError as exc:
            current = self.store.get(job_id)
            if current is not None and current.state == JobState.cancelled:
                return
            self.store.update(job_id, state=JobState.cancelled, error=str(exc))
            return

        self.store.update_queue_positions()
        self.store.update(job_id, state=JobState.starting, gpu=lease.to_model(), queue_position=0)

        runtime_spec = self._write_runtime_files(job, lease.to_model())
        self.store.update(job_id, container_name=lease.container_name)

        def on_event(payload: SimulationStreamMessage) -> None:
            self.store.append_event(job_id, payload)
            current = self.store.get(job_id)
            if current is None:
                return
            updates = {}
            if current.state == JobState.starting:
                updates["state"] = JobState.running
            if payload.error and current.state != JobState.cancelled:
                updates["error"] = payload.error
            if payload.recording is not None:
                updates["run_id"] = payload.recording.run_id
            if updates:
                self.store.update(job_id, **updates)

        try:
            result = self.runtime_backend.run_job(runtime_spec, on_event, cancel_event)
            local_artifacts = job.artifacts.model_copy(
                update={
                    "manifest_path": result.manifest_path,
                    "recording_path": result.recording_path,
                    "scenario_log_path": result.scenario_log_path,
                    "debug_log_path": result.debug_log_path,
                }
            )
            current = self.store.update(
                job_id,
                error=result.error,
                run_id=result.run_id,
                artifacts=local_artifacts,
            )
            uploaded_artifacts = self.artifact_storage.upload_job_artifacts(current)
            final_artifacts = current.artifacts
            if uploaded_artifacts:
                final_artifacts = current.artifacts.model_copy(update={"uploaded_artifacts": uploaded_artifacts})
            self.store.update(
                job_id,
                state=result.state,
                error=result.error,
                run_id=result.run_id,
                artifacts=final_artifacts,
            )
        except Exception as exc:  # noqa: BLE001
            final_state = JobState.cancelled if cancel_event.is_set() else JobState.failed
            self.store.update(job_id, state=final_state, error=str(exc))
        finally:
            self.scheduler.release(job_id)
            self.store.update_queue_positions()
            self._fire_webhook(job_id)

    def _fire_webhook(self, job_id: str) -> None:
        url = self.settings.webhook_url
        if not url:
            return
        job = self.store.get(job_id)
        if job is None:
            return
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.webhook_secret:
            headers["Authorization"] = f"Bearer {self.settings.webhook_secret}"
        body = job.model_dump(mode="json")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = _requests_lib.post(url, json=body, headers=headers, timeout=10)
                logger.info("Webhook POST to %s completed status=%s job=%s", url, resp.status_code, job_id)
                if resp.status_code < 500:
                    return
                logger.warning("Webhook POST to %s returned %s for job %s (attempt %d/%d)", url, resp.status_code, job_id, attempt + 1, max_attempts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Webhook POST to %s failed for job %s (attempt %d/%d): %s", url, job_id, attempt + 1, max_attempts, exc)
            if attempt < max_attempts - 1:
                import time
                backoff = 4 ** attempt  # 1s, 4s, 16s
                logger.info("Retrying webhook in %ds...", backoff)
                time.sleep(backoff)
        logger.error("Webhook POST to %s failed after %d attempts for job %s", url, max_attempts, job_id)

    def _write_runtime_files(self, job: JobRecord, gpu) -> RuntimeLaunchSpec:
        job_dir = Path(job.artifacts.output_dir)
        request_file = Path(job.artifacts.request_file)
        runtime_settings_file = Path(job.artifacts.runtime_settings_file)
        request_file.write_text(job.request.model_dump_json(indent=2), encoding="utf-8")
        runtime_settings = {
            "carla_host": "127.0.0.1",
            "carla_port": gpu.carla_rpc_port,
            "carla_timeout": self.settings.carla_timeout_seconds,
            "tm_port": gpu.traffic_manager_port,
            "output_root": str(job_dir),
        }
        runtime_settings_file.write_text(json.dumps(runtime_settings, indent=2), encoding="utf-8")
        return RuntimeLaunchSpec(
            job_id=job.job_id,
            request_file=str(request_file),
            runtime_settings_file=str(runtime_settings_file),
            output_dir=str(job_dir),
            gpu=gpu,
        )
