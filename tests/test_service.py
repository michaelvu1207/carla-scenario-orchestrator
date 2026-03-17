from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.models import JobState, RuntimeExecutionResult, RuntimeLaunchSpec, StoredArtifact
from orchestrator.service import OrchestratorService
from orchestrator.carla_runner.models import (
    ActorDraft,
    ActorRoadAnchor,
    RecordingInfo,
    SelectedRoad,
    SimulationRunRequest,
    SimulationStreamMessage,
)


class FakeRuntimeBackend:
    def __init__(self) -> None:
        self.started_specs: list[RuntimeLaunchSpec] = []
        self.paused_jobs: set[str] = set()
        self.initialize_calls = 0

    def initialize_pool(self, scheduler) -> None:
        self.initialize_calls += 1

    def run_job(self, spec, on_event, cancel_event):
        self.started_specs.append(spec)
        on_event(
            SimulationStreamMessage(
                frame=1,
                timestamp=time.time(),
                actors=[],
            )
        )
        on_event(
            SimulationStreamMessage(
                frame=2,
                timestamp=time.time(),
                actors=[],
                recording=RecordingInfo(
                    run_id=f"run-{spec.job_id}",
                    label="Test",
                    created_at="2026-03-16T00:00:00Z",
                ),
            )
        )
        if cancel_event.is_set():
            return RuntimeExecutionResult(state=JobState.cancelled, error="Job cancelled.")
        return RuntimeExecutionResult(
            state=JobState.succeeded,
            run_id=f"run-{spec.job_id}",
            manifest_path=str(Path(spec.output_dir) / "run" / "manifest.json"),
            recording_path=str(Path(spec.output_dir) / "run" / "recording.mp4"),
        )

    def pause_job(self, job_id: str) -> bool:
        self.paused_jobs.add(job_id)
        return True

    def resume_job(self, job_id: str) -> bool:
        self.paused_jobs.discard(job_id)
        return True

    def is_job_paused(self, job_id: str) -> bool:
        return job_id in self.paused_jobs


class FakeArtifactStorage:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def upload_job_artifacts(self, job):
        self.calls.append(job.job_id)
        return [
            StoredArtifact(
                kind="MP4",
                label="recording.mp4",
                local_path=job.artifacts.recording_path,
                content_type="video/mp4",
                file_ext="mp4",
                size_bytes=123,
                checksum_sha256="abc123",
                s3_bucket="simcloud-assets-public-test",
                s3_key=f"runs/{job.request.source_run_id or job.job_id}/executions/{job.job_id}/recording.mp4",
                s3_uri=f"s3://simcloud-assets-public-test/runs/{job.request.source_run_id or job.job_id}/executions/{job.job_id}/recording.mp4",
            )
        ]


def sample_request() -> SimulationRunRequest:
    return SimulationRunRequest(
        map_name="Town10HD_Opt",
        selected_roads=[SelectedRoad(id="10", name="Road 10")],
        actors=[
            ActorDraft(
                id="ego-1",
                label="Ego",
                kind="vehicle",
                role="ego",
                blueprint="vehicle.tesla.model3",
                spawn=ActorRoadAnchor(road_id="10"),
            )
        ],
        duration_seconds=1.0,
        topdown_recording=False,
    )


class ServiceTests(unittest.TestCase):
    def make_service(self, gpu_devices=("0", "1"), artifact_storage=None) -> OrchestratorService:
        temp_dir = Path(tempfile.mkdtemp(prefix="carla-orchestrator-tests-"))
        settings = Settings(
            repo_root=temp_dir,
            jobs_root=temp_dir / "runs",
            gpu_devices=gpu_devices,
            carla_image="carla:test",
            carla_container_prefix="carla-orch",
            carla_startup_timeout_seconds=5,
            carla_rpc_port_base=2000,
            traffic_manager_port_base=8000,
            port_stride=100,
            carla_timeout_seconds=20,
            python_executable="python3",
            docker_network_mode="host",
            carla_start_command_template="./CarlaUE4.sh -carla-rpc-port={rpc_port}",
            carla_metadata_host="127.0.0.1",
            carla_metadata_port=2000,
            carla_metadata_timeout=20,
            storage_bucket="simcloud-assets-public-test",
            storage_region="us-east-1",
            storage_prefix="runs",
        )
        settings.jobs_root.mkdir(parents=True, exist_ok=True)
        return OrchestratorService(
            settings=settings,
            runtime_backend=FakeRuntimeBackend(),
            artifact_storage=artifact_storage or FakeArtifactStorage(),
        )

    def test_submit_job_runs_to_completion(self) -> None:
        service = self.make_service()
        response = service.submit_job(sample_request())
        deadline = time.time() + 3
        while time.time() < deadline:
            job = service.get_job(response.job_id)
            if job is not None and job.state == JobState.succeeded:
                break
            time.sleep(0.05)
        job = service.get_job(response.job_id)
        assert job is not None
        self.assertEqual(job.state, JobState.succeeded)
        self.assertEqual(job.gpu.device_id if job.gpu else None, "0")
        self.assertEqual(job.container_name, "carla-orch-slot-0")
        self.assertEqual(len(job.events), 2)
        self.assertTrue(job.artifacts.recording_path.endswith("recording.mp4"))
        self.assertEqual(len(job.artifacts.uploaded_artifacts), 1)
        self.assertTrue(job.artifacts.uploaded_artifacts[0].s3_uri.endswith("recording.mp4"))

    def test_cancel_queued_job_marks_it_cancelled(self) -> None:
        blocker = threading.Event()

        class BlockingRuntimeBackend(FakeRuntimeBackend):
            def run_job(self, spec, on_event, cancel_event):
                blocker.wait(timeout=1)
                return super().run_job(spec, on_event, cancel_event)

        service = self.make_service(gpu_devices=("0",))
        service = OrchestratorService(settings=service.settings, runtime_backend=BlockingRuntimeBackend())
        first = service.submit_job(sample_request())
        second = service.submit_job(sample_request())
        time.sleep(0.1)
        cancelled = service.cancel_job(second.job_id)
        self.assertEqual(cancelled.state, JobState.cancelled)
        blocker.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            job = service.get_job(first.job_id)
            if job is not None and job.state == JobState.succeeded:
                break
            time.sleep(0.05)
        queued_job = service.get_job(second.job_id)
        assert queued_job is not None
        self.assertEqual(queued_job.state, JobState.cancelled)

    def test_simulation_status_and_pause_resume_use_runtime_backend(self) -> None:
        blocker = threading.Event()

        class PausableRuntimeBackend(FakeRuntimeBackend):
            def run_job(self, spec, on_event, cancel_event):
                self.started_specs.append(spec)
                on_event(
                    SimulationStreamMessage(
                        frame=1,
                        timestamp=time.time(),
                        actors=[],
                    )
                )
                blocker.wait(timeout=1)
                return RuntimeExecutionResult(
                    state=JobState.succeeded,
                    run_id=f"run-{spec.job_id}",
                    manifest_path=str(Path(spec.output_dir) / "run" / "manifest.json"),
                    recording_path=str(Path(spec.output_dir) / "run" / "recording.mp4"),
                )

        base = self.make_service()
        service = OrchestratorService(settings=base.settings, runtime_backend=PausableRuntimeBackend())
        response = service.submit_job(sample_request())

        deadline = time.time() + 3
        while time.time() < deadline:
            status = service.simulation_status(response.job_id)
            if status["is_running"]:
                break
            time.sleep(0.05)

        paused = service.pause_job(response.job_id)
        self.assertTrue(paused["is_paused"])
        self.assertEqual(paused["status"], "paused")

        resumed = service.resume_job(response.job_id)
        self.assertFalse(resumed["is_paused"])
        self.assertEqual(resumed["status"], "running")
        blocker.set()

    def test_startup_initializes_runtime_pool_once(self) -> None:
        runtime_backend = FakeRuntimeBackend()
        base = self.make_service()
        service = OrchestratorService(settings=base.settings, runtime_backend=runtime_backend)
        service.startup()
        service.startup()
        self.assertEqual(runtime_backend.initialize_calls, 1)


if __name__ == "__main__":
    unittest.main()
