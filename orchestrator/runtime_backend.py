from __future__ import annotations

import json
import os
import select
import signal
import socket
import subprocess
import threading
from pathlib import Path
from threading import Event
from typing import Callable, Protocol

from .config import Settings
from .models import JobState, RuntimeExecutionResult, RuntimeLaunchSpec
from .carla_runner.models import SimulationStreamMessage
from .scheduler import GpuLease, GpuScheduler


class RuntimeBackend(Protocol):
    def initialize_pool(self, scheduler: GpuScheduler) -> None: ...

    def run_job(
        self,
        spec: RuntimeLaunchSpec,
        on_event: Callable[[SimulationStreamMessage], None],
        cancel_event: Event,
    ) -> RuntimeExecutionResult: ...

    def pause_job(self, job_id: str) -> bool: ...

    def resume_job(self, job_id: str) -> bool: ...

    def is_job_paused(self, job_id: str) -> bool: ...


class DockerRuntimeBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._worker_processes: dict[str, subprocess.Popen[str]] = {}
        self._paused_jobs: set[str] = set()

    def initialize_pool(self, scheduler: GpuScheduler) -> None:
        for slot in scheduler.slots():
            scheduler.mark_slot_warming(slot.slot_index)
            try:
                self._ensure_slot_container(slot)
            except Exception as exc:
                scheduler.mark_slot_unhealthy(slot.slot_index, str(exc))
            else:
                scheduler.mark_slot_ready(slot.slot_index)

    def run_job(
        self,
        spec: RuntimeLaunchSpec,
        on_event: Callable[[SimulationStreamMessage], None],
        cancel_event: Event,
    ) -> RuntimeExecutionResult:
        self._ensure_slot_container(spec.gpu, cancel_event)
        return self._run_worker(spec, on_event, cancel_event)

    def _docker_env_args(self) -> list[str]:
        return [
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=all",
            "-e",
            "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json",
            "-v",
            "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro",
        ]

    def _ensure_slot_container(self, slot: GpuLease | object, cancel_event: Event | None = None) -> None:
        container_name = getattr(slot, "container_name")
        if not self._is_container_running(container_name):
            self._remove_container(container_name)
            self._start_carla_container(slot)
        self._wait_for_tcp("127.0.0.1", getattr(slot, "carla_rpc_port"), cancel_event)

    def _is_container_running(self, container_name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def _start_carla_container(self, slot: GpuLease | object) -> None:
        rpc_port = getattr(slot, "carla_rpc_port")
        container_name = getattr(slot, "container_name")
        device_id = getattr(slot, "device_id")
        command = self.settings.carla_start_command_template.format(rpc_port=rpc_port)
        cmd = [
            "docker",
            "run",
            "-d",
            "--restart",
            "unless-stopped",
            "--name",
            container_name,
            "--privileged",
            "--gpus",
            f"device={device_id}",
            "--network",
            self.settings.docker_network_mode,
            *self._docker_env_args(),
            self.settings.carla_image,
            "/bin/bash",
            "-lc",
            command,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _remove_container(self, container_name: str) -> None:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
        )

    def _wait_for_tcp(self, host: str, port: int, cancel_event: Event | None) -> None:
        deadline = self.settings.carla_startup_timeout_seconds
        waited = 0.0
        while waited < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Job cancelled while waiting for CARLA to start.")
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError:
                pass
            waited += 1.0
            if cancel_event is None:
                threading.Event().wait(1.0)
            else:
                cancel_event.wait(1.0)
        raise RuntimeError(f"Timed out waiting for CARLA to accept connections on {host}:{port}.")

    def _run_worker(
        self,
        spec: RuntimeLaunchSpec,
        on_event: Callable[[SimulationStreamMessage], None],
        cancel_event: Event,
    ) -> RuntimeExecutionResult:
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.settings.repo_root) if not python_path else f"{self.settings.repo_root}:{python_path}"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            self.settings.python_executable,
            "-m",
            "orchestrator.runner_process",
            "--request-file",
            spec.request_file,
            "--runtime-settings-file",
            spec.runtime_settings_file,
        ]
        process = subprocess.Popen(
            cmd,
            cwd=self.settings.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self._lock:
            self._worker_processes[spec.job_id] = process
        try:
            if process.stdout is None:
                raise RuntimeError("Runner process stdout was not captured.")
            while True:
                if cancel_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                    return RuntimeExecutionResult(state=JobState.cancelled, error="Job cancelled.")
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if ready:
                    line = process.stdout.readline()
                    if line == "":
                        if process.poll() is not None:
                            break
                        continue
                    if line:
                        self._handle_runner_line(line, on_event)
                exit_code = process.poll()
                if exit_code is not None and not ready:
                    break
            if process.returncode != 0:
                return RuntimeExecutionResult(
                    state=JobState.failed,
                    error=f"Runner exited with code {process.returncode}.",
                    extra={"container_name": spec.gpu.container_name},
                )
            return self._build_result(spec.output_dir)
        finally:
            with self._lock:
                self._worker_processes.pop(spec.job_id, None)
                self._paused_jobs.discard(spec.job_id)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    def pause_job(self, job_id: str) -> bool:
        with self._lock:
            process = self._worker_processes.get(job_id)
            if process is None or process.poll() is not None:
                return False
            process.send_signal(signal.SIGUSR1)
            self._paused_jobs.add(job_id)
            return True

    def resume_job(self, job_id: str) -> bool:
        with self._lock:
            process = self._worker_processes.get(job_id)
            if process is None or process.poll() is not None:
                return False
            process.send_signal(signal.SIGUSR2)
            self._paused_jobs.discard(job_id)
            return True

    def is_job_paused(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._paused_jobs

    def _handle_runner_line(self, line: str, on_event: Callable[[SimulationStreamMessage], None]) -> None:
        stripped = line.strip()
        if not stripped:
            return
        try:
            envelope = json.loads(stripped)
        except json.JSONDecodeError:
            return
        if envelope.get("kind") != "stream":
            return
        payload = SimulationStreamMessage.model_validate(envelope.get("payload") or {})
        on_event(payload)

    def _build_result(self, output_dir: str) -> RuntimeExecutionResult:
        manifests = sorted(Path(output_dir).glob("*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not manifests:
            return RuntimeExecutionResult(state=JobState.failed, error="Runner finished without writing a manifest.")
        manifest_path = manifests[0]
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        worker_error = data.get("worker_error")
        return RuntimeExecutionResult(
            state=JobState.failed if worker_error else JobState.succeeded,
            error=worker_error,
            run_id=manifest_path.parent.name,
            manifest_path=str(manifest_path),
            recording_path=data.get("recording_path"),
            scenario_log_path=data.get("scenario_log"),
            debug_log_path=data.get("debug_log"),
        )
