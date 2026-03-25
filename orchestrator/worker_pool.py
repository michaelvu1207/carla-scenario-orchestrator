"""
Temporal-based persistent worker pool for CARLA simulation slots.

Each GPU slot runs a Temporal worker process with a persistent CARLA connection.
The orchestrator starts simulation workflows, which Temporal routes to the correct
slot's worker via task queues. Temporal handles heartbeating, retries, crash recovery.

Architecture:
  Orchestrator → Temporal Client → start_workflow("simulate", slot_task_queue)
                                          │
  Temporal Server (Docker, port 7233) ────┘
                                          │
  Worker per slot ← receives activity ────┘
    └── persistent carla.Client connection
    └── run_simulation activity (heartbeats automatically)
"""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from temporalio import activity, workflow
from temporalio.client import Client as TemporalClient
from temporalio.common import RetryPolicy
from temporalio.worker import Worker as TemporalWorker

from .carla_runner.dataset_repository import normalize_map_name
from .carla_runner.models import SimulationStreamMessage
from .models import JobState, RuntimeExecutionResult
from .scheduler import GpuLease, GpuScheduler

try:
    from prometheus_client import Counter, Histogram, Gauge
    JOBS_TOTAL = Counter('carla_jobs_total', 'Total simulation jobs', ['status'])
    JOB_DURATION = Histogram('carla_job_duration_seconds', 'Job wall-clock duration',
                             buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120])
    WORKER_RESTARTS = Counter('carla_worker_restarts_total', 'Worker process restarts', ['slot'])
    SLOT_BUSY = Gauge('carla_slot_busy', 'Whether slot is running a job', ['slot'])
except ImportError:
    JOBS_TOTAL = JOB_DURATION = WORKER_RESTARTS = SLOT_BUSY = None

logger = logging.getLogger(__name__)

TEMPORAL_HOST = "localhost:7233"
TASK_QUEUE_PREFIX = "carla-slot-"

# Health check constants
MAX_RESTART_FAILURES = 3          # Mark unhealthy after this many consecutive failures
BACKOFF_BASE_SECONDS = 30         # Initial backoff delay
BACKOFF_MAX_SECONDS = 300         # Max backoff (5 minutes)


def task_queue_for_slot(slot_index: int) -> str:
    return f"{TASK_QUEUE_PREFIX}{slot_index}"


# ---------------------------------------------------------------------------
# Activity: run a CARLA simulation on a specific slot
# ---------------------------------------------------------------------------

# Global state per worker process (set during worker startup)
_worker_carla_client = None
_worker_carla_world = None
_worker_slot_index = None


@activity.defn
async def run_simulation_activity(input: dict) -> dict:
    """Run a CARLA simulation. Called by Temporal on the slot's worker."""
    import threading
    import requests

    # Push events to orchestrator in real-time via HTTP callback
    job_id = input["job_id"]
    orch_url = f"http://127.0.0.1:18421/api/jobs/{job_id}/events"

    class EventPusher:
        def __init__(self):
            self._batch: list[dict] = []
            self._batch_size = 20
            self._log = logging.getLogger(f"event-pusher-{job_id}")

        def put(self, envelope: dict) -> None:
            self._batch.append(envelope)
            if len(self._batch) >= self._batch_size:
                self._flush()
            # Heartbeat Temporal on every event to prevent timeout during encoding
            activity.heartbeat("frame")

        def _flush(self) -> None:
            if not self._batch:
                return
            batch_to_send = list(self._batch)
            self._batch = []
            for attempt in range(2):
                try:
                    requests.post(orch_url, json=batch_to_send, timeout=5)
                    return
                except Exception as exc:
                    if attempt == 0:
                        time.sleep(1)  # Retry once after 1s
                    else:
                        self._log.warning(
                            "Dropped %d events after 2 attempts: %s",
                            len(batch_to_send), exc,
                        )

        def close(self) -> None:
            self._flush()

    stop_event = threading.Event()
    pause_event = threading.Event()
    heartbeat_stop = threading.Event()
    collector = EventPusher()

    # Background heartbeat thread — keeps activity alive during long encoding phases
    def _heartbeat_loop():
        while not heartbeat_stop.is_set():
            try:
                activity.heartbeat("alive")
            except Exception:
                pass
            heartbeat_stop.wait(timeout=15)  # Heartbeat every 15s
    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat")
    heartbeat_thread.start()

    def _run_sim():
        _simulation_worker(
            input["request_payload"],
            input["runtime_settings"],
            collector,
            stop_event,
            pause_event,
            carla_client=_worker_carla_client,
        )
        collector.close()  # Flush remaining events
        heartbeat_stop.set()

    # Import and run synchronously in thread (Temporal activities support sync)
    from .carla_runner.simulation_service import _simulation_worker

    # Run _simulation_worker in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_sim)

    # Read manifest for result
    output_dir = Path(input["runtime_settings"]["output_root"])
    manifests = sorted(
        output_dir.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if manifests:
        manifest = json.loads(manifests[0].read_text())
        worker_error = manifest.get("worker_error")
        return {
            "state": "failed" if worker_error else "succeeded",
            "error": worker_error,
            "run_id": manifests[0].parent.name,
            "manifest_path": str(manifests[0]),
            "recording_path": manifest.get("recording_path"),
            "scenario_log_path": manifest.get("scenario_log"),
            "debug_log_path": manifest.get("debug_log"),
        }
    else:
        return {"state": "failed", "error": "No manifest produced."}


@activity.defn
async def preload_map_activity(map_name: str) -> str:
    """Pre-load a map on this slot's CARLA instance."""
    global _worker_carla_client
    if _worker_carla_client is None:
        return "error: no CARLA connection"

    import carla
    loop = asyncio.get_running_loop()

    def _load():
        current = normalize_map_name(_worker_carla_client.get_world().get_map().name)
        if current != normalize_map_name(map_name):
            _worker_carla_client.load_world(map_name)
            time.sleep(1.0)
        return normalize_map_name(_worker_carla_client.get_world().get_map().name)

    loaded_map = await loop.run_in_executor(None, _load)
    return loaded_map


# ---------------------------------------------------------------------------
# Workflow: wraps the simulation activity with retry policy
# ---------------------------------------------------------------------------

@workflow.defn(sandboxed=False)
class SimulationWorkflow:
    @workflow.run
    async def run(self, input: dict) -> dict:
        return await workflow.execute_activity(
            run_simulation_activity,
            input,
            start_to_close_timeout=timedelta(minutes=20),
            heartbeat_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_attempts=2,  # 1 retry on failure
                non_retryable_error_types=["NonRetryableError"],
            ),
        )


@workflow.defn(sandboxed=False)
class PreloadMapWorkflow:
    @workflow.run
    async def run(self, map_name: str) -> str:
        return await workflow.execute_activity(
            preload_map_activity,
            map_name,
            start_to_close_timeout=timedelta(minutes=2),
        )


# ---------------------------------------------------------------------------
# Worker process: one per GPU slot
# ---------------------------------------------------------------------------

def _slot_worker_main(
    slot_index: int,
    carla_port: int,
    tm_port: int,
    device_id: str,
) -> None:
    """Long-lived worker process for a single GPU slot."""
    global _worker_carla_client, _worker_carla_world, _worker_slot_index

    logging.basicConfig(level=logging.INFO, format=f"[slot-{slot_index}] %(message)s")
    log = logging.getLogger(f"slot-{slot_index}")

    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    _worker_slot_index = slot_index

    log.info(f"Starting, CARLA port={carla_port}, TM port={tm_port}")

    # Connect to CARLA (persistent — survives across activities)
    try:
        import carla
        _worker_carla_client = carla.Client("127.0.0.1", carla_port)
        _worker_carla_client.set_timeout(30.0)
        _worker_carla_world = _worker_carla_client.get_world()
        current_map = normalize_map_name(_worker_carla_world.get_map().name)
        # TM is NOT pre-warmed here — simulation_service creates it fresh each run.
        # Pre-warming would bind the port, blocking simulation_service from using it.
        log.info(f"CARLA connected, map={current_map}")
    except Exception as exc:
        log.error(f"CARLA connection failed: {exc}")
        return

    # Run Temporal worker (blocking — listens on slot-specific task queue)
    async def _run_worker():
        client = await TemporalClient.connect(TEMPORAL_HOST)
        task_queue = task_queue_for_slot(slot_index)
        log.info(f"Temporal worker starting on queue '{task_queue}'")

        worker = TemporalWorker(
            client,
            task_queue=task_queue,
            workflows=[SimulationWorkflow, PreloadMapWorkflow],
            activities=[run_simulation_activity, preload_map_activity],
            max_concurrent_activities=1,
            max_cached_workflows=2,
            graceful_shutdown_timeout=timedelta(seconds=30),
        )
        await worker.run()

    asyncio.run(_run_worker())


# ---------------------------------------------------------------------------
# WorkerPool: manages worker processes + integrates with orchestrator
# ---------------------------------------------------------------------------

class WorkerPool:
    """Manages Temporal worker processes for CARLA simulation slots."""

    def __init__(self, settings: Any, scheduler: GpuScheduler) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._workers: dict[int, multiprocessing.Process] = {}
        self._worker_slots: dict[int, GpuLease] = {}
        self._temporal_client: TemporalClient | None = None
        self._ready_slots: set[int] = set()
        # Health check state: consecutive failure counts and next retry times
        self._restart_failures: dict[int, int] = {}
        self._next_retry_time: dict[int, float] = {}

    async def start(self) -> None:
        """Start all workers and connect to Temporal."""
        # Connect from the parent for dispatching workflows
        self._temporal_client = await TemporalClient.connect(TEMPORAL_HOST)
        logger.info("Connected to Temporal server")

        # Use 'spawn' to avoid fork-related issues with Temporal's Rust runtime
        ctx = multiprocessing.get_context('spawn')

        for slot in self._scheduler.slots():
            if slot.role != "execution":
                continue
            self._start_worker(slot, ctx)

        logger.info(f"WorkerPool started with {len(self._workers)} workers")

    def _start_worker(self, slot: GpuLease, ctx=None) -> None:
        if ctx is None:
            ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(
            target=_slot_worker_main,
            args=(slot.slot_index, slot.carla_rpc_port, slot.traffic_manager_port, slot.device_id),
            daemon=True,
            name=f"carla-slot-{slot.slot_index}",
        )
        p.start()
        self._workers[slot.slot_index] = p
        self._worker_slots[slot.slot_index] = slot
        self._ready_slots.add(slot.slot_index)
        logger.info(f"Started worker for slot {slot.slot_index} (PID {p.pid})")

    async def dispatch_job(
        self,
        slot_index: int,
        job_id: str,
        request_payload: dict,
        runtime_settings: dict,
        on_event: Callable[[SimulationStreamMessage], None],
    ) -> RuntimeExecutionResult:
        """Start a simulation workflow on the specified slot."""
        if SLOT_BUSY:
            SLOT_BUSY.labels(slot=str(slot_index)).set(1)

        start_time = time.time()
        task_queue = task_queue_for_slot(slot_index)

        # Reuse the shared Temporal client (thread-safe, multiplexes gRPC)
        client = self._temporal_client
        if client is None:
            raise RuntimeError("WorkerPool not started — no Temporal client")

        try:
            # Start workflow
            handle = await client.start_workflow(
                SimulationWorkflow.run,
                {
                    "job_id": job_id,
                    "request_payload": request_payload,
                    "runtime_settings": runtime_settings,
                },
                id=f"sim-{job_id}",
                task_queue=task_queue,
            )

            # Wait for result
            output: dict = await handle.result()

            elapsed = time.time() - start_time
            if JOB_DURATION:
                JOB_DURATION.observe(elapsed)

            # Events were already pushed in real-time via HTTP callback during activity execution
            state = JobState.succeeded if output.get("state", "failed") == "succeeded" else JobState.failed
            if JOBS_TOTAL:
                JOBS_TOTAL.labels(status=output.get("state", "failed")).inc()

            # Update slot map
            try:
                import carla
                c = carla.Client("127.0.0.1", self._worker_slots[slot_index].carla_rpc_port)
                c.set_timeout(5.0)
                actual_map = normalize_map_name(c.get_world().get_map().name)
                self._scheduler.set_slot_map(slot_index, actual_map)
            except Exception:
                pass

            return RuntimeExecutionResult(
                state=state,
                error=output.get("error"),
                run_id=output.get("run_id"),
                manifest_path=output.get("manifest_path"),
                recording_path=output.get("recording_path"),
                scenario_log_path=output.get("scenario_log_path"),
                debug_log_path=output.get("debug_log_path"),
            )

        except Exception as exc:
            if JOBS_TOTAL:
                JOBS_TOTAL.labels(status="failed").inc()
            return RuntimeExecutionResult(state=JobState.failed, error=str(exc))
        finally:
            if SLOT_BUSY:
                SLOT_BUSY.labels(slot=str(slot_index)).set(0)

    async def dispatch_preload(self, slot_index: int, map_name: str) -> str | None:
        """Pre-load a map on an idle slot."""
        if self._temporal_client is None:
            return None
        task_queue = task_queue_for_slot(slot_index)
        try:
            handle = await self._temporal_client.start_workflow(
                PreloadMapWorkflow.run,
                map_name,
                id=f"preload-{slot_index}-{int(time.time())}",
                task_queue=task_queue,
            )
            return await handle.result()
        except Exception as exc:
            logger.error(f"Preload failed on slot {slot_index}: {exc}")
            return None

    def check_workers(self) -> None:
        """Check worker process health, restart crashed ones with backoff."""
        import subprocess as sp

        now = time.time()
        for slot_index, proc in list(self._workers.items()):
            if proc.is_alive():
                # Worker is healthy — reset failure counter
                if self._restart_failures.get(slot_index, 0) > 0:
                    logger.info(f"Worker for slot {slot_index} recovered after restart")
                    self._restart_failures[slot_index] = 0
                    self._next_retry_time.pop(slot_index, None)
                    self._scheduler.mark_slot_ready(slot_index)
                continue

            # Worker is dead — check backoff
            failures = self._restart_failures.get(slot_index, 0)
            next_retry = self._next_retry_time.get(slot_index, 0)

            # Still in backoff period — skip
            if now < next_retry:
                continue

            failures += 1
            self._restart_failures[slot_index] = failures

            if WORKER_RESTARTS:
                WORKER_RESTARTS.labels(slot=str(slot_index)).inc()

            # After MAX_RESTART_FAILURES consecutive failures, mark unhealthy
            if failures >= MAX_RESTART_FAILURES:
                logger.error(
                    f"Worker for slot {slot_index} failed {failures} times consecutively, "
                    f"marking unhealthy"
                )
                self._scheduler.mark_slot_unhealthy(
                    slot_index,
                    error=f"Worker failed {failures} times, CARLA may be dead",
                )
                # Set long backoff before next attempt
                backoff = min(BACKOFF_BASE_SECONDS * (2 ** failures), BACKOFF_MAX_SECONDS)
                self._next_retry_time[slot_index] = now + backoff
                continue

            logger.warning(
                f"Worker for slot {slot_index} died (exit={proc.exitcode}), "
                f"attempt {failures}/{MAX_RESTART_FAILURES}"
            )

            slot = self._worker_slots.get(slot_index)
            if slot:
                # On 2nd+ failure, try restarting the Docker container
                if failures >= 2:
                    container = f"carla-orch-slot-{slot_index}"
                    try:
                        r = sp.run(
                            ["docker", "inspect", "--format", "{{.State.Running}}", container],
                            capture_output=True, text=True, timeout=5,
                        )
                        if r.returncode != 0 or r.stdout.strip().lower() != "true":
                            logger.info(f"Restarting Docker container {container}")
                            sp.run(["docker", "restart", container], capture_output=True, timeout=60)
                            time.sleep(10)
                    except Exception:
                        pass

                self._start_worker(slot, multiprocessing.get_context('spawn'))

            # Exponential backoff for next retry
            backoff = min(BACKOFF_BASE_SECONDS * (2 ** (failures - 1)), BACKOFF_MAX_SECONDS)
            self._next_retry_time[slot_index] = now + backoff

    def stop(self) -> None:
        for proc in self._workers.values():
            proc.terminate()
        for proc in self._workers.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
        logger.info("WorkerPool stopped")
