from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.scheduler import GpuScheduler


def make_settings() -> Settings:
    repo_root = Path("/tmp/carla-scenario-orchestrator-tests")
    return Settings(
        repo_root=repo_root,
        jobs_root=repo_root / "runs",
        gpu_devices=("0", "1"),
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
        storage_bucket=None,
        storage_region="us-east-1",
        storage_prefix="runs",
    )


class GpuSchedulerTests(unittest.TestCase):
    def test_allocates_unique_ports_per_gpu(self) -> None:
        scheduler = GpuScheduler(make_settings())
        lease_a = scheduler.acquire("job-a", threading.Event())
        lease_b = scheduler.acquire("job-b", threading.Event())
        self.assertNotEqual(lease_a.device_id, lease_b.device_id)
        self.assertEqual(lease_a.carla_rpc_port, 2000)
        self.assertEqual(lease_b.carla_rpc_port, 2100)
        scheduler.release("job-a")
        scheduler.release("job-b")

    def test_waits_until_slot_is_free(self) -> None:
        settings = make_settings()
        scheduler = GpuScheduler(
            Settings(
                repo_root=settings.repo_root,
                jobs_root=settings.jobs_root,
                gpu_devices=("0",),
                carla_image=settings.carla_image,
                carla_container_prefix=settings.carla_container_prefix,
                carla_startup_timeout_seconds=settings.carla_startup_timeout_seconds,
                carla_rpc_port_base=settings.carla_rpc_port_base,
                traffic_manager_port_base=settings.traffic_manager_port_base,
                port_stride=settings.port_stride,
                carla_timeout_seconds=settings.carla_timeout_seconds,
                python_executable=settings.python_executable,
                docker_network_mode=settings.docker_network_mode,
                carla_start_command_template=settings.carla_start_command_template,
                carla_metadata_host=settings.carla_metadata_host,
                carla_metadata_port=settings.carla_metadata_port,
                carla_metadata_timeout=settings.carla_metadata_timeout,
                storage_bucket=None,
                storage_region="us-east-1",
                storage_prefix="runs",
            )
        )
        first = scheduler.acquire("job-a", threading.Event())
        acquired: list[str] = []

        def worker() -> None:
            lease = scheduler.acquire("job-b", threading.Event())
            acquired.append(lease.device_id)
            scheduler.release("job-b")

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.2)
        self.assertEqual(acquired, [])
        scheduler.release("job-a")
        thread.join(timeout=1)
        self.assertEqual(acquired, ["0"])
        scheduler.release("job-a")


if __name__ == "__main__":
    unittest.main()
