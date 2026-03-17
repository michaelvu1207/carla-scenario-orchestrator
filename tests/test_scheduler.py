from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.scheduler import GpuScheduler


def make_settings(gpu_devices=("0", "1", "2"), metadata_slot_index=2) -> Settings:
    repo_root = Path("/tmp/carla-scenario-orchestrator-tests")
    return Settings(
        repo_root=repo_root,
        jobs_root=repo_root / "runs",
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
        metadata_slot_index=metadata_slot_index,
        carla_metadata_host="127.0.0.1",
        carla_metadata_port=2000 + metadata_slot_index * 100,
        carla_metadata_timeout=20,
        storage_bucket=None,
        storage_region="us-east-1",
        storage_prefix="runs",
    )


class GpuSchedulerTests(unittest.TestCase):
    def test_allocates_unique_ports_per_execution_gpu(self) -> None:
        scheduler = GpuScheduler(make_settings())
        lease_a = scheduler.acquire("job-a", threading.Event())
        lease_b = scheduler.acquire("job-b", threading.Event())
        self.assertNotEqual(lease_a.device_id, lease_b.device_id)
        self.assertEqual(lease_a.carla_rpc_port, 2000)
        self.assertEqual(lease_b.carla_rpc_port, 2100)
        self.assertEqual(lease_a.role, "execution")
        self.assertEqual(lease_b.role, "execution")
        scheduler.release("job-a")
        scheduler.release("job-b")

    def test_reserves_metadata_slot_from_job_leasing(self) -> None:
        scheduler = GpuScheduler(make_settings())
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.total_slots, 2)
        self.assertEqual(snapshot.metadata_slots, 1)
        self.assertTrue(snapshot.metadata_ready)
        self.assertEqual(snapshot.metadata_slot_index, 2)
        metadata_slots = [slot for slot in snapshot.slots if slot.role == "metadata"]
        self.assertEqual(len(metadata_slots), 1)
        self.assertEqual(metadata_slots[0].container_name, "carla-orch-metadata")

    def test_waits_until_execution_slot_is_free(self) -> None:
        scheduler = GpuScheduler(make_settings(gpu_devices=("0", "1"), metadata_slot_index=1))
        scheduler.acquire("job-a", threading.Event())
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


if __name__ == "__main__":
    unittest.main()
