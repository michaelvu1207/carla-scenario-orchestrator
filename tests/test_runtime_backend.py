from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import call, patch

from orchestrator.config import Settings
from orchestrator.models import GpuLeaseInfo
from orchestrator.runtime_backend import DockerRuntimeBackend


def make_settings() -> Settings:
    repo_root = Path("/tmp/carla-scenario-orchestrator-tests")
    return Settings(
        repo_root=repo_root,
        jobs_root=repo_root / "runs",
        gpu_devices=("0", "1", "2"),
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
        metadata_slot_index=2,
        carla_metadata_host="127.0.0.1",
        carla_metadata_port=2200,
        carla_metadata_timeout=20,
        storage_bucket=None,
        storage_region="us-east-1",
        storage_prefix="runs",
    )


def make_slot() -> GpuLeaseInfo:
    return GpuLeaseInfo(
        slot_index=0,
        device_id="0",
        role="execution",
        container_name="carla-orch-slot-0",
        carla_rpc_port=2000,
        traffic_manager_port=8000,
    )


class RuntimeBackendTests(unittest.TestCase):
    def test_ensure_slot_container_reuses_running_container(self) -> None:
        backend = DockerRuntimeBackend(make_settings())
        slot = make_slot()
        completed = subprocess.CompletedProcess(
            args=["docker", "inspect"],
            returncode=0,
            stdout="true\n",
            stderr="",
        )

        with patch("orchestrator.runtime_backend.subprocess.run", return_value=completed) as run_mock:
            with patch.object(backend, "_wait_for_tcp") as wait_mock:
                backend._ensure_slot_container(slot)

        run_mock.assert_called_once_with(
            ["docker", "inspect", "--format", "{{.State.Running}}", "carla-orch-slot-0"],
            check=False,
            capture_output=True,
            text=True,
        )
        wait_mock.assert_called_once_with("127.0.0.1", 2000, None)

    def test_ensure_slot_container_recreates_missing_container_with_restart_policy(self) -> None:
        backend = DockerRuntimeBackend(make_settings())
        slot = make_slot()
        inspect_missing = subprocess.CompletedProcess(
            args=["docker", "inspect"],
            returncode=1,
            stdout="",
            stderr="No such container",
        )
        remove_result = subprocess.CompletedProcess(args=["docker", "rm"], returncode=0, stdout="", stderr="")
        start_result = subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="cid\n", stderr="")

        with patch(
            "orchestrator.runtime_backend.subprocess.run",
            side_effect=[inspect_missing, remove_result, start_result],
        ) as run_mock:
            with patch.object(backend, "_wait_for_tcp") as wait_mock:
                backend._ensure_slot_container(slot)

        self.assertEqual(
            run_mock.call_args_list[0],
            call(
                ["docker", "inspect", "--format", "{{.State.Running}}", "carla-orch-slot-0"],
                check=False,
                capture_output=True,
                text=True,
            ),
        )
        self.assertEqual(
            run_mock.call_args_list[1],
            call(
                ["docker", "rm", "-f", "carla-orch-slot-0"],
                check=False,
                capture_output=True,
                text=True,
            ),
        )
        start_call = run_mock.call_args_list[2]
        self.assertIn("--restart", start_call.args[0])
        self.assertIn("unless-stopped", start_call.args[0])
        self.assertIn("carla-orch-slot-0", start_call.args[0])
        self.assertIn("device=0", start_call.args[0])
        wait_mock.assert_called_once_with("127.0.0.1", 2000, None)


if __name__ == "__main__":
    unittest.main()
