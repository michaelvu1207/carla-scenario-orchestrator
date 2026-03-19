from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - local fallback when deps are not installed yet
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    jobs_root: Path
    gpu_devices: tuple[str, ...]
    carla_image: str
    carla_container_prefix: str
    carla_startup_timeout_seconds: float
    carla_rpc_port_base: int
    traffic_manager_port_base: int
    port_stride: int
    carla_timeout_seconds: float
    python_executable: str
    docker_network_mode: str
    carla_start_command_template: str
    metadata_slot_index: int
    carla_metadata_host: str
    carla_metadata_port: int
    carla_metadata_timeout: float
    storage_bucket: str | None
    storage_region: str
    storage_prefix: str
    warm_metadata_cache_on_startup: bool = True
    webhook_url: str = ""
    webhook_secret: str = ""

    @classmethod
    def load(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(repo_root / ".env", override=False)
        jobs_root = Path(os.environ.get("ORCH_JOBS_ROOT", repo_root / "runs")).resolve()
        jobs_root.mkdir(parents=True, exist_ok=True)
        gpu_devices = tuple(_split_csv(os.environ.get("ORCH_GPU_DEVICES", "0,1,2,3,4,5,6,7")))
        if not gpu_devices:
            raise RuntimeError("ORCH_GPU_DEVICES must contain at least one GPU identifier.")

        carla_rpc_port_base = int(os.environ.get("ORCH_CARLA_RPC_PORT_BASE", "2000"))
        traffic_manager_port_base = int(os.environ.get("ORCH_TRAFFIC_MANAGER_PORT_BASE", "8000"))
        port_stride = int(os.environ.get("ORCH_PORT_STRIDE", "100"))
        default_metadata_slot_index = len(gpu_devices) - 1
        metadata_slot_index = int(os.environ.get("ORCH_METADATA_SLOT_INDEX", str(default_metadata_slot_index)))
        if metadata_slot_index < 0 or metadata_slot_index >= len(gpu_devices):
            raise RuntimeError(
                f"ORCH_METADATA_SLOT_INDEX must be between 0 and {len(gpu_devices) - 1}."
            )
        carla_metadata_port = carla_rpc_port_base + metadata_slot_index * port_stride

        return cls(
            repo_root=repo_root,
            jobs_root=jobs_root,
            gpu_devices=gpu_devices,
            carla_image=os.environ.get("ORCH_CARLA_IMAGE", "carlasim/carla:0.9.16-phase3"),
            carla_container_prefix=os.environ.get("ORCH_CARLA_CONTAINER_PREFIX", "carla-orch"),
            carla_startup_timeout_seconds=float(os.environ.get("ORCH_CARLA_STARTUP_TIMEOUT", "90")),
            carla_rpc_port_base=carla_rpc_port_base,
            traffic_manager_port_base=traffic_manager_port_base,
            port_stride=port_stride,
            carla_timeout_seconds=float(os.environ.get("ORCH_CARLA_TIMEOUT", "20")),
            python_executable=os.environ.get("ORCH_PYTHON_EXECUTABLE", sys.executable),
            docker_network_mode=os.environ.get("ORCH_DOCKER_NETWORK_MODE", "host"),
            carla_start_command_template=os.environ.get(
                "ORCH_CARLA_START_COMMAND",
                "./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={rpc_port}",
            ),
            metadata_slot_index=metadata_slot_index,
            carla_metadata_host=os.environ.get("ORCH_CARLA_METADATA_HOST", "127.0.0.1"),
            carla_metadata_port=carla_metadata_port,
            carla_metadata_timeout=float(os.environ.get("ORCH_CARLA_METADATA_TIMEOUT", "20")),
            warm_metadata_cache_on_startup=os.environ.get("ORCH_WARM_METADATA_CACHE_ON_STARTUP", "true").lower()
            not in {"0", "false", "no"},
            webhook_url=os.environ.get("ORCH_WEBHOOK_URL", ""),
            webhook_secret=os.environ.get("ORCH_WEBHOOK_SECRET", ""),
            storage_bucket=(os.environ.get("ORCH_STORAGE_BUCKET") or "").strip() or None,
            storage_region=os.environ.get("ORCH_STORAGE_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1",
            storage_prefix=(os.environ.get("ORCH_STORAGE_PREFIX") or "runs").strip("/"),
        )
