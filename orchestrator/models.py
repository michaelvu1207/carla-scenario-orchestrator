from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .carla_runner.models import SimulationRunRequest, SimulationStreamMessage


class JobState(str, Enum):
    queued = "queued"
    starting = "starting"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class GpuLeaseInfo(BaseModel):
    slot_index: int
    device_id: str
    role: str = "execution"
    container_name: str
    carla_rpc_port: int
    traffic_manager_port: int
    current_map: str | None = None


class CapacitySlot(BaseModel):
    slot_index: int
    device_id: str
    role: str = "execution"
    container_name: str
    busy: bool = False
    job_id: str | None = None
    status: str = "ready"
    status_error: str | None = None
    carla_rpc_port: int
    traffic_manager_port: int
    current_map: str | None = None


class CapacityResponse(BaseModel):
    total_slots: int
    busy_slots: int
    free_slots: int
    ready_slots: int
    unavailable_slots: int
    metadata_slots: int = 0
    metadata_ready: bool = False
    metadata_slot_index: int | None = None
    slots: list[CapacitySlot] = Field(default_factory=list)


class JobEvent(BaseModel):
    created_at: datetime
    payload: SimulationStreamMessage


class StoredArtifact(BaseModel):
    kind: str
    label: str | None = None
    local_path: str | None = None
    content_type: str | None = None
    file_ext: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None
    s3_bucket: str | None = None
    s3_key: str | None = None
    s3_uri: str | None = None


class JobArtifacts(BaseModel):
    output_dir: str
    request_file: str
    runtime_settings_file: str
    manifest_path: str | None = None
    recording_path: str | None = None
    scenario_log_path: str | None = None
    debug_log_path: str | None = None
    uploaded_artifacts: list[StoredArtifact] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    request: SimulationRunRequest
    queue_position: int = 0
    gpu: GpuLeaseInfo | None = None
    container_name: str | None = None
    error: str | None = None
    run_id: str | None = None
    events: list[JobEvent] = Field(default_factory=list)
    artifacts: JobArtifacts
    simulation_id: str | None = None


class JobSubmissionResponse(BaseModel):
    job_id: str
    state: JobState
    queue_position: int


class JobListResponse(BaseModel):
    items: list[JobRecord] = Field(default_factory=list)


class CancelJobResponse(BaseModel):
    job_id: str
    state: JobState


class HealthResponse(BaseModel):
    ok: bool = True
    status: str = "healthy"
    total_slots: int
    busy_slots: int
    queued_jobs: int
    carla_connected: bool = False
    metadata_connected: bool = False
    metadata_slot_index: int | None = None
    running: bool = False
    simulation_running: bool = False
    connections: int = 0
    langchain_available: bool = False
    langsmith_available: bool = False
    langsmith_tracing: bool = False



class RuntimeLaunchSpec(BaseModel):
    job_id: str
    request_file: str
    runtime_settings_file: str
    output_dir: str
    gpu: GpuLeaseInfo


class RuntimeExecutionResult(BaseModel):
    state: JobState
    error: str | None = None
    run_id: str | None = None
    manifest_path: str | None = None
    recording_path: str | None = None
    scenario_log_path: str | None = None
    debug_log_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
