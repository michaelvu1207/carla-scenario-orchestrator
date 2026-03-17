from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Settings
from .models import CapacityResponse, CapacitySlot, GpuLeaseInfo


@dataclass(frozen=True)
class GpuLease:
    slot_index: int
    device_id: str
    role: str
    container_name: str
    carla_rpc_port: int
    traffic_manager_port: int

    def to_model(self) -> GpuLeaseInfo:
        return GpuLeaseInfo(
            slot_index=self.slot_index,
            device_id=self.device_id,
            role=self.role,
            container_name=self.container_name,
            carla_rpc_port=self.carla_rpc_port,
            traffic_manager_port=self.traffic_manager_port,
        )


class GpuScheduler:
    def __init__(self, settings: Settings) -> None:
        self._condition = threading.Condition()
        self._leases_by_job: dict[str, GpuLease] = {}
        self._job_by_slot: dict[int, str] = {}
        self._slots: list[GpuLease] = []
        self._slot_status: dict[int, str] = {}
        self._slot_errors: dict[int, str | None] = {}
        self._metadata_slot_index = settings.metadata_slot_index
        for idx, device_id in enumerate(settings.gpu_devices):
            role = "metadata" if idx == self._metadata_slot_index else "execution"
            container_name = (
                f"{settings.carla_container_prefix}-metadata".lower()
                if role == "metadata"
                else f"{settings.carla_container_prefix}-slot-{idx}".lower()
            )
            slot = GpuLease(
                slot_index=idx,
                device_id=device_id,
                role=role,
                container_name=container_name,
                carla_rpc_port=settings.carla_rpc_port_base + idx * settings.port_stride,
                traffic_manager_port=settings.traffic_manager_port_base + idx * settings.port_stride,
            )
            self._slots.append(slot)
            self._slot_status[idx] = "ready"
            self._slot_errors[idx] = None

    def slots(self) -> list[GpuLease]:
        with self._condition:
            return list(self._slots)

    def metadata_slot(self) -> GpuLease:
        with self._condition:
            return self._slots[self._metadata_slot_index]

    def metadata_status(self) -> tuple[str, str | None]:
        with self._condition:
            return (
                self._slot_status.get(self._metadata_slot_index, "ready"),
                self._slot_errors.get(self._metadata_slot_index),
            )

    def mark_slot_ready(self, slot_index: int) -> None:
        self._set_slot_status(slot_index, "ready")

    def mark_slot_warming(self, slot_index: int) -> None:
        self._set_slot_status(slot_index, "warming")

    def mark_slot_unhealthy(self, slot_index: int, error: str | None = None) -> None:
        self._set_slot_status(slot_index, "unhealthy", error=error)

    def _set_slot_status(self, slot_index: int, status: str, *, error: str | None = None) -> None:
        with self._condition:
            self._slot_status[slot_index] = status
            self._slot_errors[slot_index] = error
            self._condition.notify_all()

    def acquire(self, job_id: str, cancel_event: threading.Event) -> GpuLease:
        with self._condition:
            while True:
                if cancel_event.is_set():
                    raise RuntimeError("Job cancelled before a GPU slot was assigned.")
                for slot in self._slots:
                    if slot.role != "execution":
                        continue
                    if self._slot_status.get(slot.slot_index) != "ready":
                        continue
                    if slot.slot_index in self._job_by_slot:
                        continue
                    self._job_by_slot[slot.slot_index] = job_id
                    self._leases_by_job[job_id] = slot
                    return slot
                self._condition.wait(timeout=0.5)

    def release(self, job_id: str) -> None:
        with self._condition:
            lease = self._leases_by_job.pop(job_id, None)
            if lease is None:
                return
            self._job_by_slot.pop(lease.slot_index, None)
            self._condition.notify_all()

    def queue_position(self, job_id: str, queued_job_ids: list[str]) -> int:
        try:
            return queued_job_ids.index(job_id) + 1
        except ValueError:
            return 0

    def snapshot(self) -> CapacityResponse:
        with self._condition:
            slots: list[CapacitySlot] = []
            metadata_slot: CapacitySlot | None = None
            for slot in self._slots:
                job_id = self._job_by_slot.get(slot.slot_index)
                snapshot_slot = CapacitySlot(
                    slot_index=slot.slot_index,
                    device_id=slot.device_id,
                    role=slot.role,
                    container_name=slot.container_name,
                    busy=job_id is not None,
                    job_id=job_id,
                    status=self._slot_status.get(slot.slot_index, "ready"),
                    status_error=self._slot_errors.get(slot.slot_index),
                    carla_rpc_port=slot.carla_rpc_port,
                    traffic_manager_port=slot.traffic_manager_port,
                )
                slots.append(snapshot_slot)
                if snapshot_slot.role == "metadata":
                    metadata_slot = snapshot_slot
            execution_slots = [slot for slot in slots if slot.role == "execution"]
            busy_slots = sum(1 for slot in execution_slots if slot.busy)
            ready_slots = sum(1 for slot in execution_slots if slot.status == "ready" and not slot.busy)
            unavailable_slots = sum(1 for slot in execution_slots if slot.status != "ready")
            return CapacityResponse(
                total_slots=len(execution_slots),
                busy_slots=busy_slots,
                free_slots=ready_slots,
                ready_slots=ready_slots,
                unavailable_slots=unavailable_slots,
                metadata_slots=1 if metadata_slot is not None else 0,
                metadata_ready=metadata_slot.status == "ready" if metadata_slot is not None else False,
                metadata_slot_index=metadata_slot.slot_index if metadata_slot is not None else None,
                slots=slots,
            )
