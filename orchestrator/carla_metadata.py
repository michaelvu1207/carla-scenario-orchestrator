from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

from .carla_runner.dataset_repository import (
    build_runtime_road_summaries,
    dataset_lane_type_counts,
    list_supported_maps,
    normalize_map_name,
)
from .carla_runner.models import (
    CarlaMapInfo,
    CarlaStatusResponse,
    RuntimeMapResponse,
    RuntimeRoadSegment,
)

logger = logging.getLogger(__name__)


def _require_carla():
    try:
        import carla  # type: ignore

        return carla
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CARLA Python API is not available in this environment. "
            "Run the backend in a venv that already has a matching carla package."
        ) from exc


def _make_client(host: str, port: int, timeout: float) -> Any:
    carla = _require_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def _build_waypoint_index(
    world: Any,
    distance: float = 5.0,
    allowed_lane_types: set[str] | None = None,
) -> dict[int, list[Any]]:
    buckets: dict[int, list[Any]] = defaultdict(list)
    waypoints = world.get_map().generate_waypoints(distance)
    for waypoint in waypoints:
        if getattr(waypoint, "lane_type", None) is not None:
            lane_type = str(waypoint.lane_type).split(".")[-1]
            if allowed_lane_types is not None and lane_type not in allowed_lane_types:
                continue
        buckets[int(waypoint.road_id)].append(waypoint)
    for points in buckets.values():
        points.sort(key=lambda item: float(getattr(item, "s", 0.0)))
    return buckets


def _carla_to_frontend(location: Any, rotation: Any) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": -float(location.y),
        "z": float(location.z),
        "yaw": -float(rotation.yaw),
    }


def _drivable_adjacent_lane_id(waypoint: Any, side: str) -> int | None:
    candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if candidate is None:
        return None
    lane_type = str(getattr(candidate, "lane_type", "")).split(".")[-1]
    if lane_type not in {"Driving", "Bidirectional"}:
        return None
    return int(getattr(candidate, "lane_id", 0))


class CarlaMetadataService:
    """Read-only CARLA client for metadata queries (status, maps, blueprints)."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = host or os.environ.get("ORCH_CARLA_METADATA_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("ORCH_CARLA_METADATA_PORT", "2000"))
        self.timeout = timeout or float(os.environ.get("ORCH_CARLA_METADATA_TIMEOUT", "20"))

    def _client(self) -> Any:
        return _make_client(self.host, self.port, self.timeout)

    def get_status(self) -> CarlaStatusResponse:
        warnings: list[str] = []
        try:
            client = self._client()
            world = client.get_world()
            current_map = world.get_map().name
            supported = set(list_supported_maps())
            available = []
            for item in client.get_available_maps():
                normalized = normalize_map_name(item)
                available.append(
                    CarlaMapInfo(
                        name=item,
                        normalized_name=normalized,
                        supported_in_dataset=normalized in supported,
                    )
                )
            return CarlaStatusResponse(
                connected=True,
                current_map=current_map,
                normalized_map_name=normalize_map_name(current_map),
                server_version=client.get_server_version(),
                client_version=client.get_client_version(),
                available_maps=available,
                supported_dataset_maps=sorted(supported),
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CARLA metadata connection failed: %s", exc)
            warnings.append(str(exc))
            return CarlaStatusResponse(
                connected=False,
                supported_dataset_maps=sorted(list_supported_maps()),
                warnings=warnings,
            )

    def load_map(self, map_name: str) -> CarlaStatusResponse:
        client = self._client()
        current = client.get_world().get_map().name
        if normalize_map_name(current) != normalize_map_name(map_name):
            client.load_world(map_name)
            time.sleep(1.0)
        return self.get_status()

    def get_runtime_map(self) -> RuntimeMapResponse:
        client = self._client()
        world = client.get_world()
        map_name = world.get_map().name
        normalized_map_name = normalize_map_name(map_name)
        waypoint_index = _build_waypoint_index(
            world,
            distance=5.0,
            allowed_lane_types={
                "Driving",
                "Bidirectional",
                "Parking",
                "Shoulder",
                "Sidewalk",
                "Biking",
            },
        )
        segments: list[RuntimeRoadSegment] = []
        lane_type_counts: dict[str, int] = defaultdict(int)
        for road_id, waypoints in waypoint_index.items():
            grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
            for waypoint in waypoints:
                grouped[(int(getattr(waypoint, "section_id", 0)), int(waypoint.lane_id))].append(waypoint)
            for (section_id, lane_id), lane_waypoints in grouped.items():
                line = []
                for waypoint in lane_waypoints:
                    coords = _carla_to_frontend(waypoint.transform.location, waypoint.transform.rotation)
                    line.append(
                        {
                            "x": coords["x"],
                            "y": coords["y"],
                            "z": coords["z"],
                            "yaw": coords["yaw"],
                            "s": float(getattr(waypoint, "s", 0.0)),
                        }
                    )
                if len(line) < 2:
                    continue
                lane_type = str(getattr(lane_waypoints[0], "lane_type", "")).split(".")[-1]
                lane_type_counts[lane_type.lower()] += 1
                segments.append(
                    RuntimeRoadSegment(
                        id=f"road-{road_id}-section-{section_id}-lane-{lane_id}",
                        road_id=road_id,
                        section_id=section_id,
                        lane_id=lane_id,
                        lane_type=lane_type,
                        is_junction=bool(getattr(lane_waypoints[0], "is_junction", False)),
                        left_lane_id=_drivable_adjacent_lane_id(lane_waypoints[0], "left"),
                        right_lane_id=_drivable_adjacent_lane_id(lane_waypoints[0], "right"),
                        centerline=line,
                    )
                )
        road_summaries = build_runtime_road_summaries(map_name)
        dataset_counts = dataset_lane_type_counts(map_name)
        return RuntimeMapResponse(
            map_name=map_name,
            normalized_map_name=normalized_map_name,
            road_segments=segments,
            lane_type_counts=dict(sorted(lane_type_counts.items())),
            dataset_lane_type_counts=dataset_counts,
            road_summaries=road_summaries,
            dataset_augmented=bool(road_summaries),
        )

    def list_blueprints(self) -> dict[str, list[str]]:
        client = self._client()
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        return {
            "vehicles": sorted(bp.id for bp in blueprint_library.filter("vehicle.*")),
            "walkers": sorted(bp.id for bp in blueprint_library.filter("walker.pedestrian.*")),
        }
