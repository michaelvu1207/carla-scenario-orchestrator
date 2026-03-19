from __future__ import annotations

import copy
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any

from .carla_runner.dataset_repository import (
    build_runtime_road_summaries,
    dataset_lane_type_counts,
    normalize_map_name,
)
from .carla_runner.models import (
    CarlaMapInfo,
    CarlaStatusResponse,
    RuntimeMapResponse,
    RuntimeRoadSegment,
)
from .generated_map import build_generated_map

logger = logging.getLogger(__name__)

_STATUS_CACHE_TTL_SECONDS = 300.0


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
        self._status_cache_ttl = float(
            os.environ.get("ORCH_CARLA_STATUS_CACHE_TTL", str(_STATUS_CACHE_TTL_SECONDS))
        )
        self._lock = threading.Lock()
        self._status_cache: tuple[float, CarlaStatusResponse] | None = None
        self._status_refresh_in_flight = False
        self._blueprints_cache: dict[str, list[str]] | None = None
        self._runtime_map_cache: dict[str, RuntimeMapResponse] = {}
        self._map_xodr_cache: dict[str, str] = {}
        self._generated_map_cache: dict[str, dict[str, Any]] = {}
        self._last_xodr_map_name: str | None = None

    def _client(self) -> Any:
        return _make_client(self.host, self.port, self.timeout)

    def _invalidate_status_cache(self) -> None:
        with self._lock:
            self._status_cache = None

    def _set_status_cache(self, status: CarlaStatusResponse) -> None:
        with self._lock:
            self._status_cache = (time.time(), status)

    def _get_status_cache(self, allow_stale: bool = False) -> CarlaStatusResponse | None:
        with self._lock:
            if self._status_cache is None:
                return None
            cached_at, status = self._status_cache
            if not allow_stale and time.time() - cached_at > self._status_cache_ttl:
                return None
            return status.model_copy(deep=True)

    def _get_cached_current_map_name(self) -> str | None:
        cached = self._get_status_cache()
        return cached.current_map if cached and cached.connected else None

    def warm_cache(self) -> None:
        try:
            self.get_status(force_refresh=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to warm CARLA metadata cache: %s", exc)
        try:
            self.list_blueprints()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to warm CARLA blueprint cache: %s", exc)

    def _refresh_status_cache_async(self) -> None:
        with self._lock:
            if self._status_refresh_in_flight:
                return
            self._status_refresh_in_flight = True

        def _worker() -> None:
            try:
                self.get_status(force_refresh=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to refresh CARLA metadata cache: %s", exc)
            finally:
                with self._lock:
                    self._status_refresh_in_flight = False

        threading.Thread(target=_worker, daemon=True).start()

    def get_status(self, force_refresh: bool = False) -> CarlaStatusResponse:
        warnings: list[str] = []
        stale_cached = self._get_status_cache(allow_stale=True)
        if not force_refresh:
            cached = self._get_status_cache()
            if cached is not None:
                return cached
            if stale_cached is not None:
                self._refresh_status_cache_async()
                return stale_cached
        try:
            client = self._client()
            world = client.get_world()
            current_map = world.get_map().name
            available = []
            for item in client.get_available_maps():
                normalized = normalize_map_name(item)
                available.append(
                    CarlaMapInfo(
                        name=item,
                        normalized_name=normalized,
                    )
                )
            status = CarlaStatusResponse(
                connected=True,
                current_map=current_map,
                normalized_map_name=normalize_map_name(current_map),
                server_version=client.get_server_version(),
                client_version=client.get_client_version(),
                available_maps=available,
                warnings=warnings,
            )
            self._set_status_cache(status)
            return status.model_copy(deep=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CARLA metadata connection failed: %s", exc)
            if stale_cached is not None:
                warning_message = f"Returning cached CARLA metadata after refresh failure: {exc}"
                return stale_cached.model_copy(
                    deep=True,
                    update={"warnings": [*stale_cached.warnings, warning_message]},
                )
            warnings.append(str(exc))
            return CarlaStatusResponse(
                connected=False,
                warnings=warnings,
            )

    def load_map(self, map_name: str) -> CarlaStatusResponse:
        client = self._client()
        current = client.get_world().get_map().name
        if normalize_map_name(current) != normalize_map_name(map_name):
            client.load_world(map_name)
            time.sleep(1.0)
        self._invalidate_status_cache()
        return self.get_status(force_refresh=True)

    def get_runtime_map(self, force_refresh: bool = False) -> RuntimeMapResponse:
        cached_map_name = normalize_map_name(self._get_cached_current_map_name() or "")
        if cached_map_name and not force_refresh:
            with self._lock:
                cached_runtime = self._runtime_map_cache.get(cached_map_name)
            if cached_runtime is not None:
                return cached_runtime.model_copy(deep=True)

        client = self._client()
        world = client.get_world()
        map_name = world.get_map().name
        normalized_map_name = normalize_map_name(map_name)
        if not force_refresh:
            with self._lock:
                cached_runtime = self._runtime_map_cache.get(normalized_map_name)
            if cached_runtime is not None:
                return cached_runtime.model_copy(deep=True)

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
        runtime_map = RuntimeMapResponse(
            map_name=map_name,
            normalized_map_name=normalized_map_name,
            road_segments=segments,
            lane_type_counts=dict(sorted(lane_type_counts.items())),
            dataset_lane_type_counts=dataset_counts,
            road_summaries=road_summaries,
            dataset_augmented=bool(road_summaries),
        )
        with self._lock:
            self._runtime_map_cache[normalized_map_name] = runtime_map
        return runtime_map.model_copy(deep=True)

    def get_map_xodr(self, force_refresh: bool = False) -> str:
        cached_map_name = normalize_map_name(self._get_cached_current_map_name() or "")
        if cached_map_name and not force_refresh:
            with self._lock:
                cached_xodr = self._map_xodr_cache.get(cached_map_name)
            if cached_xodr is not None:
                return cached_xodr
        if not cached_map_name and not force_refresh:
            with self._lock:
                last_cached_map_name = self._last_xodr_map_name
                cached_xodr = self._map_xodr_cache.get(last_cached_map_name or "")
            if cached_xodr is not None:
                return cached_xodr

        client = self._client()
        world = client.get_world()
        map_name = world.get_map().name
        normalized_map_name = normalize_map_name(map_name)
        if not force_refresh:
            with self._lock:
                cached_xodr = self._map_xodr_cache.get(normalized_map_name)
            if cached_xodr is not None:
                return cached_xodr

        xodr_text = world.get_map().to_opendrive()
        with self._lock:
            self._map_xodr_cache[normalized_map_name] = xodr_text
            self._last_xodr_map_name = normalized_map_name
        return xodr_text

    def get_generated_map(self, force_refresh: bool = False) -> dict[str, Any]:
        cached_map_name = normalize_map_name(self._get_cached_current_map_name() or "")
        if cached_map_name and not force_refresh:
            with self._lock:
                cached_generated = self._generated_map_cache.get(cached_map_name)
            if cached_generated is not None:
                return copy.deepcopy(cached_generated)
        if not cached_map_name and not force_refresh:
            with self._lock:
                last_cached_map_name = self._last_xodr_map_name
                cached_generated = self._generated_map_cache.get(last_cached_map_name or "")
            if cached_generated is not None:
                return copy.deepcopy(cached_generated)

        client = self._client()
        world = client.get_world()
        map_name = world.get_map().name
        normalized_map_name = normalize_map_name(map_name)
        if not force_refresh:
            with self._lock:
                cached_generated = self._generated_map_cache.get(normalized_map_name)
            if cached_generated is not None:
                return copy.deepcopy(cached_generated)

        with self._lock:
            cached_xodr = None if force_refresh else self._map_xodr_cache.get(normalized_map_name)
        if cached_xodr is None:
            xodr_text = world.get_map().to_opendrive()
            with self._lock:
                self._map_xodr_cache[normalized_map_name] = xodr_text
                self._last_xodr_map_name = normalized_map_name
        else:
            xodr_text = cached_xodr

        generated_map = build_generated_map(normalized_map_name, xodr_text)
        with self._lock:
            self._generated_map_cache[normalized_map_name] = generated_map
            self._last_xodr_map_name = normalized_map_name
        return copy.deepcopy(generated_map)

    def get_generated_map_with_runtime(self, force_refresh: bool = False) -> dict[str, object]:
        """Return generated map data with runtime road data merged under a 'runtime' key."""
        generated = self.get_generated_map(force_refresh=force_refresh)
        try:
            runtime = self.get_runtime_map(force_refresh=force_refresh)
            generated["runtime"] = {
                "map_name": runtime.map_name,
                "normalized_map_name": runtime.normalized_map_name,
                "road_segments": [seg.model_dump() for seg in runtime.road_segments],
                "road_summaries": [rs.model_dump() for rs in runtime.road_summaries],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to merge runtime map data into generated map: %s", exc)
            generated["runtime"] = None
        return generated

    def list_blueprints(self) -> dict[str, list[str]]:
        with self._lock:
            cached = self._blueprints_cache
        if cached is not None:
            return {kind: list(items) for kind, items in cached.items()}

        client = self._client()
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        blueprints = {
            "vehicles": sorted(bp.id for bp in blueprint_library.filter("vehicle.*")),
            "walkers": sorted(bp.id for bp in blueprint_library.filter("walker.pedestrian.*")),
        }
        with self._lock:
            self._blueprints_cache = blueprints
        return {kind: list(items) for kind, items in blueprints.items()}
