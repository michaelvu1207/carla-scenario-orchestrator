from __future__ import annotations

import copy
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Callable

from .carla_runner.dataset_repository import (
    build_runtime_road_summaries,
    dataset_lane_type_counts,
    list_supported_maps,
    normalize_map_name,
    set_generated_map_cache,
)
from .road_corridors import build_corridors, build_corridor_lookup
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


class SlotInfo:
    """Lightweight info about an execution slot for metadata routing."""
    def __init__(self, slot_index: int, port: int, current_map: str | None, busy: bool):
        self.slot_index = slot_index
        self.port = port
        self.current_map = current_map
        self.busy = busy


class CarlaMetadataService:
    """Read-only CARLA metadata queries, distributed across execution slots.

    Instead of connecting to a dedicated metadata GPU, routes queries to
    whichever execution slot already has the requested map loaded. This
    avoids costly map switches and allows all GPUs to serve simulations.

    Architecture:
      Query arrives (e.g. get_runtime_map for VW_Poc)
        → _resolve_slot("VW_Poc") finds idle slot with VW_Poc loaded
        → Creates temporary carla.Client to that slot's port
        → Executes query, returns result
        → Slot is never "acquired" — just borrowed for a read-only query
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
        slot_resolver: Callable[[str | None], SlotInfo | None] | None = None,
    ) -> None:
        self.host = host or os.environ.get("ORCH_CARLA_METADATA_HOST", "127.0.0.1")
        self.default_port = port or int(os.environ.get("ORCH_CARLA_METADATA_PORT", "2000"))
        self.timeout = timeout or float(os.environ.get("ORCH_CARLA_METADATA_TIMEOUT", "20"))
        self._slot_resolver = slot_resolver
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
        self._corridor_cache: dict[str, list] = {}
        self._last_xodr_map_name: str | None = None
        # Track which map was last requested (for implicit map context)
        self._current_map_name: str | None = None

    def set_current_map(self, map_name: str) -> None:
        """Set the current map context (called when user switches maps in the editor)."""
        self._current_map_name = normalize_map_name(map_name)

    def _resolve_port(self, map_name: str | None = None) -> int:
        """Find the best slot port for the given map. Falls back to default_port."""
        if self._slot_resolver is None:
            return self.default_port

        target = map_name or self._current_map_name
        slot = self._slot_resolver(target)
        if slot is not None:
            return slot.port
        return self.default_port

    def _client(self, map_name: str | None = None) -> Any:
        port = self._resolve_port(map_name)
        return _make_client(self.host, port, self.timeout)

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
        # Prefer the explicitly set current map
        if self._current_map_name:
            return self._current_map_name
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
            try:
                current_map = world.get_map().name
            except RuntimeError:
                current_map = None
            supported = set(list_supported_maps())
            available = []
            for item in client.get_available_maps():
                normalized = normalize_map_name(item)
                if normalized not in supported:
                    continue
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
                warning_message = "Returning cached CARLA metadata after refresh failure: %s" % exc
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
        """Load a map. Routes to a slot that already has it, or picks any idle slot."""
        normalized = normalize_map_name(map_name)

        # Try to find a slot that already has this map
        client = self._client(map_name=normalized)
        world = client.get_world()
        try:
            current = world.get_map().name
            needs_load = normalize_map_name(current) != normalized
        except RuntimeError:
            needs_load = True

        if needs_load:
            client.load_world(map_name)
            time.sleep(1.0)

        self._current_map_name = normalized
        self._invalidate_status_cache()
        return self.get_status(force_refresh=True)

    def get_runtime_map(self, force_refresh: bool = False) -> RuntimeMapResponse:
        target_map = self._current_map_name
        if target_map and not force_refresh:
            with self._lock:
                cached_runtime = self._runtime_map_cache.get(target_map)
            if cached_runtime is not None:
                return cached_runtime.model_copy(deep=True)

        client = self._client(map_name=target_map)
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
                        id="road-%d-section-%d-lane-%d" % (road_id, section_id, lane_id),
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
        # Ensure generated map cache is populated before building summaries
        self.get_generated_map(force_refresh=False)
        all_summaries = build_runtime_road_summaries(map_name)
        # Filter to only roads that exist in CARLA runtime
        runtime_road_ids = {str(seg.road_id) for seg in segments}
        road_summaries = [rs for rs in all_summaries if rs.id in runtime_road_ids]
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
        target_map = self._current_map_name
        if target_map and not force_refresh:
            with self._lock:
                cached_xodr = self._map_xodr_cache.get(target_map)
            if cached_xodr is not None:
                return cached_xodr

        client = self._client(map_name=target_map)
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
        target_map = self._current_map_name
        if target_map and not force_refresh:
            with self._lock:
                cached_generated = self._generated_map_cache.get(target_map)
            if cached_generated is not None:
                return copy.deepcopy(cached_generated)

        client = self._client(map_name=target_map)
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
        set_generated_map_cache(normalized_map_name, generated_map)
        # Build road corridors from xodr
        try:
            from .road_corridors import build_corridors
            corridors = build_corridors(xodr_text)
            generated_map["corridors"] = [
                {
                    "id": c.id,
                    "road_ids": c.road_ids,
                    "junction_ids": c.junction_ids,
                    "total_length": round(c.total_length, 1),
                    "segment_lengths": [round(sl, 1) for sl in c.segment_lengths],
                    "segment_offsets": [round(so, 1) for so in c.segment_offsets],
                    "has_parking": c.has_parking,
                    "lane_config": c.lane_config,
                    "bearing_deg": c.bearing_deg,
                    "description": c.description,
                    "road_count": len(c.road_ids),
                }
                for c in corridors
            ]
            with self._lock:
                self._corridor_cache[normalized_map_name] = corridors
        except Exception as exc:
            logger.warning("Failed to build corridors: %s", exc)
            generated_map["corridors"] = []
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
            # Filter corridors to only include roads that exist in CARLA runtime
            rt_ids = {str(seg.road_id) for seg in runtime.road_segments}
            corridors = generated.get("corridors", [])
            filtered = [
                c for c in corridors
                if all(rid in rt_ids for rid in c["road_ids"])
            ]
            # Build RoadCorridor objects from filtered dicts
            from .road_corridors import RoadCorridor, adjust_corridors_for_junction_gaps
            filtered_objs = []
            for cd in filtered:
                filtered_objs.append(RoadCorridor(
                    id=cd["id"], road_ids=cd["road_ids"], junction_ids=cd.get("junction_ids", []),
                    total_length=cd["total_length"], segment_lengths=cd["segment_lengths"],
                    segment_offsets=cd["segment_offsets"], has_parking=cd["has_parking"],
                    lane_config=cd["lane_config"], bearing_deg=cd["bearing_deg"],
                    description=cd["description"],
                ))
            # Adjust corridor distances to include physical junction gaps
            rt_segments = [seg.model_dump() for seg in runtime.road_segments]
            adjusted_objs = adjust_corridors_for_junction_gaps(filtered_objs, rt_segments)
            # Serialize adjusted corridors for API response
            generated["corridors"] = [
                {
                    "id": ac.id, "road_ids": ac.road_ids, "junction_ids": ac.junction_ids,
                    "total_length": ac.total_length,
                    "segment_lengths": [round(sl, 1) for sl in ac.segment_lengths],
                    "segment_offsets": [round(so, 1) for so in ac.segment_offsets],
                    "has_parking": ac.has_parking, "lane_config": ac.lane_config,
                    "bearing_deg": ac.bearing_deg, "description": ac.description,
                    "road_count": len(ac.road_ids),
                }
                for ac in adjusted_objs
            ]
            # Cache adjusted corridors for scene assistant
            normalized = generated.get("name", "")
            if normalized:
                with self._lock:
                    self._corridor_cache[normalized] = adjusted_objs
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to merge runtime map data into generated map: %s", exc)
            generated["runtime"] = None
        return generated

    def list_blueprints(self) -> dict[str, list[str]]:
        with self._lock:
            cached = self._blueprints_cache
        if cached is not None:
            return {kind: list(items) for kind, items in cached.items()}

        # Blueprints are the same across all maps — use any available slot
        client = self._client(map_name=None)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        blueprints = {
            "vehicles": sorted(bp.id for bp in blueprint_library.filter("vehicle.*")),
            "walkers": sorted(bp.id for bp in blueprint_library.filter("walker.pedestrian.*")),
        }
        with self._lock:
            self._blueprints_cache = blueprints
        return {kind: list(items) for kind, items in blueprints.items()}

    def get_street_furniture(self, force_refresh: bool = False) -> dict[str, Any]:
        """Return streetlight poles and traffic light positions for the current map."""
        target_map = self._current_map_name
        cache_key = f"street_furniture_{target_map}"
        if not force_refresh:
            with self._lock:
                cached = getattr(self, "_street_furniture_cache", {}).get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)

        client = self._client(map_name=target_map)
        world = client.get_world()
        carla = _require_carla()

        result: dict[str, Any] = {"poles": [], "traffic_lights": []}

        # Get pole environment objects (includes streetlights)
        try:
            poles = world.get_environment_objects(carla.CityObjectLabel.Poles)
            for obj in poles:
                t = obj.transform
                coords = _carla_to_frontend(t.location, t.rotation)
                result["poles"].append({
                    "id": obj.id,
                    "name": obj.name,
                    **coords,
                })
        except Exception as exc:
            logger.warning("Failed to get poles: %s", exc)

        # Get traffic light actors (richer API with stop waypoints)
        try:
            for tl in world.get_actors().filter("traffic.traffic_light"):
                t = tl.get_transform()
                coords = _carla_to_frontend(t.location, t.rotation)
                stop_wps = []
                try:
                    for wp in tl.get_stop_waypoints():
                        wc = _carla_to_frontend(wp.transform.location, wp.transform.rotation)
                        stop_wps.append(wc)
                except Exception:
                    pass
                result["traffic_lights"].append({
                    "actor_id": tl.id,
                    "type_id": tl.type_id,
                    **coords,
                    "pole_index": tl.get_pole_index(),
                    "opendrive_id": tl.get_opendrive_id(),
                    "stop_waypoints": stop_wps,
                })
        except Exception as exc:
            logger.warning("Failed to get traffic lights: %s", exc)

        with self._lock:
            if not hasattr(self, "_street_furniture_cache"):
                self._street_furniture_cache: dict[str, Any] = {}
            self._street_furniture_cache[cache_key] = result
        return copy.deepcopy(result)

    def set_weather(self, weather_params: dict) -> dict[str, Any]:
        """Set weather parameters on the CARLA world."""
        carla = _require_carla()
        client = self._client()
        world = client.get_world()
        weather = world.get_weather()

        param_map = {
            'cloudiness': 'cloudiness',
            'precipitation': 'precipitation',
            'precipitation_deposits': 'precipitation_deposits',
            'wind_intensity': 'wind_intensity',
            'fog_density': 'fog_density',
            'fog_distance': 'fog_distance',
            'sun_altitude_angle': 'sun_altitude_angle',
            'sun_azimuth_angle': 'sun_azimuth_angle',
        }
        applied = {}
        for key, attr in param_map.items():
            if key in weather_params and weather_params[key] is not None:
                value = float(weather_params[key])
                setattr(weather, attr, value)
                applied[key] = value

        world.set_weather(weather)
        return {
            'status': 'ok',
            'applied': applied,
            'current_weather': {
                'cloudiness': weather.cloudiness,
                'precipitation': weather.precipitation,
                'precipitation_deposits': weather.precipitation_deposits,
                'wind_intensity': weather.wind_intensity,
                'fog_density': weather.fog_density,
                'fog_distance': weather.fog_distance,
                'sun_altitude_angle': weather.sun_altitude_angle,
                'sun_azimuth_angle': weather.sun_azimuth_angle,
            },
        }

    def set_traffic_light_state(self, traffic_light_id: int, state: str, duration_seconds: float | None = None) -> dict[str, Any]:
        """Set a specific traffic light to a given state (red/yellow/green). Freezes all lights first."""
        carla = _require_carla()
        client = self._client()
        world = client.get_world()

        state_map = {
            'red': carla.TrafficLightState.Red,
            'yellow': carla.TrafficLightState.Yellow,
            'green': carla.TrafficLightState.Green,
        }
        carla_state = state_map.get(state.lower())
        if carla_state is None:
            raise ValueError(f"Invalid traffic light state: {state}. Must be red, yellow, or green.")

        # Find the target traffic light
        target_tl = None
        for tl in world.get_actors().filter('traffic.traffic_light'):
            if tl.id == traffic_light_id:
                target_tl = tl
                break

        if target_tl is None:
            raise ValueError(f"Traffic light with id {traffic_light_id} not found.")

        # Freeze all traffic lights first
        for tl in world.get_actors().filter('traffic.traffic_light'):
            tl.freeze(True)

        # Set desired state
        target_tl.set_state(carla_state)
        if duration_seconds is not None:
            target_tl.set_green_time(duration_seconds if state == 'green' else 0.0)
            target_tl.set_red_time(duration_seconds if state == 'red' else 0.0)
            target_tl.set_yellow_time(duration_seconds if state == 'yellow' else 0.0)

        t = target_tl.get_transform()
        coords = _carla_to_frontend(t.location, t.rotation)
        return {
            'status': 'ok',
            'traffic_light_id': traffic_light_id,
            'state': state.lower(),
            'frozen': True,
            'duration_seconds': duration_seconds,
            **coords,
        }

    def road_position_to_world(self, road_id: int, s_fraction: float, lane_id: int) -> dict[str, Any]:
        """Convert (road_id, s_fraction, lane_id) to world (x, y, yaw) using CARLA waypoints."""
        client = self._client()
        world = client.get_world()
        carla_map = world.get_map()

        # Get all waypoints for the road and lane
        all_waypoints = carla_map.generate_waypoints(1.0)
        road_waypoints = [
            wp for wp in all_waypoints
            if int(wp.road_id) == road_id and int(wp.lane_id) == lane_id
        ]

        if not road_waypoints:
            raise ValueError(f"No waypoints found for road_id={road_id}, lane_id={lane_id}")

        # Sort by s value
        road_waypoints.sort(key=lambda wp: wp.s)

        # Interpolate by s_fraction
        total_length = road_waypoints[-1].s - road_waypoints[0].s
        if total_length <= 0:
            wp = road_waypoints[0]
        else:
            target_s = road_waypoints[0].s + s_fraction * total_length
            # Find closest waypoint
            wp = min(road_waypoints, key=lambda w: abs(w.s - target_s))

        coords = _carla_to_frontend(wp.transform.location, wp.transform.rotation)
        return {
            'road_id': road_id,
            's_fraction': s_fraction,
            'lane_id': lane_id,
            **coords,
        }
