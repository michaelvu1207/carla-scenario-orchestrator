from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_repository import build_selected_roads, build_runtime_road_summaries, dataset_lane_type_counts, list_supported_maps, normalize_map_name
from .models import (
    ActorDraft,
    ActorTimelineClip,
    CarlaMapInfo,
    CarlaStatusResponse,
    RecordingInfo,
    RuntimeMapResponse,
    RuntimeRoadSegment,
    SelectedRoad,
    SimulationActorState,
    SimulationRunDiagnostics,
    SimulationRunRequest,
    SimulationStreamMessage,
)


@dataclass
class TimelineDirective:
    target_speed_mps: float | None = None
    hold_position: bool = False
    autopilot_enabled: bool | None = None
    route_follow_enabled: bool = False
    lane_change_direction: str | None = None
    route_instruction: str | None = None
    chase_target_actor_id: str | None = None
    ram_target_actor_id: str | None = None
    follow_distance_m: float | None = None


@dataclass
class TimelineActorState:
    applied_clips: set[str] = field(default_factory=set)
    autopilot_applied: bool = False
    pending_lane_change: str | None = None
    lane_change_requests_remaining: int = 0
    pending_route_instruction: str | None = None
    pending_route_origin_road_id: int | None = None
    pending_route_locations: list[Any] | None = None
    authored_route_locations: list[Any] | None = None
    authored_route_applied: bool = False


def _require_carla():
    try:
        import carla  # type: ignore

        return carla
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CARLA Python API is not available in this environment. "
            "Run the backend in a venv that already has a matching carla package."
        ) from exc


def carla_to_frontend(location: Any, rotation: Any) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": -float(location.y),
        "z": float(location.z),
        "yaw": -float(rotation.yaw),
    }


def frontend_to_carla_xy(x: float, y: float) -> tuple[float, float]:
    return float(x), -float(y)


def _make_client(host: str, port: int, timeout: float) -> Any:
    carla = _require_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def _destroy_carla_handles(client: Any, handles: list[Any], debug_log: Path) -> None:
    alive_ids: list[int] = []
    for handle in handles:
        try:
            if handle and getattr(handle, "is_alive", False):
                alive_ids.append(int(handle.id))
        except Exception:
            continue
    if not alive_ids:
        return
    try:
        carla = _require_carla()
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in alive_ids])
        _append_debug_log(debug_log, f"Destroyed CARLA handles: {', '.join(str(actor_id) for actor_id in alive_ids)}.")
    except Exception as exc:  # noqa: BLE001
        _append_debug_log(debug_log, f"Failed to destroy CARLA handles {alive_ids}: {exc}")


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


def _anchor_for_actor(actor: ActorDraft, anchor_name: str) -> Any:
    anchor = actor.spawn if anchor_name == "spawn" else actor.destination
    if anchor is None:
        anchor = actor.spawn
    return anchor


def _resolve_anchor_value(
    waypoint_index: dict[int, list[Any]],
    anchor: Any,
    *,
    exact_waypoint_index: dict[int, list[Any]] | None = None,
    carla_map: Any | None = None,
    road_length_lookup: dict[int, float] | None = None,
) -> Any:
    road_id = int(anchor.road_id)

    if exact_waypoint_index is not None and (anchor.section_id is not None or anchor.lane_id is not None):
        exact_points = exact_waypoint_index.get(road_id) or []
        exact_candidates = _filter_anchor_candidates(
            exact_points,
            anchor,
            require_section=anchor.section_id is not None,
            require_lane=anchor.lane_id is not None,
        )
        if exact_candidates:
            return _select_anchor_candidate(exact_candidates, anchor.s_fraction)

    xodr_waypoint = _waypoint_from_xodr(
        carla_map,
        anchor,
        (road_length_lookup or {}).get(road_id),
    )
    if xodr_waypoint is not None:
        return xodr_waypoint

    points = waypoint_index.get(road_id)
    if not points:
        raise RuntimeError(f"Road {road_id} has no drivable waypoints in the active CARLA map.")
    lane_candidates = _filter_anchor_candidates(points, anchor)
    return _select_anchor_candidate(lane_candidates, anchor.s_fraction)


def _filter_anchor_candidates(
    points: list[Any],
    anchor: Any,
    *,
    require_section: bool = False,
    require_lane: bool = False,
) -> list[Any]:
    candidates = points
    if anchor.section_id is not None:
        section_matches = [point for point in candidates if int(getattr(point, "section_id", 0)) == anchor.section_id]
        if section_matches:
            candidates = section_matches
        elif require_section:
            return []
    if anchor.lane_id is not None:
        lane_matches = [point for point in candidates if int(getattr(point, "lane_id", 0)) == anchor.lane_id]
        if lane_matches:
            candidates = lane_matches
        elif require_lane:
            return []
    return candidates


def _select_anchor_candidate(points: list[Any], s_fraction: float) -> Any:
    if not points:
        raise RuntimeError("Anchor selection requires at least one waypoint candidate.")
    index = min(
        len(points) - 1,
        max(0, int(round(s_fraction * (len(points) - 1)))),
    )
    return points[index]


def _road_length_lookup(selected_roads: list[SelectedRoad]) -> dict[int, float]:
    lookup: dict[int, float] = {}
    for road in selected_roads:
        try:
            road_id = int(road.id)
        except (TypeError, ValueError):
            continue
        if road.length > 0:
            lookup[road_id] = float(road.length)
    return lookup


def _canonical_selected_roads_for_request(request: SimulationRunRequest) -> list[SelectedRoad]:
    road_ids: list[str] = []
    seen: set[str] = set()

    def add_road_id(road_id: str | None) -> None:
        if not road_id:
            return
        normalized = str(road_id)
        if normalized in seen:
            return
        seen.add(normalized)
        road_ids.append(normalized)

    for road in request.selected_roads:
        add_road_id(road.id)
    for actor in request.actors:
        if actor.placement_mode != "road":
            continue
        add_road_id(actor.spawn.road_id)
        if actor.destination is not None:
            add_road_id(actor.destination.road_id)
        for anchor in actor.route:
            add_road_id(anchor.road_id)
    return build_selected_roads(request.map_name, road_ids)


def _anchor_s_from_fraction(anchor: Any, road_length: float | None) -> float | None:
    if road_length is None or road_length <= 0:
        return None
    upper_bound = max(0.5, road_length - 0.5)
    return max(0.5, min(upper_bound, float(anchor.s_fraction) * road_length))


def _anchor_s_candidates(anchor: Any, road_length: float | None, *, max_offset: float = 12.0, step: float = 2.0) -> list[float]:
    target_s = _anchor_s_from_fraction(anchor, road_length)
    if target_s is None:
        return []
    upper_bound = max(0.5, (road_length or target_s) - 0.5)
    candidates = [target_s]
    offset = step
    while offset <= max_offset + 1e-6:
        for direction in (1.0, -1.0):
            candidate = target_s + (direction * offset)
            candidate = max(0.5, min(upper_bound, candidate))
            if all(abs(candidate - existing) > 1e-6 for existing in candidates):
                candidates.append(candidate)
        offset += step
    return candidates


def _waypoint_from_xodr(carla_map: Any, anchor: Any, road_length: float | None, s_value: float | None = None) -> Any | None:
    if carla_map is None or anchor.lane_id is None:
        return None
    if not hasattr(carla_map, "get_waypoint_xodr"):
        return None
    target_s = s_value if s_value is not None else _anchor_s_from_fraction(anchor, road_length)
    if target_s is None:
        return None
    try:
        waypoint = carla_map.get_waypoint_xodr(int(anchor.road_id), int(anchor.lane_id), float(target_s))
    except Exception:
        return None
    if waypoint is None:
        return None
    if anchor.section_id is not None and int(getattr(waypoint, "section_id", 0)) != int(anchor.section_id):
        return None
    return waypoint


def _resolve_anchor(
    waypoint_index: dict[int, list[Any]],
    actor: ActorDraft,
    anchor_name: str = "spawn",
    *,
    exact_waypoint_index: dict[int, list[Any]] | None = None,
    carla_map: Any | None = None,
    road_length_lookup: dict[int, float] | None = None,
) -> Any:
    anchor = _anchor_for_actor(actor, anchor_name)
    return _resolve_anchor_value(
        waypoint_index,
        anchor,
        exact_waypoint_index=exact_waypoint_index,
        carla_map=carla_map,
        road_length_lookup=road_length_lookup,
    )


def _route_locations_for_actor(
    waypoint_index: dict[int, list[Any]],
    actor: ActorDraft,
    *,
    exact_waypoint_index: dict[int, list[Any]] | None = None,
    carla_map: Any | None = None,
    road_length_lookup: dict[int, float] | None = None,
) -> list[Any]:
    locations: list[Any] = []
    for anchor in actor.route:
        try:
            waypoint = _resolve_anchor_value(
                waypoint_index,
                anchor,
                exact_waypoint_index=exact_waypoint_index,
                carla_map=carla_map,
                road_length_lookup=road_length_lookup,
            )
        except Exception:
            continue
        location = waypoint.transform.location
        if locations:
            previous = locations[-1]
            if (
                abs(float(previous.x) - float(location.x)) < 0.25
                and abs(float(previous.y) - float(location.y)) < 0.25
                and abs(float(previous.z) - float(location.z)) < 0.25
            ):
                continue
        locations.append(location)
    return locations


def _road_spawn_transform_candidates(
    world: Any,
    actor: ActorDraft,
    anchor_name: str,
    *,
    fallback_anchor: Any | None,
    road_length_lookup: dict[int, float],
    z_offset: float,
) -> list[Any]:
    carla_map = world.get_map()
    anchor = _anchor_for_actor(actor, anchor_name)
    candidates: list[Any] = []
    against_lane = (
        actor.kind == "vehicle"
        and actor.placement_mode == "road"
        and str(getattr(actor, "lane_facing", "with_lane")) == "against_lane"
    )

    if fallback_anchor is not None:
        candidates.append(
            _raised_waypoint_transform(
                fallback_anchor,
                z_offset=z_offset,
                against_lane=against_lane,
            )
        )

    if actor.placement_mode != "road" or anchor.lane_id is None:
        return candidates

    road_length = road_length_lookup.get(int(anchor.road_id))
    for s_value in _anchor_s_candidates(anchor, road_length):
        waypoint = _waypoint_from_xodr(carla_map, anchor, road_length, s_value=s_value)
        if waypoint is None:
            continue
        transform = _raised_waypoint_transform(
            waypoint,
            z_offset=z_offset,
            against_lane=against_lane,
        )
        if any(
            abs(float(existing.location.x) - float(transform.location.x)) < 1e-3
            and abs(float(existing.location.y) - float(transform.location.y)) < 1e-3
            and abs(float(existing.location.z) - float(transform.location.z)) < 1e-3
            for existing in candidates
        ):
            continue
        candidates.append(transform)
    return candidates


def _try_spawn_vehicle_from_candidates(world: Any, blueprint: Any, transforms: list[Any]) -> Any | None:
    for transform in transforms:
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            return vehicle
    return None


def _is_static_road_vehicle(actor: ActorDraft) -> bool:
    return actor.kind == "vehicle" and actor.placement_mode == "road" and bool(actor.is_static)


def _count_static_road_vehicles(actors: list[ActorDraft]) -> int:
    return sum(1 for actor in actors if _is_static_road_vehicle(actor))


def _should_skip_vehicle_spawn_failure(actor: ActorDraft, static_road_vehicle_count: int) -> bool:
    return _is_static_road_vehicle(actor) and static_road_vehicle_count > 1


def _estimate_ground_z(world: Any, x: float, y: float) -> float:
    carla = _require_carla()
    probe = carla.Location(x=float(x), y=float(y), z=2.0)
    try:
        waypoint = world.get_map().get_waypoint(probe, project_to_road=True)
    except TypeError:
        waypoint = world.get_map().get_waypoint(probe)
    if waypoint:
        return float(waypoint.transform.location.z)
    return 0.0


def _location_from_map_point(world: Any, point: Any, z_offset: float = 0.15) -> Any:
    carla = _require_carla()
    x, y = frontend_to_carla_xy(point.x, point.y)
    ground_z = _estimate_ground_z(world, x, y)
    return carla.Location(x=x, y=y, z=ground_z + z_offset)


def _walker_transform_from_points(world: Any, spawn_point: Any, destination_point: Any | None) -> Any:
    carla = _require_carla()
    location = _location_from_map_point(world, spawn_point, z_offset=0.2)
    yaw = 0.0
    if destination_point is not None:
        dx = float(destination_point.x) - float(spawn_point.x)
        dy = float(destination_point.y) - float(spawn_point.y)
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            yaw = math.degrees(math.atan2(-dy, dx))
    return carla.Transform(location, carla.Rotation(yaw=yaw))


def _vehicle_transform_from_points(world: Any, spawn_point: Any, destination_point: Any | None) -> Any:
    carla = _require_carla()
    location = _location_from_map_point(world, spawn_point, z_offset=0.35)
    yaw = 0.0
    if destination_point is not None:
        dx = float(destination_point.x) - float(spawn_point.x)
        dy = float(destination_point.y) - float(spawn_point.y)
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            yaw = math.degrees(math.atan2(-dy, dx))
    return carla.Transform(location, carla.Rotation(yaw=yaw))


def _raised_waypoint_transform(
    waypoint: Any,
    z_offset: float,
    *,
    against_lane: bool = False,
) -> Any:
    carla = _require_carla()
    transform = waypoint.transform
    location = transform.location
    rotation = transform.rotation
    yaw = float(rotation.yaw) + (180.0 if against_lane else 0.0)
    return carla.Transform(
        carla.Location(
            x=float(location.x),
            y=float(location.y),
            z=float(location.z) + z_offset,
        ),
        carla.Rotation(
            pitch=float(rotation.pitch),
            yaw=yaw,
            roll=float(rotation.roll),
        ),
    )


def _chase_camera_transform() -> Any:
    carla = _require_carla()
    return carla.Transform(
        carla.Location(x=-7.5, y=0.0, z=3.2),
        carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0),
    )


def _normalize_angle_radians(value: float) -> float:
    while value > math.pi:
        value -= 2 * math.pi
    while value < -math.pi:
        value += 2 * math.pi
    return value


def _sorted_timeline(actor: ActorDraft) -> list[ActorTimelineClip]:
    return sorted(actor.timeline, key=lambda clip: (float(clip.start_time), clip.id))


def _clear_direct_control_modes(directive: TimelineDirective) -> None:
    directive.route_follow_enabled = False
    directive.chase_target_actor_id = None
    directive.ram_target_actor_id = None
    directive.follow_distance_m = None


def _evaluate_timeline(actor: ActorDraft, state: TimelineActorState, simulation_time: float) -> TimelineDirective:
    directive = TimelineDirective(
        target_speed_mps=max(0.0, float(actor.speed_kph) / 3.6),
        hold_position=bool(actor.is_static),
        autopilot_enabled=bool(actor.autopilot),
        route_follow_enabled=False,
    )

    for clip in _sorted_timeline(actor):
        if not clip.enabled or simulation_time < float(clip.start_time):
            continue
        if clip.action == "follow_route":
            _clear_direct_control_modes(directive)
            directive.autopilot_enabled = False
            directive.hold_position = False
            directive.route_follow_enabled = True
            if clip.target_speed_kph is not None:
                directive.target_speed_mps = max(0.0, float(clip.target_speed_kph) / 3.6)
        elif clip.action == "set_speed" and clip.target_speed_kph is not None:
            directive.target_speed_mps = max(0.0, float(clip.target_speed_kph) / 3.6)
            directive.hold_position = False
        elif clip.action in {"stop", "hold_position"}:
            _clear_direct_control_modes(directive)
            directive.hold_position = True
            directive.target_speed_mps = 0.0
        elif clip.action == "enable_autopilot":
            _clear_direct_control_modes(directive)
            directive.autopilot_enabled = True
        elif clip.action == "disable_autopilot":
            directive.autopilot_enabled = False
        elif clip.action == "lane_change_left" and clip.id not in state.applied_clips:
            state.applied_clips.add(clip.id)
            directive.lane_change_direction = "left"
        elif clip.action == "lane_change_right" and clip.id not in state.applied_clips:
            state.applied_clips.add(clip.id)
            directive.lane_change_direction = "right"
        elif clip.action == "turn_left_at_next_intersection" and clip.id not in state.applied_clips:
            state.applied_clips.add(clip.id)
            directive.route_instruction = "Left"
        elif clip.action == "turn_right_at_next_intersection" and clip.id not in state.applied_clips:
            state.applied_clips.add(clip.id)
            directive.route_instruction = "Right"
        elif clip.action == "chase_actor":
            _clear_direct_control_modes(directive)
            directive.autopilot_enabled = False
            directive.hold_position = False
            directive.chase_target_actor_id = clip.target_actor_id
            directive.follow_distance_m = max(6.0, float(clip.following_distance_m or 10.0))
            if clip.target_speed_kph is not None:
                directive.target_speed_mps = max(0.0, float(clip.target_speed_kph) / 3.6)
        elif clip.action == "ram_actor":
            _clear_direct_control_modes(directive)
            directive.autopilot_enabled = False
            directive.hold_position = False
            directive.ram_target_actor_id = clip.target_actor_id
            directive.follow_distance_m = 0.0
            if clip.target_speed_kph is not None:
                directive.target_speed_mps = max(0.0, float(clip.target_speed_kph) / 3.6)
    return directive


def _set_vehicle_autopilot(vehicle: Any, enabled: bool, tm_port: int) -> None:
    try:
        vehicle.set_autopilot(enabled, tm_port)
    except TypeError:
        vehicle.set_autopilot(enabled)


def _apply_tm_speed_target(tm: Any, vehicle: Any, target_speed_mps: float | None) -> None:
    if tm is None or target_speed_mps is None:
        return
    if not hasattr(tm, "vehicle_percentage_speed_difference"):
        return
    try:
        speed_limit = max(1.0, float(vehicle.get_speed_limit()))
    except Exception:
        speed_limit = max(1.0, target_speed_mps)
    percentage = 100.0 - ((target_speed_mps / speed_limit) * 100.0)
    percentage = max(-90.0, min(95.0, percentage))
    try:
        tm.vehicle_percentage_speed_difference(vehicle, percentage)
    except Exception:
        pass


def _drivable_adjacent_lane_id(waypoint: Any, side: str) -> int | None:
    candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if candidate is None:
        return None
    lane_type = str(getattr(candidate, "lane_type", "")).split(".")[-1]
    if lane_type not in {"Driving", "Bidirectional"}:
        return None
    return int(getattr(candidate, "lane_id", 0))


def _tm_force_lane_change(tm: Any, vehicle: Any, direction: str) -> bool:
    if tm is None or not hasattr(tm, "force_lane_change"):
        return False

    try:
        if hasattr(tm, "auto_lane_change"):
            tm.auto_lane_change(vehicle, True)
    except Exception:
        pass
    try:
        # Use vehicle-relative lane direction. Verified on the live simulator:
        # for Town05 road 37 lane 2, `get_right_lane()` is lane 3 (Driving),
        # so a move from lane 2 -> 3 is a right lane change.
        tm.force_lane_change(vehicle, direction == "right")
        return True
    except Exception:
        return False


def _tm_set_route(tm: Any, vehicle: Any, route_instruction: str) -> bool:
    if tm is None or not hasattr(tm, "set_route"):
        return False
    try:
        tm.set_route(vehicle, [str(route_instruction)])
        return True
    except Exception:
        return False


def _tm_set_path(tm: Any, vehicle: Any, locations: list[Any]) -> bool:
    if tm is None or not hasattr(tm, "set_path") or not locations:
        return False
    try:
        tm.set_path(vehicle, locations)
        return True
    except Exception:
        return False


def _classify_turn_direction(current_yaw: float, candidate_yaw: float) -> str:
    delta = math.degrees(_normalize_angle_radians(math.radians(candidate_yaw - current_yaw)))
    if delta > 20.0:
        return "Left"
    if delta < -20.0:
        return "Right"
    return "Straight"


def _find_turn_path_locations(carla_map: Any, waypoint: Any, route_instruction: str, carla: Any) -> list[Any]:
    if waypoint is None:
        return []
    step = 2.0
    current = waypoint
    base_yaw = float(waypoint.transform.rotation.yaw)
    traveled = 0.0
    while current is not None and not bool(getattr(current, "is_junction", False)) and traveled < 80.0:
        next_waypoints = current.next(step)
        if not next_waypoints:
            return []
        current = next_waypoints[0]
        traveled += step
    if current is None:
        return []

    frontier: list[tuple[Any, list[Any]]] = [(candidate, [candidate]) for candidate in current.next(step)]
    completed: list[tuple[str, list[Any]]] = []
    visited: set[tuple[int, int, int, int]] = set()
    while frontier:
        node, path = frontier.pop(0)
        key = (
            int(getattr(node, "road_id", 0)),
            int(getattr(node, "section_id", 0)),
            int(getattr(node, "lane_id", 0)),
            int(round(float(getattr(node, "s", 0.0)) * 10.0)),
        )
        if key in visited:
            continue
        visited.add(key)
        if not bool(getattr(node, "is_junction", False)):
            completed.append((_classify_turn_direction(base_yaw, float(node.transform.rotation.yaw)), path))
            continue
        next_waypoints = node.next(step)
        if not next_waypoints:
            completed.append((_classify_turn_direction(base_yaw, float(node.transform.rotation.yaw)), path))
            continue
        for candidate in next_waypoints:
            frontier.append((candidate, [*path, candidate]))

    matching_paths = [path for direction, path in completed if direction == route_instruction]
    if not matching_paths:
        return []
    chosen_path = max(matching_paths, key=len)
    final_waypoint = chosen_path[-1]
    path_locations = [final_waypoint.transform.location]
    next_waypoints = final_waypoint.next(8.0)
    if next_waypoints:
        path_locations.append(next_waypoints[0].transform.location)
    return [
        carla.Location(x=float(location.x), y=float(location.y), z=float(location.z))
        for location in path_locations
    ]


def _apply_path_vehicle_control(
    carla: Any,
    vehicle: Any,
    target_location: Any,
    target_speed_mps: float,
    *,
    stop_at_target: bool = True,
    arrival_distance_m: float = 2.5,
    reverse: bool = False,
) -> bool:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    location = transform.location

    dx = float(target_location.x) - float(location.x)
    dy = float(target_location.y) - float(location.y)
    distance = math.sqrt(dx * dx + dy * dy)
    speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)

    if distance <= arrival_distance_m:
        if stop_at_target:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        return True

    desired_yaw = math.atan2(dy, dx)
    current_yaw = math.radians(float(transform.rotation.yaw))
    heading_yaw = current_yaw + (math.pi if reverse else 0.0)
    yaw_error = _normalize_angle_radians(desired_yaw - heading_yaw)

    steer = max(-1.0, min(1.0, yaw_error / (1.05 if reverse else 0.9)))
    cruise_speed = max(1.5, target_speed_mps)
    throttle = 0.55 if reverse and speed < cruise_speed * 0.78 else 0.24 if reverse and speed < cruise_speed else 0.65 if speed < cruise_speed * 0.82 else 0.3 if speed < cruise_speed else 0.0
    if abs(yaw_error) > 0.9:
        throttle *= 0.45 if reverse else 0.35

    brake = 0.0
    if distance < 8.0 and speed > max(1.2, cruise_speed * 0.5):
        brake = min(0.75, (speed - cruise_speed * 0.4) / max(cruise_speed, 1.0))
    if abs(yaw_error) > 1.35 and speed > 3.0:
        brake = max(brake, 0.3 if reverse else 0.45)

    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse))
    return False


def _apply_target_vehicle_control(
    carla: Any,
    vehicle: Any,
    target_location: Any,
    target_speed_mps: float,
    *,
    aggressive: bool,
    target_velocity_mps: float = 0.0,
    follow_distance_m: float | None = None,
) -> bool:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    location = transform.location

    dx = float(target_location.x) - float(location.x)
    dy = float(target_location.y) - float(location.y)
    distance = math.sqrt(dx * dx + dy * dy)
    speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
    lead_speed = max(0.0, float(target_velocity_mps))
    desired_gap = 0.0 if aggressive else max(4.0, float(follow_distance_m or 8.0))
    hard_stop_gap = 0.0 if aggressive else max(2.5, desired_gap - 1.0)
    closing_speed = max(0.0, speed - lead_speed)
    if not aggressive and distance <= hard_stop_gap:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        return True

    desired_yaw = math.atan2(dy, dx)
    current_yaw = math.radians(float(transform.rotation.yaw))
    yaw_error = _normalize_angle_radians(desired_yaw - current_yaw)

    steer_divisor = 0.65 if aggressive else 0.9
    steer = max(-1.0, min(1.0, yaw_error / steer_divisor))
    cruise_speed = max(2.0, target_speed_mps)
    if aggressive:
        throttle = 0.9 if speed < cruise_speed * 0.98 else 0.45 if speed < cruise_speed * 1.15 else 0.1
        if abs(yaw_error) > 1.15:
            throttle *= 0.6
        brake = 0.0
        if abs(yaw_error) > 1.5 and speed > 12.0:
            brake = 0.15
    else:
        gap_error = distance - desired_gap
        stopping_margin = (closing_speed * closing_speed) / max(2.0, 2.0 * 4.5)
        caution_gap = desired_gap + stopping_margin + max(2.0, lead_speed * 0.6)
        cruise_speed = min(cruise_speed, lead_speed + max(0.0, gap_error) * 0.35)
        if distance < desired_gap + 6.0:
            cruise_speed = min(cruise_speed, lead_speed + max(0.0, gap_error) * 0.12)
        throttle = 0.55 if speed + 0.6 < cruise_speed else 0.16 if speed < cruise_speed else 0.0
        if abs(yaw_error) > 0.9:
            throttle *= 0.45
        if distance <= caution_gap:
            throttle = 0.0
        brake = 0.0
        if distance <= caution_gap:
            brake_window = max(1.0, caution_gap - desired_gap)
            brake = max(brake, min(1.0, (caution_gap - distance) / brake_window))
        if distance < desired_gap + 2.0 and closing_speed > 0.0:
            brake = max(brake, min(1.0, 0.6 + (closing_speed / 6.0)))
        if gap_error <= 0.5:
            brake = max(brake, 0.8 if closing_speed > 0.25 else 0.35)
        if abs(yaw_error) > 1.35 and speed > 3.0:
            brake = max(brake, 0.45)
        if brake > 0.2:
            throttle = 0.0

    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
    return False


def _apply_path_walker_control(carla: Any, walker: Any, target_location: Any, target_speed_mps: float) -> bool:
    transform = walker.get_transform()
    location = transform.location

    dx = float(target_location.x) - float(location.x)
    dy = float(target_location.y) - float(location.y)
    distance = math.sqrt(dx * dx + dy * dy)
    if distance <= 0.35:
        walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=0.0, y=0.0, z=0.0), speed=0.0))
        return True

    inv_distance = 1.0 / max(distance, 1e-6)
    direction = carla.Vector3D(x=dx * inv_distance, y=dy * inv_distance, z=0.0)
    walker.apply_control(
        carla.WalkerControl(
            direction=direction,
            speed=max(0.5, target_speed_mps),
            jump=False,
        )
    )
    try:
        yaw = math.degrees(math.atan2(dy, dx))
        walker.set_transform(carla.Transform(location, carla.Rotation(yaw=yaw)))
    except Exception:
        pass
    return False


def _encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> None:
    png_pattern = str(frames_dir / "%06d.png")
    numbers = sorted(int(path.stem) for path in frames_dir.glob("*.png") if path.stem.isdigit())
    if not numbers:
        raise RuntimeError(f"No frames found in {frames_dir}")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-start_number",
            str(numbers[0]),
            "-framerate",
            str(max(1, fps)),
            "-i",
            png_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def _put_message(message_queue: Any, payload: dict[str, Any]) -> None:
    try:
        message_queue.put(payload)
    except Exception:
        pass


def _append_debug_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _simulation_worker(
    request_payload: dict[str, Any],
    settings: dict[str, Any],
    message_queue: Any,
    stop_event: Any,
    pause_event: Any,
) -> None:
    request = SimulationRunRequest.model_validate(request_payload)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(settings["output_root"]) / run_id
    frames_dir = run_dir / "ego_camera_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    recording_path = run_dir / "recording.mp4"
    scenario_log = run_dir / "scenario.log"
    debug_log = run_dir / "run.log"
    actors: list[tuple[ActorDraft, Any]] = []
    walker_controllers: list[Any] = []
    sensor = None
    sensor_queue = None
    world = None
    original_settings = None
    tm = None
    frame = 0
    ego_vehicle = None
    fallback_vehicle = None
    saved_frame_count = 0
    sensor_timeout_count = 0
    last_sensor_frame = None
    worker_error = None
    skipped_actors: list[dict[str, Any]] = []
    path_vehicle_targets: list[tuple[Any, Any, float]] = []
    path_walker_targets: list[tuple[Any, Any, float]] = []
    timeline_states: dict[str, TimelineActorState] = {}
    actor_by_handle_id: dict[int, ActorDraft] = {}
    handle_by_actor_id: dict[str, Any] = {}

    try:
        _append_debug_log(
            debug_log,
            (
                f"Run started map={request.map_name} duration={request.duration_seconds:.2f}s "
                f"fixed_delta={request.fixed_delta_seconds:.3f}s actors={len(request.actors)} "
                f"recording={request.topdown_recording}"
            ),
        )
        carla = _require_carla()
        client = _make_client(settings["carla_host"], settings["carla_port"], settings["carla_timeout"])
        current = client.get_world().get_map().name
        if normalize_map_name(current) != normalize_map_name(request.map_name):
            _append_debug_log(debug_log, f"Loading CARLA map {request.map_name} from {current}.")
            client.load_world(request.map_name)
            time.sleep(1.0)

        world = client.get_world()
        original_settings = world.get_settings()
        sim_settings = world.get_settings()
        sim_settings.synchronous_mode = True
        sim_settings.fixed_delta_seconds = request.fixed_delta_seconds
        world.apply_settings(sim_settings)

        tm = client.get_trafficmanager(settings["tm_port"])
        _append_debug_log(debug_log, f"Traffic Manager ready on port {settings['tm_port']}.")
        if hasattr(tm, "set_synchronous_mode"):
            tm.set_synchronous_mode(True)
        if hasattr(tm, "set_global_distance_to_leading_vehicle"):
            tm.set_global_distance_to_leading_vehicle(2.5)
        if hasattr(tm, "set_global_percentage_speed_difference"):
            tm.set_global_percentage_speed_difference(-10.0)

        try:
            client.start_recorder(str(scenario_log))
            _append_debug_log(debug_log, f"CARLA recorder started at {scenario_log}.")
        except Exception:
            scenario_log = None
            _append_debug_log(debug_log, "CARLA recorder unavailable; continuing without scenario.log.")

        waypoint_index = _build_waypoint_index(
            world,
            distance=2.0,
            allowed_lane_types={"Driving", "Bidirectional"},
        )
        exact_spawn_waypoint_index = _build_waypoint_index(
            world,
            distance=2.0,
            allowed_lane_types={
                "Driving",
                "Bidirectional",
                "Parking",
                "Shoulder",
                "Sidewalk",
                "Biking",
            },
        )
        road_lengths = _road_length_lookup(request.selected_roads)
        blueprint_library = world.get_blueprint_library()
        carla_map = world.get_map()
        static_road_vehicle_count = _count_static_road_vehicles(request.actors)
        route_vehicle_targets: list[dict[str, Any]] = []

        for actor in request.actors:
            timeline_states[actor.id] = TimelineActorState()
            _append_debug_log(
                debug_log,
                (
                    f"Spawning actor label={actor.label} kind={actor.kind} role={actor.role} "
                    f"placement={actor.placement_mode} road={actor.spawn.road_id} "
                    f"section={actor.spawn.section_id} lane={actor.spawn.lane_id}"
                ),
            )
            if actor.kind == "vehicle":
                anchor = _resolve_anchor(
                    waypoint_index,
                    actor,
                    "spawn",
                    exact_waypoint_index=exact_spawn_waypoint_index,
                    carla_map=carla_map,
                    road_length_lookup=road_lengths,
                )
                if actor.placement_mode in {"path", "point"} and actor.spawn_point is not None:
                    vehicle_spawn_candidates = [
                        _vehicle_transform_from_points(world, actor.spawn_point, actor.destination_point)
                    ]
                else:
                    vehicle_spawn_candidates = _road_spawn_transform_candidates(
                        world,
                        actor,
                        "spawn",
                        fallback_anchor=anchor,
                        road_length_lookup=road_lengths,
                        z_offset=0.35,
                    )
                blueprints = blueprint_library.filter(actor.blueprint)
                if not blueprints:
                    raise RuntimeError(f"Vehicle blueprint {actor.blueprint} was not found.")
                blueprint = blueprints[0]
                role_name = "hero" if actor.role == "ego" else "autopilot"
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", role_name)
                if actor.color and blueprint.has_attribute("color"):
                    blueprint.set_attribute("color", actor.color)
                vehicle = _try_spawn_vehicle_from_candidates(world, blueprint, vehicle_spawn_candidates)
                if vehicle is None:
                    if actor.placement_mode in {"path", "point"} and actor.spawn_point is not None:
                        _append_debug_log(
                            debug_log,
                            f"Spawn failed for {actor.label} at freeform point ({actor.spawn_point.x:.2f}, {actor.spawn_point.y:.2f}).",
                        )
                        raise RuntimeError(
                            f"Failed to spawn vehicle {actor.label} at ({actor.spawn_point.x:.1f}, {actor.spawn_point.y:.1f})."
                        )
                    if _should_skip_vehicle_spawn_failure(actor, static_road_vehicle_count):
                        skipped_actor = {
                            "id": actor.id,
                            "label": actor.label,
                            "road_id": actor.spawn.road_id,
                            "section_id": actor.spawn.section_id,
                            "lane_id": actor.spawn.lane_id,
                            "s_fraction": actor.spawn.s_fraction,
                            "reason": "blocked_spawn",
                        }
                        skipped_actors.append(skipped_actor)
                        _append_debug_log(
                            debug_log,
                            (
                                f"Skipped actor {actor.label}: blocked spawn on road={actor.spawn.road_id} "
                                f"section={actor.spawn.section_id} lane={actor.spawn.lane_id} s_fraction={actor.spawn.s_fraction:.3f}."
                            ),
                        )
                        _put_message(
                            message_queue,
                            {
                                "kind": "stream",
                                "payload": {
                                    "frame": frame,
                                    "timestamp": time.time(),
                                    "actors": [],
                                    "warning": f"Skipped {actor.label}: no free spawn slot on road {actor.spawn.road_id}.",
                                    "skipped_actor": skipped_actor,
                                },
                            },
                        )
                        continue
                    _append_debug_log(
                        debug_log,
                        (
                            f"Spawn failed for {actor.label} on road={actor.spawn.road_id} "
                            f"section={actor.spawn.section_id} lane={actor.spawn.lane_id}."
                        ),
                    )
                    raise RuntimeError(f"Failed to spawn vehicle {actor.label} on road {actor.spawn.road_id}.")
                actors.append((actor, vehicle))
                _append_debug_log(debug_log, f"Spawned vehicle {actor.label} handle_id={int(vehicle.id)}.")
                actor_by_handle_id[int(vehicle.id)] = actor
                handle_by_actor_id[actor.id] = vehicle
                if fallback_vehicle is None:
                    fallback_vehicle = vehicle
                if actor.role == "ego" and ego_vehicle is None:
                    ego_vehicle = vehicle
                timeline_states[actor.id].authored_route_locations = (
                    _route_locations_for_actor(
                        waypoint_index,
                        actor,
                        exact_waypoint_index=exact_spawn_waypoint_index,
                        carla_map=carla_map,
                        road_length_lookup=road_lengths,
                    )
                    if actor.placement_mode == "road" and not actor.is_static and actor.route
                    else None
                )
                timeline_states[actor.id].authored_route_applied = False
                if actor.placement_mode == "road" and not actor.is_static and timeline_states[actor.id].authored_route_locations:
                    route_vehicle_targets.append(
                        {
                            "vehicle": vehicle,
                            "locations": list(timeline_states[actor.id].authored_route_locations or []),
                            "target_speed_mps": max(1.0, actor.speed_kph / 3.6),
                            "reverse": str(getattr(actor, "route_direction", "forward")) == "reverse",
                            "index": 0,
                        }
                    )
                if actor.placement_mode == "path":
                    if actor.destination_point is not None:
                        destination_location = _location_from_map_point(world, actor.destination_point, z_offset=0.0)
                    else:
                        destination_wp = _resolve_anchor(
                            waypoint_index,
                            actor,
                            "destination",
                            exact_waypoint_index=exact_spawn_waypoint_index,
                            carla_map=carla_map,
                            road_length_lookup=road_lengths,
                        )
                        destination_location = destination_wp.transform.location
                    _set_vehicle_autopilot(vehicle, False, settings["tm_port"])
                    path_vehicle_targets.append(
                        (vehicle, destination_location, max(1.0, actor.speed_kph / 3.6))
                    )
                elif actor.placement_mode == "point" or actor.is_static:
                    _set_vehicle_autopilot(vehicle, False, settings["tm_port"])
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                elif actor.autopilot:
                    _set_vehicle_autopilot(vehicle, True, settings["tm_port"])
                    timeline_states[actor.id].autopilot_applied = True
                    timeline_states[actor.id].authored_route_applied = False
                else:
                    _set_vehicle_autopilot(vehicle, False, settings["tm_port"])
            else:
                if actor.placement_mode in {"path", "point"} and actor.spawn_point is not None:
                    transform = _walker_transform_from_points(world, actor.spawn_point, actor.destination_point)
                else:
                    anchor = _resolve_anchor(
                        waypoint_index,
                        actor,
                        "spawn",
                        exact_waypoint_index=exact_spawn_waypoint_index,
                        carla_map=carla_map,
                        road_length_lookup=road_lengths,
                    )
                    transform = _raised_waypoint_transform(
                        anchor,
                        z_offset=0.2,
                        against_lane=(
                            actor.kind == "vehicle"
                            and actor.placement_mode == "road"
                            and str(getattr(actor, "lane_facing", "with_lane")) == "against_lane"
                        ),
                    )
                blueprints = blueprint_library.filter(actor.blueprint)
                if not blueprints:
                    blueprints = blueprint_library.filter("walker.pedestrian.*")
                blueprint = blueprints[0]
                walker = world.try_spawn_actor(blueprint, transform)
                if walker is None:
                    if actor.placement_mode in {"path", "point"} and actor.spawn_point is not None:
                        _append_debug_log(
                            debug_log,
                            f"Spawn failed for walker {actor.label} at freeform point ({actor.spawn_point.x:.2f}, {actor.spawn_point.y:.2f}).",
                        )
                        raise RuntimeError(
                            f"Failed to spawn walker {actor.label} at ({actor.spawn_point.x:.1f}, {actor.spawn_point.y:.1f})."
                        )
                    _append_debug_log(
                        debug_log,
                        (
                            f"Spawn failed for walker {actor.label} on road={actor.spawn.road_id} "
                            f"section={actor.spawn.section_id} lane={actor.spawn.lane_id}."
                        ),
                    )
                    raise RuntimeError(f"Failed to spawn walker {actor.label} on road {actor.spawn.road_id}.")
                if actor.placement_mode == "path" and actor.destination_point is not None:
                    destination_location = _location_from_map_point(world, actor.destination_point, z_offset=0.0)
                    path_walker_targets.append((walker, destination_location, max(0.5, actor.speed_kph / 3.6)))
                else:
                    controller_bp = blueprint_library.find("controller.ai.walker")
                    controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                    if controller:
                        controller.start()
                        controller.set_max_speed(max(0.5, actor.speed_kph / 3.6))
                        destination_wp = _resolve_anchor(
                            waypoint_index,
                            actor,
                            "destination",
                            exact_waypoint_index=exact_spawn_waypoint_index,
                            carla_map=carla_map,
                            road_length_lookup=road_lengths,
                        )
                        controller.go_to_location(destination_wp.transform.location)
                        walker_controllers.append(controller)
                actors.append((actor, walker))
                _append_debug_log(debug_log, f"Spawned walker {actor.label} handle_id={int(walker.id)}.")
                actor_by_handle_id[int(walker.id)] = actor
                handle_by_actor_id[actor.id] = walker

        if request.topdown_recording:
            sensor_bp = blueprint_library.find("sensor.camera.rgb")
            sensor_bp.set_attribute("image_size_x", str(request.recording_width))
            sensor_bp.set_attribute("image_size_y", str(request.recording_height))
            sensor_bp.set_attribute("fov", str(request.recording_fov))
            camera_parent = ego_vehicle or fallback_vehicle
            if camera_parent is None:
                _append_debug_log(debug_log, "Recording requested but no vehicle actor was available for the chase camera.")
                raise RuntimeError("Recording requires an ego or vehicle actor in the scenario.")
            sensor = world.spawn_actor(sensor_bp, _chase_camera_transform(), attach_to=camera_parent)
            sensor_queue = queue.Queue()
            sensor.listen(sensor_queue.put)
            _append_debug_log(debug_log, f"Camera sensor attached to actor_id={int(camera_parent.id)}.")
            try:
                world.get_spectator().set_transform(sensor.get_transform())
            except Exception:
                pass

        max_steps = int(request.duration_seconds / request.fixed_delta_seconds)
        for step in range(max_steps):
            if stop_event.is_set():
                break
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.1)
            frame = world.tick()
            simulation_time = step * request.fixed_delta_seconds

            for actor_draft, handle in actors:
                if not handle or not handle.is_alive or actor_draft.kind != "vehicle":
                    continue
                if actor_draft.placement_mode not in {"road", "point"}:
                    continue
                timeline_state = timeline_states.get(actor_draft.id)
                if timeline_state is None:
                    continue
                directive = _evaluate_timeline(actor_draft, timeline_state, simulation_time)
                desired_autopilot = (
                    directive.autopilot_enabled
                    if actor_draft.placement_mode == "road" and not actor_draft.is_static
                    else False
                )
                has_direct_control = any(
                    (
                        directive.chase_target_actor_id is not None,
                        directive.ram_target_actor_id is not None,
                    )
                )
                has_authored_route = bool(timeline_state.authored_route_locations)
                route_follow_active = has_authored_route and directive.route_follow_enabled
                if actor_draft.placement_mode == "road" and not actor_draft.is_static:
                    if route_follow_active and not any(
                        (
                            directive.route_instruction is not None,
                            directive.lane_change_direction is not None,
                            directive.chase_target_actor_id is not None,
                            directive.ram_target_actor_id is not None,
                        )
                    ):
                        if timeline_state.autopilot_applied:
                            _set_vehicle_autopilot(handle, False, settings["tm_port"])
                            timeline_state.autopilot_applied = False
                            timeline_state.authored_route_applied = False
                        if directive.hold_position:
                            handle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False))
                        continue
                    if directive.lane_change_direction:
                        timeline_state.pending_lane_change = directive.lane_change_direction
                        timeline_state.lane_change_requests_remaining = max(
                            timeline_state.lane_change_requests_remaining,
                            int(round(1.5 / request.fixed_delta_seconds)),
                        )
                    if directive.route_instruction:
                        timeline_state.pending_route_instruction = directive.route_instruction
                        timeline_state.pending_route_origin_road_id = None
                        timeline_state.pending_route_locations = None
                    if desired_autopilot and not directive.hold_position and not has_direct_control:
                        if not timeline_state.autopilot_applied:
                            _set_vehicle_autopilot(handle, True, settings["tm_port"])
                            timeline_state.autopilot_applied = True
                            timeline_state.authored_route_applied = False
                        if (
                            timeline_state.pending_lane_change is not None
                            and timeline_state.lane_change_requests_remaining > 0
                        ):
                            _tm_force_lane_change(tm, handle, timeline_state.pending_lane_change)
                            timeline_state.lane_change_requests_remaining -= 1
                            if timeline_state.lane_change_requests_remaining <= 0:
                                timeline_state.pending_lane_change = None
                        if timeline_state.pending_route_instruction is not None:
                            current_waypoint = None
                            try:
                                current_waypoint = world.get_map().get_waypoint(handle.get_transform().location, project_to_road=True)
                            except Exception:
                                current_waypoint = None
                            current_road_id = int(current_waypoint.road_id) if current_waypoint is not None else None
                            if timeline_state.pending_route_origin_road_id is None and current_road_id is not None:
                                timeline_state.pending_route_origin_road_id = current_road_id
                            if timeline_state.pending_route_locations is None and current_waypoint is not None:
                                timeline_state.pending_route_locations = _find_turn_path_locations(
                                    world.get_map(),
                                    current_waypoint,
                                    timeline_state.pending_route_instruction,
                                    carla,
                                )
                            if timeline_state.pending_route_locations:
                                _tm_set_path(tm, handle, timeline_state.pending_route_locations)
                            else:
                                _tm_set_route(tm, handle, timeline_state.pending_route_instruction)
                            if (
                                timeline_state.pending_route_origin_road_id is not None
                                and current_road_id is not None
                                and current_road_id != timeline_state.pending_route_origin_road_id
                            ):
                                timeline_state.pending_route_instruction = None
                                timeline_state.pending_route_origin_road_id = None
                                timeline_state.pending_route_locations = None
                                timeline_state.authored_route_applied = False
                    else:
                        if timeline_state.autopilot_applied:
                            _set_vehicle_autopilot(handle, False, settings["tm_port"])
                            timeline_state.autopilot_applied = False
                            timeline_state.authored_route_applied = False
                        if directive.hold_position:
                            handle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False))
                        elif directive.chase_target_actor_id or directive.ram_target_actor_id:
                            target_actor_id = directive.ram_target_actor_id or directive.chase_target_actor_id
                            target_handle = handle_by_actor_id.get(target_actor_id or "")
                            if target_handle is not None and getattr(target_handle, "is_alive", False):
                                _apply_target_vehicle_control(
                                    carla,
                                    handle,
                                    target_handle.get_transform().location,
                                    directive.target_speed_mps if directive.target_speed_mps is not None else actor_draft.speed_kph / 3.6,
                                    aggressive=directive.ram_target_actor_id is not None,
                                    target_velocity_mps=math.sqrt(
                                        float(target_handle.get_velocity().x) ** 2
                                        + float(target_handle.get_velocity().y) ** 2
                                        + float(target_handle.get_velocity().z) ** 2
                                    ),
                                    follow_distance_m=directive.follow_distance_m,
                                )
                            else:
                                handle.apply_control(
                                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)
                                )
                else:
                    handle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))

            remaining_path_targets: list[tuple[Any, Any, float]] = []
            for vehicle, target_location, target_speed_mps in path_vehicle_targets:
                if not vehicle or not vehicle.is_alive:
                    continue
                matched_actor = actor_by_handle_id.get(int(vehicle.id))
                timeline_state = timeline_states.get(matched_actor.id) if matched_actor else None
                directive = (
                    _evaluate_timeline(matched_actor, timeline_state, simulation_time)
                    if matched_actor is not None and timeline_state is not None
                    else TimelineDirective()
                )
                if directive.hold_position:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                    remaining_path_targets.append((vehicle, target_location, target_speed_mps))
                    continue
                desired_speed_mps = (
                    directive.target_speed_mps
                    if directive.target_speed_mps is not None
                    else target_speed_mps
                )
                reached = _apply_path_vehicle_control(carla, vehicle, target_location, desired_speed_mps)
                if not reached:
                    remaining_path_targets.append((vehicle, target_location, target_speed_mps))
            path_vehicle_targets = remaining_path_targets

            remaining_route_targets: list[dict[str, Any]] = []
            for route_target in route_vehicle_targets:
                vehicle = route_target.get("vehicle")
                locations = route_target.get("locations") or []
                index = int(route_target.get("index") or 0)
                target_speed_mps = float(route_target.get("target_speed_mps") or 0.0)
                reverse = bool(route_target.get("reverse"))
                if not vehicle or not getattr(vehicle, "is_alive", False) or not locations or index >= len(locations):
                    continue
                matched_actor = actor_by_handle_id.get(int(vehicle.id))
                timeline_state = timeline_states.get(matched_actor.id) if matched_actor else None
                directive = (
                    _evaluate_timeline(matched_actor, timeline_state, simulation_time)
                    if matched_actor is not None and timeline_state is not None
                    else TimelineDirective()
                )
                if not directive.route_follow_enabled:
                    remaining_route_targets.append(route_target)
                    continue
                if any(
                    (
                        directive.route_instruction is not None,
                        directive.lane_change_direction is not None,
                        directive.chase_target_actor_id is not None,
                        directive.ram_target_actor_id is not None,
                    )
                ):
                    remaining_route_targets.append(route_target)
                    continue
                if directive.hold_position:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                    remaining_route_targets.append(route_target)
                    continue
                desired_speed_mps = directive.target_speed_mps if directive.target_speed_mps is not None else target_speed_mps
                is_final_target = index >= len(locations) - 1
                reached = _apply_path_vehicle_control(
                    carla,
                    vehicle,
                    locations[index],
                    desired_speed_mps,
                    stop_at_target=is_final_target,
                    arrival_distance_m=3.0 if is_final_target else 5.0,
                    reverse=reverse,
                )
                if reached:
                    route_target["index"] = index + 1
                if int(route_target.get("index") or 0) < len(locations):
                    remaining_route_targets.append(route_target)
            route_vehicle_targets = remaining_route_targets
            remaining_walker_targets: list[tuple[Any, Any, float]] = []
            for walker, target_location, target_speed_mps in path_walker_targets:
                if not walker or not walker.is_alive:
                    continue
                matched_actor = actor_by_handle_id.get(int(walker.id))
                timeline_state = timeline_states.get(matched_actor.id) if matched_actor else None
                directive = (
                    _evaluate_timeline(matched_actor, timeline_state, simulation_time)
                    if matched_actor is not None and timeline_state is not None
                    else TimelineDirective()
                )
                desired_speed_mps = (
                    directive.target_speed_mps
                    if directive.target_speed_mps is not None
                    else target_speed_mps
                )
                if directive.hold_position:
                    walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=0.0, y=0.0, z=0.0), speed=0.0))
                    remaining_walker_targets.append((walker, target_location, target_speed_mps))
                    continue
                reached = _apply_path_walker_control(carla, walker, target_location, desired_speed_mps)
                if not reached:
                    remaining_walker_targets.append((walker, target_location, target_speed_mps))
            path_walker_targets = remaining_walker_targets
            if sensor_queue is not None:
                try:
                    image = sensor_queue.get(timeout=2.0)
                    image.save_to_disk(str(frames_dir / f"{image.frame:06d}.png"))
                    saved_frame_count += 1
                    last_sensor_frame = int(image.frame)
                except queue.Empty:
                    sensor_timeout_count += 1
            if sensor and sensor.is_alive:
                try:
                    world.get_spectator().set_transform(sensor.get_transform())
                except Exception:
                    pass
            actor_states: list[dict[str, Any]] = []
            for actor_draft, handle in actors:
                if not handle or not handle.is_alive:
                    continue
                transform = handle.get_transform()
                velocity = handle.get_velocity()
                waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True)
                coords = carla_to_frontend(transform.location, transform.rotation)
                actor_states.append(
                    SimulationActorState(
                        id=int(handle.id),
                        label=actor_draft.label,
                        kind=actor_draft.kind,
                        role=actor_draft.role,
                        x=coords["x"],
                        y=coords["y"],
                        z=coords["z"],
                        yaw=coords["yaw"],
                        speed_mps=math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2),
                        road_id=int(waypoint.road_id) if waypoint else None,
                        section_id=int(getattr(waypoint, "section_id", 0)) if waypoint else None,
                        lane_id=int(waypoint.lane_id) if waypoint else None,
                    ).model_dump()
                )
            _put_message(
                message_queue,
                {
                    "kind": "stream",
                    "payload": SimulationStreamMessage(
                        frame=frame,
                        timestamp=step * request.fixed_delta_seconds,
                        actors=[SimulationActorState.model_validate(item) for item in actor_states],
                    ).model_dump(),
                },
            )
    except Exception as exc:  # noqa: BLE001
        worker_error = str(exc)
        _append_debug_log(debug_log, f"Worker error: {exc}")
        _append_debug_log(debug_log, traceback.format_exc())
        _put_message(
            message_queue,
            {
                "kind": "stream",
                "payload": SimulationStreamMessage(
                    frame=frame,
                    timestamp=time.time(),
                    actors=[],
                    error=str(exc),
                ).model_dump(),
            },
        )
    finally:
        _append_debug_log(debug_log, "Finalizing simulation run.")
        client = None
        try:
            client = _make_client(settings["carla_host"], settings["carla_port"], settings["carla_timeout"])
        except Exception as exc:  # noqa: BLE001
            _append_debug_log(debug_log, f"Failed to reconnect CARLA client during finalization: {exc}")

        if sensor and sensor.is_alive:
            try:
                _append_debug_log(debug_log, f"Stopping camera sensor actor_id={int(sensor.id)}.")
                sensor.stop()
            except Exception as exc:  # noqa: BLE001
                _append_debug_log(debug_log, f"Failed to stop camera sensor cleanly: {exc}")

        if world and original_settings is not None:
            try:
                _append_debug_log(debug_log, "Restoring original CARLA world settings.")
                world.apply_settings(original_settings)
            except Exception as exc:  # noqa: BLE001
                _append_debug_log(debug_log, f"Failed to restore CARLA world settings: {exc}")

        if tm is not None:
            try:
                _append_debug_log(debug_log, "Disabling Traffic Manager synchronous mode.")
                tm.set_synchronous_mode(False)
            except Exception as exc:  # noqa: BLE001
                _append_debug_log(debug_log, f"Failed to disable Traffic Manager synchronous mode: {exc}")

        if client is not None:
            try:
                _append_debug_log(debug_log, "Stopping CARLA recorder.")
                client.stop_recorder()
            except Exception as exc:  # noqa: BLE001
                _append_debug_log(debug_log, f"Failed to stop CARLA recorder: {exc}")

        if client is not None:
            destroy_handles: list[Any] = []
            if sensor is not None:
                destroy_handles.append(sensor)
            destroy_handles.extend(controller for controller in walker_controllers if controller is not None)
            destroy_handles.extend(actor for _, actor in actors if actor is not None)
            _destroy_carla_handles(client, destroy_handles, debug_log)

        recording = None
        if frames_dir.exists() and any(frames_dir.glob("*.png")):
            try:
                fps = int(round(1.0 / request.fixed_delta_seconds))
                _encode_mp4(frames_dir, recording_path, fps)
                recording = RecordingInfo(
                    run_id=run_id,
                    label=f"{normalize_map_name(request.map_name)} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    mp4_path=str(recording_path),
                    frames_path=str(frames_dir),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ).model_dump()
                _append_debug_log(
                    debug_log,
                    f"MP4 encoded successfully frames={saved_frame_count} fps={fps} path={recording_path}.",
                )
            except Exception as exc:  # noqa: BLE001
                error_path = run_dir / "encoding_error.txt"
                error_path.write_text(str(exc), encoding="utf-8")
                _append_debug_log(debug_log, f"MP4 encoding failed: {exc}")

        manifest = {
            "map_name": request.map_name,
            "selected_roads": [road.model_dump() for road in request.selected_roads],
            "actors": [actor.model_dump() for actor in request.actors],
            "duration_seconds": request.duration_seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recording_path": str(recording_path) if recording else None,
            "scenario_log": str(scenario_log) if scenario_log else None,
            "debug_log": str(debug_log),
            "saved_frame_count": saved_frame_count,
            "sensor_timeout_count": sensor_timeout_count,
            "last_sensor_frame": last_sensor_frame,
            "worker_error": worker_error,
            "skipped_actors": skipped_actors,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _append_debug_log(
            debug_log,
            (
                f"Run finished worker_error={worker_error!r} saved_frames={saved_frame_count} "
                f"sensor_timeouts={sensor_timeout_count} skipped_actors={len(skipped_actors)}."
            ),
        )
        _put_message(
            message_queue,
            {
                "kind": "stream",
                "payload": SimulationStreamMessage(
                    frame=frame,
                    timestamp=time.time(),
                    actors=[],
                    simulation_ended=True,
                    recording=RecordingInfo.model_validate(recording) if recording else None,
                    skipped_actors=skipped_actors,
                ).model_dump(),
            },
        )


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: set[Any] = set()

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: Any) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.connections:
            return
        dead: list[Any] = []
        message = json.dumps(payload)
        for websocket in self.connections:
            try:
                await websocket.send_text(message)
            except Exception:  # noqa: BLE001
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)


class CarlaSimulationService:
    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager
        self.carla_host = os.environ.get("CARLA_HOST", "127.0.0.1")
        self.carla_port = int(os.environ.get("CARLA_PORT", "2000"))
        self.carla_timeout = float(os.environ.get("CARLA_TIMEOUT", "20"))
        self.tm_port = int(os.environ.get("CARLA_TM_PORT", "8000"))
        self.output_root = Path(
            os.environ.get(
                "VW_INTERACTIVE_OUTPUT_ROOT",
                str(Path(__file__).resolve().parents[2] / "visualizations" / "interactive_runs"),
            )
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._is_running = False
        self._recordings: list[RecordingInfo] = []
        self._last_recording: RecordingInfo | None = None
        self._process: mp.Process | None = None
        self._listener_thread: threading.Thread | None = None
        self._queue: Any | None = None
        self._stop_event: Any | None = None
        self._pause_event: Any | None = None
        self._stop_requested = False

    def _allocate_tm_port(self, search_span: int = 64) -> int:
        for offset in range(search_span):
            port = self.tm_port + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError(f"No free CARLA Traffic Manager port found in range {self.tm_port}-{self.tm_port + search_span - 1}.")

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def is_running(self) -> bool:
        process = self._process
        if process is not None and not process.is_alive() and self._is_running:
            self._set_running(False)
        with self._lock:
            return self._is_running

    def _set_running(self, value: bool) -> None:
        with self._lock:
            self._is_running = value

    def _client(self) -> Any:
        return _make_client(self.carla_host, self.carla_port, self.carla_timeout)

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
            warnings.append(str(exc))
            return CarlaStatusResponse(
                connected=False,
                supported_dataset_maps=sorted(list_supported_maps()),
                warnings=warnings,
            )

    def list_recordings(self) -> list[RecordingInfo]:
        return sorted(self._recordings, key=lambda item: item.created_at, reverse=True)

    def last_recording(self) -> RecordingInfo | None:
        return self._last_recording

    def _run_manifest_paths(self) -> list[Path]:
        return sorted(self.output_root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)

    def _read_run_diagnostics(self, manifest_path: Path) -> SimulationRunDiagnostics:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest_path.parent.name
        debug_log_path = data.get("debug_log")
        log_excerpt = ""
        if debug_log_path:
            debug_path = Path(debug_log_path)
            if debug_path.is_file():
                try:
                    lines = debug_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    log_excerpt = "\n".join(lines[-80:])
                except Exception:
                    log_excerpt = ""
        return SimulationRunDiagnostics(
            run_id=run_id,
            map_name=str(data.get("map_name") or ""),
            created_at=str(data.get("created_at") or ""),
            selected_roads=[SelectedRoad.model_validate(item) for item in list(data.get("selected_roads") or [])],
            actors=[ActorDraft.model_validate(item) for item in list(data.get("actors") or [])],
            recording_path=data.get("recording_path"),
            scenario_log_path=data.get("scenario_log"),
            debug_log_path=debug_log_path,
            worker_error=data.get("worker_error"),
            saved_frame_count=int(data.get("saved_frame_count") or 0),
            sensor_timeout_count=int(data.get("sensor_timeout_count") or 0),
            last_sensor_frame=data.get("last_sensor_frame"),
            skipped_actors=list(data.get("skipped_actors") or []),
            log_excerpt=log_excerpt,
        )

    def latest_run_diagnostics(self) -> SimulationRunDiagnostics | None:
        for manifest_path in self._run_manifest_paths():
            try:
                return self._read_run_diagnostics(manifest_path)
            except Exception:
                continue
        return None

    def run_diagnostics(self, run_id: str) -> SimulationRunDiagnostics | None:
        manifest_path = self.output_root / run_id / "manifest.json"
        if not manifest_path.is_file():
            return None
        return self._read_run_diagnostics(manifest_path)

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
                    coords = carla_to_frontend(waypoint.transform.location, waypoint.transform.rotation)
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

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_requested = True
            self._stop_event.set()
            threading.Thread(target=self._force_stop_worker_after_timeout, daemon=True).start()

    def pause(self) -> None:
        if self._pause_event is not None:
            self._pause_event.set()

    def resume(self) -> None:
        if self._pause_event is not None:
            self._pause_event.clear()

    def _publish(self, message: SimulationStreamMessage) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self.manager.broadcast(message.model_dump()), self._loop)

    def run(self, request: SimulationRunRequest) -> None:
        if self.is_running():
            raise RuntimeError("A simulation is already running.")
        normalized_request = request.model_copy(update={"selected_roads": _canonical_selected_roads_for_request(request)})
        tm_port = self._allocate_tm_port()

        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._pause_event = ctx.Event()
        self._process = ctx.Process(
            target=_simulation_worker,
            args=(
                normalized_request.model_dump(),
                {
                    "carla_host": self.carla_host,
                    "carla_port": self.carla_port,
                    "carla_timeout": self.carla_timeout,
                    "tm_port": tm_port,
                    "output_root": str(self.output_root),
                },
                self._queue,
                self._stop_event,
                self._pause_event,
            ),
            daemon=True,
        )
        self._process.start()
        self._stop_requested = False
        self._set_running(True)
        self._listener_thread = threading.Thread(target=self._consume_worker_messages, daemon=True)
        self._listener_thread.start()

    def _force_stop_worker_after_timeout(self, timeout_seconds: float = 5.0) -> None:
        time.sleep(timeout_seconds)
        process = self._process
        if process is None or not process.is_alive():
            return
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        self._publish(
            SimulationStreamMessage(
                frame=0,
                timestamp=time.time(),
                actors=[],
                error="Simulation was force-stopped after the worker stopped responding.",
                simulation_ended=True,
            )
        )
        self._cleanup_worker_state()

    def _consume_worker_messages(self) -> None:
        saw_terminal_message = False
        process = self._process
        message_queue = self._queue

        while True:
            got_message = False
            if message_queue is not None:
                try:
                    envelope = message_queue.get(timeout=0.5)
                    got_message = True
                except queue.Empty:
                    envelope = None
                except Exception:
                    envelope = None
            else:
                envelope = None

            if envelope and envelope.get("kind") == "stream":
                message = SimulationStreamMessage.model_validate(envelope["payload"])
                if message.recording:
                    self._recordings.append(message.recording)
                    self._last_recording = message.recording
                if message.simulation_ended or message.error:
                    saw_terminal_message = True
                self._publish(message)

            alive = process.is_alive() if process is not None else False
            if not alive and not got_message:
                break

        exit_code = process.exitcode if process is not None else None
        if self._stop_requested and not saw_terminal_message:
            self._publish(
                SimulationStreamMessage(
                    frame=0,
                    timestamp=time.time(),
                    actors=[],
                    simulation_ended=True,
                )
            )
        elif exit_code not in {0, None} and not saw_terminal_message:
            self._publish(
                SimulationStreamMessage(
                    frame=0,
                    timestamp=time.time(),
                    actors=[],
                    error=f"Simulation worker exited unexpectedly with code {exit_code}.",
                    simulation_ended=True,
                )
            )

        self._cleanup_worker_state()

    def _cleanup_worker_state(self) -> None:
        self._set_running(False)
        self._process = None
        self._listener_thread = None
        self._queue = None
        self._stop_event = None
        self._pause_event = None
        self._stop_requested = False
