from __future__ import annotations

import json
import math
import os
import re
import uuid
from copy import deepcopy
from textwrap import dedent
from typing import Any

import boto3
from pydantic import BaseModel, Field

from .langchain_support import (
    LANGCHAIN_AVAILABLE,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    StructuredTool,
    create_chat_model,
    langsmith_run_config,
    serialize_ai_message,
    traceable,
)

from ..carla_runner.dataset_repository import build_selected_roads
from ..carla_runner.dataset_repository import search_maps_by_road as dataset_search_maps_by_road
from ..carla_runner.dataset_repository import search_roads as dataset_search_roads
from ..carla_runner.models import (
    ActorDraft,
    ActorMapPoint,
    ActorRoadAnchor,
    ActorTimelineClip,
    RuntimeRoadSegment,
    SceneAssistantRequest,
    SceneAssistantResponse,
    SceneAssistantToolTrace,
    SelectedRoad,
    SimulationActorState,
)


DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("BEDROCK_MODEL_ID", "").strip(),
    "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
]

DEFAULT_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
DEFAULT_WALKER_BLUEPRINT = "walker.pedestrian.0001"


class SceneAssistantGetSceneOverviewInput(BaseModel):
    pass


class SceneAssistantGetActorInput(BaseModel):
    actor_ref: str


class SceneAssistantGetRoadInput(BaseModel):
    road_id: str


class SceneAssistantGetAdjacentLanesInput(BaseModel):
    road_id: str
    section_id: int
    lane_id: int


class SceneAssistantFindNearestLaneInput(BaseModel):
    x: float
    y: float


class SceneAssistantApplyEditsInput(BaseModel):
    operations: list[dict[str, Any]] = Field(default_factory=list)


class SceneAssistantFindRoadsInput(BaseModel):
    query: str = ""
    tags: list[str] = Field(default_factory=list)
    lane_types: list[str] = Field(default_factory=list)
    is_intersection: bool | None = None
    has_parking: bool | None = None
    driving_left: int | None = None
    driving_right: int | None = None
    total_driving: int | None = None
    parking_left_min: int | None = None
    parking_right_min: int | None = None
    require_parking_on_both_sides: bool | None = None
    limit: int = Field(default=12, ge=1, le=50)


class SceneAssistantSearchMapsByRoadInput(BaseModel):
    query: str = ""
    tags: list[str] = Field(default_factory=list)
    lane_types: list[str] = Field(default_factory=list)
    is_intersection: bool | None = None
    has_parking: bool | None = None
    driving_left: int | None = None
    driving_right: int | None = None
    total_driving: int | None = None
    parking_left_min: int | None = None
    parking_right_min: int | None = None
    require_parking_on_both_sides: bool | None = None
    map_limit: int = Field(default=8, ge=1, le=20)
    roads_per_map_limit: int = Field(default=5, ge=1, le=20)


class SceneAssistantSwitchMapInput(BaseModel):
    map_name: str


def _project_point_to_line_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
) -> dict[str, float]:
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-9:
        offset_x = float(point["x"]) - float(start["x"])
        offset_y = float(point["y"]) - float(start["y"])
        return {
            "x": float(start["x"]),
            "y": float(start["y"]),
            "t": 0.0,
            "distance_squared": offset_x * offset_x + offset_y * offset_y,
        }

    raw_t = (
        ((float(point["x"]) - float(start["x"])) * dx)
        + ((float(point["y"]) - float(start["y"])) * dy)
    ) / length_squared
    t = max(0.0, min(1.0, raw_t))
    x = float(start["x"]) + dx * t
    y = float(start["y"]) + dy * t
    offset_x = float(point["x"]) - x
    offset_y = float(point["y"]) - y
    return {
        "x": x,
        "y": y,
        "t": t,
        "distance_squared": offset_x * offset_x + offset_y * offset_y,
    }


class SceneEditorState:
    def __init__(self, request: SceneAssistantRequest) -> None:
        self.map_name = request.map_name
        self.runtime_map = request.runtime_map
        self.selected_roads = [SelectedRoad.model_validate(item.model_dump()) for item in request.selected_roads]
        self.actors = [ActorDraft.model_validate(item.model_dump()) for item in request.actors]
        self.live_actors_by_label = {
            item.label.lower(): SimulationActorState.model_validate(item.model_dump())
            for item in request.live_actors
        }
        self.selected_actor_id = request.selected_actor_id
        self._rebuild_runtime_indexes()

    def _rebuild_runtime_indexes(self) -> None:
        self.road_summaries_by_id = {
            road.id: road for road in getattr(self.runtime_map, "road_summaries", [])
        }
        self.segments_by_id = {segment.id: segment for segment in self.runtime_map.road_segments}
        self.segments_by_key: dict[tuple[str, int, int], RuntimeRoadSegment] = {}
        self.segments_by_road: dict[str, list[RuntimeRoadSegment]] = {}
        for segment in self.runtime_map.road_segments:
            road_id = str(segment.road_id)
            self.segments_by_key[(road_id, segment.section_id, segment.lane_id)] = segment
            self.segments_by_road.setdefault(road_id, []).append(segment)

    def replace_runtime_context(self, map_name: str, runtime_map: Any) -> None:
        self.map_name = map_name
        self.runtime_map = runtime_map
        self.selected_roads = []
        self.actors = []
        self.live_actors_by_label = {}
        self.selected_actor_id = None
        self._rebuild_runtime_indexes()

    def scene_overview(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "selected_actor_id": self.selected_actor_id,
            "selected_roads": [
                {
                    "id": road.id,
                    "name": road.name,
                    "tags": road.tags,
                    "section_labels": road.section_labels,
                }
                for road in self.selected_roads
            ],
            "actors": [self._actor_summary(actor) for actor in self.actors],
            "live_actor_count": len(self.live_actors_by_label),
            "runtime_road_segment_count": len(self.runtime_map.road_segments),
            "dataset_augmented": bool(getattr(self.runtime_map, "dataset_augmented", False)),
        }

    def actor_details(self, actor_ref: str) -> dict[str, Any]:
        actor = self._resolve_actor(actor_ref)
        return self._actor_summary(actor, include_timeline=True, include_anchor_details=True)

    def road_details(self, road_id: str) -> dict[str, Any]:
        road_key = str(road_id)
        summary = self.road_summaries_by_id.get(road_key)
        segments = sorted(
            self.segments_by_road.get(road_key, []),
            key=lambda item: (item.section_id, item.lane_id),
        )
        if summary is None and not segments:
            raise RuntimeError(f"Road {road_key} is not available in the runtime map.")
        return {
            "road_id": road_key,
            "summary": summary.model_dump() if summary is not None else None,
            "segments": [
                {
                    "section_id": segment.section_id,
                    "lane_id": segment.lane_id,
                    "lane_type": segment.lane_type,
                    "is_junction": segment.is_junction,
                    "left_lane_id": segment.left_lane_id,
                    "right_lane_id": segment.right_lane_id,
                    "point_count": len(segment.centerline),
                }
                for segment in segments
            ],
        }

    def adjacent_lanes(self, road_id: str, section_id: int, lane_id: int) -> dict[str, Any]:
        segment = self._segment_for_anchor(
            ActorRoadAnchor(
                road_id=str(road_id),
                section_id=int(section_id),
                lane_id=int(lane_id),
                s_fraction=0.5,
            )
        )
        if segment is None:
            raise RuntimeError(
                f"Road {road_id} section {section_id} lane {lane_id} is not available in the runtime map."
            )
        left_lane_id, right_lane_id = self._adjacent_lane_ids(segment)
        return {
            "road_id": str(segment.road_id),
            "section_id": segment.section_id,
            "lane_id": segment.lane_id,
            "left_lane_id": left_lane_id,
            "right_lane_id": right_lane_id,
            "left_lane": self._lane_descriptor(str(segment.road_id), segment.section_id, left_lane_id)
            if left_lane_id is not None
            else None,
            "right_lane": self._lane_descriptor(str(segment.road_id), segment.section_id, right_lane_id)
            if right_lane_id is not None
            else None,
        }

    def nearest_lane(self, x: float, y: float) -> dict[str, Any]:
        point = {"x": float(x), "y": float(y)}
        best: dict[str, Any] | None = None
        for segment in self.runtime_map.road_segments:
            if len(segment.centerline) < 2:
                continue
            for index in range(len(segment.centerline) - 1):
                start = segment.centerline[index]
                end = segment.centerline[index + 1]
                projection = _project_point_to_line_segment(point, start, end)
                if best is None or float(projection["distance_squared"]) < float(best["distance_squared"]):
                    start_s = float(start.get("s", index))
                    end_s = float(end.get("s", index + 1))
                    segment_s = start_s + ((end_s - start_s) * float(projection["t"]))
                    first_s = float(segment.centerline[0].get("s", 0.0))
                    last_s = float(segment.centerline[-1].get("s", len(segment.centerline) - 1))
                    total_s = max(1e-6, last_s - first_s)
                    best = {
                        "segment": segment,
                        "distance_squared": float(projection["distance_squared"]),
                        "s_fraction": max(0.0, min(1.0, (segment_s - first_s) / total_s)),
                    }
        if best is None:
            raise RuntimeError("The runtime map has no lane geometry to project against.")
        segment = best["segment"]
        return {
            "road_id": str(segment.road_id),
            "section_id": segment.section_id,
            "lane_id": segment.lane_id,
            "lane_type": segment.lane_type,
            "s_fraction": round(float(best["s_fraction"]), 4),
            "distance": round(math.sqrt(float(best["distance_squared"])), 3),
        }

    def find_roads(self, criteria: dict[str, Any]) -> dict[str, Any]:
        limit = int(criteria.get("limit") or 12)
        matches = dataset_search_roads(
            self.map_name,
            query=str(criteria.get("query") or ""),
            tags=[str(item) for item in criteria.get("tags") or []],
            lane_types=[str(item) for item in criteria.get("lane_types") or []],
            is_intersection=criteria.get("is_intersection"),
            has_parking=criteria.get("has_parking"),
            driving_left=criteria.get("driving_left"),
            driving_right=criteria.get("driving_right"),
            total_driving=criteria.get("total_driving"),
            parking_left_min=criteria.get("parking_left_min"),
            parking_right_min=criteria.get("parking_right_min"),
            require_parking_on_both_sides=criteria.get("require_parking_on_both_sides"),
            limit=limit,
        )
        return {
            "map_name": self.map_name,
            "match_count": len(matches),
            "roads": matches,
        }

    def search_maps_by_road(self, criteria: dict[str, Any]) -> dict[str, Any]:
        matches = dataset_search_maps_by_road(
            query=str(criteria.get("query") or ""),
            tags=[str(item) for item in criteria.get("tags") or []],
            lane_types=[str(item) for item in criteria.get("lane_types") or []],
            is_intersection=criteria.get("is_intersection"),
            has_parking=criteria.get("has_parking"),
            driving_left=criteria.get("driving_left"),
            driving_right=criteria.get("driving_right"),
            total_driving=criteria.get("total_driving"),
            parking_left_min=criteria.get("parking_left_min"),
            parking_right_min=criteria.get("parking_right_min"),
            require_parking_on_both_sides=criteria.get("require_parking_on_both_sides"),
            map_limit=int(criteria.get("map_limit") or 8),
            roads_per_map_limit=int(criteria.get("roads_per_map_limit") or 5),
        )
        return {
            "current_map_name": self.map_name,
            "map_match_count": len(matches),
            "maps": matches,
        }

    def apply_operations(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for raw_operation in operations:
            operation = dict(raw_operation)
            op_type = str(operation.get("type") or "").strip()
            try:
                if op_type == "add_actor":
                    actor = self._add_actor(self._actor_payload_from_operation(operation), select=bool(operation.get("select", True)))
                    applied.append(
                        {
                            "type": op_type,
                            "actor_id": actor.id,
                            "label": actor.label,
                        }
                    )
                elif op_type == "add_actor_row":
                    row_operation = {
                        **operation,
                        "actor": self._actor_payload_from_operation(operation),
                    }
                    actors = self._add_actor_row(row_operation, select=bool(operation.get("select", True)))
                    applied.append(
                        {
                            "type": op_type,
                            "count": len(actors),
                            "actor_ids": [actor.id for actor in actors],
                            "labels": [actor.label for actor in actors],
                        }
                    )
                elif op_type == "update_actor":
                    actor = self._update_actor(
                        str(operation.get("actor_id") or ""),
                        self._changes_payload_from_operation(operation),
                    )
                    applied.append(
                        {
                            "type": op_type,
                            "actor_id": actor.id,
                            "label": actor.label,
                        }
                    )
                elif op_type == "remove_actor":
                    removed = self._remove_actor(str(operation.get("actor_id") or ""))
                    applied.append({"type": op_type, "actor_id": removed.id, "label": removed.label})
                elif op_type == "replace_timeline":
                    actor = self._replace_timeline(
                        str(operation.get("actor_id") or ""),
                        operation.get("timeline") or [],
                    )
                    applied.append({"type": op_type, "actor_id": actor.id, "timeline_count": len(actor.timeline)})
                elif op_type == "add_timeline_clip":
                    actor = self._add_timeline_clip(
                        str(operation.get("actor_id") or ""),
                        operation.get("clip") or {},
                    )
                    applied.append({"type": op_type, "actor_id": actor.id, "timeline_count": len(actor.timeline)})
                elif op_type == "remove_timeline_clip":
                    actor = self._remove_timeline_clip(
                        str(operation.get("actor_id") or ""),
                        str(operation.get("clip_id") or ""),
                    )
                    applied.append({"type": op_type, "actor_id": actor.id, "timeline_count": len(actor.timeline)})
                elif op_type == "select_actor":
                    actor = self._resolve_actor(str(operation.get("actor_id") or ""))
                    self.selected_actor_id = actor.id
                    applied.append({"type": op_type, "actor_id": actor.id, "label": actor.label})
                elif op_type == "add_selected_roads":
                    added = self._add_selected_roads([str(item) for item in operation.get("road_ids") or []])
                    applied.append({"type": op_type, "road_ids": added})
                elif op_type == "remove_selected_roads":
                    removed_ids = self._remove_selected_roads([str(item) for item in operation.get("road_ids") or []])
                    applied.append({"type": op_type, "road_ids": removed_ids})
                elif op_type == "set_selected_roads":
                    next_ids = [str(item) for item in operation.get("road_ids") or []]
                    self._set_selected_roads(next_ids)
                    applied.append({"type": op_type, "road_ids": next_ids})
                else:
                    raise RuntimeError(f"Unsupported scene edit operation: {op_type or 'missing type'}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"operation": operation, "error": str(exc)})
        return {
            "applied": applied,
            "errors": errors,
            "selected_actor_id": self.selected_actor_id,
            "selected_roads": [road.id for road in self.selected_roads],
            "actors": [self._actor_summary(actor) for actor in self.actors],
        }

    def _actor_payload_from_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(operation.get("actor") or {})
        ignored_keys = {
            "type",
            "actor",
            "changes",
            "clip",
            "timeline",
            "clip_id",
            "actor_id",
            "road_ids",
            "count",
            "lane_ids",
            "s_start",
            "s_end",
            "is_relative_s",
            "label_prefix",
            "select",
        }
        for key, value in operation.items():
            if key in ignored_keys:
                continue
            payload.setdefault(key, value)
        if "actor_type" in payload and "kind" not in payload:
            payload["kind"] = payload.pop("actor_type")
        return payload

    def _changes_payload_from_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(operation.get("changes") or {})
        ignored_keys = {
            "type",
            "actor",
            "changes",
            "clip",
            "timeline",
            "clip_id",
            "actor_id",
            "road_ids",
            "count",
            "lane_ids",
            "s_start",
            "s_end",
            "is_relative_s",
            "label_prefix",
            "select",
        }
        for key, value in operation.items():
            if key in ignored_keys:
                continue
            payload.setdefault(key, value)
        if "actor_type" in payload and "kind" not in payload:
            payload["kind"] = payload.pop("actor_type")
        return self._coerce_actor_payload(payload)

    def response(self, model_id: str, reply: str, trace: list[SceneAssistantToolTrace], raw_response: dict[str, Any]) -> SceneAssistantResponse:
        return SceneAssistantResponse(
            model=model_id,
            reply=reply,
            map_name=self.runtime_map.map_name,
            normalized_map_name=self.runtime_map.normalized_map_name,
            actors=self.actors,
            selected_roads=self.selected_roads,
            selected_actor_id=self.selected_actor_id,
            tool_trace=trace,
            raw_response=raw_response,
        )

    def _actor_summary(
        self,
        actor: ActorDraft,
        *,
        include_timeline: bool = False,
        include_anchor_details: bool = False,
    ) -> dict[str, Any]:
        authored_spawn = self._authored_point(actor, "spawn")
        authored_destination = self._authored_point(actor, "destination")
        live_state = self.live_actors_by_label.get(actor.label.lower())
        payload = {
            "id": actor.id,
            "label": actor.label,
            "kind": actor.kind,
            "role": actor.role,
            "placement_mode": actor.placement_mode,
            "is_static": actor.is_static,
            "blueprint": actor.blueprint,
            "speed_kph": actor.speed_kph,
            "autopilot": actor.autopilot,
            "spawn": actor.spawn.model_dump(),
            "destination": actor.destination.model_dump() if actor.destination is not None else None,
            "authored_spawn_point": authored_spawn,
            "authored_destination_point": authored_destination,
            "lane_change_options": self._lane_change_options(actor.spawn)
            if actor.kind == "vehicle" and actor.placement_mode == "road"
            else None,
            "live_state": live_state.model_dump() if live_state is not None else None,
        }
        if include_timeline:
            payload["timeline"] = [clip.model_dump() for clip in actor.timeline]
        if include_anchor_details:
            payload["spawn_lane"] = self._lane_descriptor(
                actor.spawn.road_id,
                actor.spawn.section_id,
                actor.spawn.lane_id,
            )
            payload["destination_lane"] = (
                self._lane_descriptor(
                    actor.destination.road_id,
                    actor.destination.section_id,
                    actor.destination.lane_id,
                )
                if actor.destination is not None
                else None
            )
        return payload

    def _resolve_actor(self, actor_ref: str) -> ActorDraft:
        if not actor_ref:
            raise RuntimeError("actor_ref is required.")
        lowered = actor_ref.strip().lower()
        for actor in self.actors:
            if actor.id == actor_ref or actor.label.lower() == lowered:
                return actor
        partial = [actor for actor in self.actors if lowered in actor.id.lower() or lowered in actor.label.lower()]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise RuntimeError(f"No actor matched {actor_ref!r}.")
        raise RuntimeError(f"Actor reference {actor_ref!r} is ambiguous.")

    def _segment_for_anchor(
        self,
        anchor: ActorRoadAnchor | None,
    ) -> RuntimeRoadSegment | None:
        if anchor is None or anchor.section_id is None or anchor.lane_id is None:
            return None
        return self.segments_by_key.get((anchor.road_id, anchor.section_id, anchor.lane_id))

    def _lane_descriptor(self, road_id: str, section_id: int | None, lane_id: int | None) -> dict[str, Any] | None:
        if section_id is None or lane_id is None:
            return None
        segment = self.segments_by_key.get((str(road_id), int(section_id), int(lane_id)))
        if segment is None:
            return None
        return {
            "road_id": str(segment.road_id),
            "section_id": segment.section_id,
            "lane_id": segment.lane_id,
            "lane_type": segment.lane_type,
            "left_lane_id": segment.left_lane_id,
            "right_lane_id": segment.right_lane_id,
        }

    def _adjacent_lane_ids(self, segment: RuntimeRoadSegment) -> tuple[int | None, int | None]:
        left_lane_id = segment.left_lane_id
        right_lane_id = segment.right_lane_id
        if left_lane_id is not None or right_lane_id is not None:
            return left_lane_id, right_lane_id

        siblings = [
            candidate
            for candidate in self.segments_by_road.get(str(segment.road_id), [])
            if candidate.section_id == segment.section_id
            and candidate.lane_id != segment.lane_id
            and math.copysign(1, candidate.lane_id) == math.copysign(1, segment.lane_id)
            and str(candidate.lane_type or "").lower() in {"driving", "bidirectional"}
        ]
        if segment.lane_id > 0:
            return (
                next((item.lane_id for item in siblings if item.lane_id == segment.lane_id - 1), None),
                next((item.lane_id for item in siblings if item.lane_id == segment.lane_id + 1), None),
            )
        if segment.lane_id < 0:
            return (
                next((item.lane_id for item in siblings if item.lane_id == segment.lane_id + 1), None),
                next((item.lane_id for item in siblings if item.lane_id == segment.lane_id - 1), None),
            )
        return None, None

    def _lane_change_options(self, anchor: ActorRoadAnchor) -> dict[str, Any]:
        segment = self._segment_for_anchor(anchor)
        if segment is None:
            return {"known": False, "left": False, "right": False}
        left_lane_id, right_lane_id = self._adjacent_lane_ids(segment)
        return {
            "known": True,
            "left": left_lane_id is not None,
            "right": right_lane_id is not None,
            "left_lane_id": left_lane_id,
            "right_lane_id": right_lane_id,
        }

    def _authored_point(self, actor: ActorDraft, target: str) -> dict[str, float] | None:
        if actor.placement_mode in {"path", "point"}:
            point = actor.spawn_point if target == "spawn" else actor.destination_point
            return point.model_dump() if point is not None else None
        anchor = actor.spawn if target == "spawn" else actor.destination
        if anchor is None:
            return None
        segment = self._segment_for_anchor(anchor)
        if segment is None or not segment.centerline:
            return None
        index = min(
            len(segment.centerline) - 1,
            max(0, int(round(float(anchor.s_fraction) * (len(segment.centerline) - 1)))),
        )
        point = segment.centerline[index]
        return {
            "x": float(point["x"]),
            "y": float(point["y"]),
        }

    def _ensure_selected_road(self, road_id: str) -> None:
        if any(road.id == road_id for road in self.selected_roads):
            return
        road_payload = build_selected_roads(self.map_name, [road_id])
        if road_payload:
            self.selected_roads.append(road_payload[0])
            return
        summary = self.road_summaries_by_id.get(road_id)
        self.selected_roads.append(
            SelectedRoad(
                id=road_id,
                name=summary.name if summary is not None else f"Road {road_id}",
                tags=summary.tags if summary is not None else [],
                section_labels=[section.label for section in summary.section_summaries] if summary is not None else [],
            )
        )

    def _default_destination_for_spawn(self, spawn: ActorRoadAnchor) -> ActorRoadAnchor:
        return ActorRoadAnchor(
            road_id=spawn.road_id,
            section_id=spawn.section_id,
            lane_id=spawn.lane_id,
            s_fraction=min(0.95, float(spawn.s_fraction) + 0.18),
        )

    def _default_destination_point(self, point: ActorMapPoint | None) -> ActorMapPoint | None:
        if point is None:
            return None
        return ActorMapPoint(x=float(point.x) + 12.0, y=float(point.y))

    def _road_length(self, road_id: str) -> float | None:
        for road in self.selected_roads:
            if road.id == road_id and road.length > 0:
                return float(road.length)
        return None

    def _lane_type_for_spawn_payload(self, payload: dict[str, Any]) -> str | None:
        if str(payload.get("placement_mode") or "road") != "road":
            return None
        spawn = payload.get("spawn") or {}
        road_id = str(spawn.get("road_id") or "")
        section_id = spawn.get("section_id")
        lane_id = spawn.get("lane_id")
        if not road_id or section_id is None or lane_id is None:
            return None
        segment = self.segments_by_key.get((road_id, int(section_id), int(lane_id)))
        if segment is None:
            road = self.road_summaries_by_id.get(road_id)
            if road is None:
                return None
            section = next(
                (item for item in road.section_summaries if int(item.index) == int(section_id)),
                None,
            )
            if section is None:
                return None
            lane_offset = abs(int(lane_id))
            driving_count = int(section.driving_left if int(lane_id) > 0 else section.driving_right)
            parking_count = int(section.parking_left if int(lane_id) > 0 else section.parking_right)
            if lane_offset <= driving_count:
                return "driving"
            if lane_offset <= driving_count + parking_count:
                return "parking"
            return None
        return str(segment.lane_type or "").strip().lower() or None

    def _next_label_index(self, prefix: str) -> int:
        base = prefix.strip() or "Actor"
        pattern = re.compile(rf"^{re.escape(base)}(?:\s+(\d+))?$", re.IGNORECASE)
        max_index = 0
        for actor in self.actors:
            match = pattern.match(actor.label.strip())
            if not match:
                continue
            suffix = match.group(1)
            max_index = max(max_index, int(suffix) if suffix is not None else 1)
        return max_index + 1

    def _series_s_fractions(self, count: int, *, start: float, end: float) -> list[float]:
        if count <= 0:
            raise RuntimeError("count must be greater than zero.")
        start = max(0.0, min(1.0, float(start)))
        end = max(0.0, min(1.0, float(end)))
        if end < start:
            start, end = end, start
        if count == 1:
            return [round((start + end) / 2.0, 4)]
        step = (end - start) / float(count - 1)
        return [round(start + step * index, 4) for index in range(count)]

    def _coerce_actor_payload(self, actor_payload: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(actor_payload)
        if "spawn" not in payload and "road_id" in payload:
            road_id = str(payload.pop("road_id"))
            section_id = payload.pop("section_id", None)
            lane_id = payload.pop("lane_id", None)
            s_offset = payload.pop("s_offset", payload.pop("s_fraction", 0.5))
            is_relative_s = bool(payload.pop("is_relative_s", True))
            s_fraction = float(s_offset)
            if not is_relative_s:
                road_length = self._road_length(road_id)
                if road_length and road_length > 0:
                    s_fraction = s_fraction / road_length
            payload["spawn"] = {
                "road_id": road_id,
                "section_id": int(section_id) if section_id is not None else None,
                "lane_id": int(lane_id) if lane_id is not None else None,
                "s_fraction": max(0.0, min(1.0, s_fraction)),
            }
        return payload

    def _normalize_actor(self, actor: ActorDraft) -> ActorDraft:
        payload = actor.model_dump()
        if actor.placement_mode == "road":
            payload["spawn_point"] = None
            payload["destination_point"] = None
            payload["destination"] = None
            if actor.is_static:
                payload["speed_kph"] = 0.0
                payload["autopilot"] = False
            else:
                payload["autopilot"] = bool(payload.get("autopilot", actor.kind == "vehicle"))
        elif actor.placement_mode == "path":
            payload["autopilot"] = False
            if payload.get("spawn_point") is None:
                point = self._authored_point(actor, "spawn")
                if point is not None:
                    payload["spawn_point"] = point
            if payload.get("destination") is None:
                payload["destination"] = self._default_destination_for_spawn(actor.spawn).model_dump()
            if payload.get("destination_point") is None and payload.get("spawn_point") is not None:
                payload["destination_point"] = self._default_destination_point(
                    ActorMapPoint.model_validate(payload["spawn_point"])
                ).model_dump()
        elif actor.placement_mode == "point":
            payload["is_static"] = True
            payload["speed_kph"] = 0.0
            payload["autopilot"] = False
            payload["destination"] = None
            payload["destination_point"] = None
            if payload.get("spawn_point") is None:
                point = self._authored_point(actor, "spawn")
                if point is not None:
                    payload["spawn_point"] = point

        normalized = ActorDraft.model_validate(payload)
        normalized = normalized.model_copy(update={"timeline": self._sanitize_timeline(normalized)})
        return normalized

    def _sanitize_timeline(self, actor: ActorDraft) -> list[ActorTimelineClip]:
        if actor.is_static or actor.placement_mode == "point":
            if actor.timeline:
                raise RuntimeError(f"{actor.label} is static and cannot have timeline events.")
            return []
        allowed_actions = {"set_speed", "stop", "hold_position"}
        if actor.kind == "vehicle" and actor.placement_mode == "road":
            allowed_actions.update(
                {
                    "follow_route",
                    "enable_autopilot",
                    "disable_autopilot",
                    "lane_change_left",
                    "lane_change_right",
                    "turn_left_at_next_intersection",
                    "turn_right_at_next_intersection",
                    "chase_actor",
                    "ram_actor",
                }
            )
        sanitized: list[ActorTimelineClip] = []
        for raw_clip in actor.timeline:
            clip = ActorTimelineClip.model_validate(raw_clip.model_dump())
            if clip.action not in allowed_actions:
                raise RuntimeError(f"{clip.action} is not allowed for {actor.label}.")
            if clip.action in {"lane_change_left", "lane_change_right"}:
                options = self._lane_change_options(actor.spawn)
                if not options.get("known"):
                    raise RuntimeError(f"Lane change availability is unknown for {actor.label}.")
                if clip.action == "lane_change_left" and not options.get("left"):
                    raise RuntimeError(f"{actor.label} does not have a drivable lane to the left.")
                if clip.action == "lane_change_right" and not options.get("right"):
                    raise RuntimeError(f"{actor.label} does not have a drivable lane to the right.")
            if clip.action in {"chase_actor", "ram_actor"}:
                if not clip.target_actor_id:
                    raise RuntimeError(f"{clip.action} requires a target_actor_id for {actor.label}.")
                if clip.target_actor_id == actor.id:
                    raise RuntimeError(f"{actor.label} cannot target itself for {clip.action}.")
                target = next((candidate for candidate in self.actors if candidate.id == clip.target_actor_id), None)
                if target is None:
                    raise RuntimeError(f"{clip.action} target {clip.target_actor_id} was not found for {actor.label}.")
                if target.kind != "vehicle":
                    raise RuntimeError(f"{clip.action} target must be a vehicle for {actor.label}.")
            sanitized.append(clip)
        return sorted(sanitized, key=lambda item: (float(item.start_time), item.id))

    def _add_actor(self, actor_payload: dict[str, Any], *, select: bool) -> ActorDraft:
        payload = self._coerce_actor_payload(actor_payload)
        payload.setdefault("id", str(payload.get("id") or uuid.uuid4()))
        payload.setdefault("label", f"Actor {len(self.actors) + 1}")
        payload.setdefault("kind", "vehicle")
        payload.setdefault("role", "traffic" if payload["kind"] == "vehicle" else "pedestrian")
        payload.setdefault("placement_mode", "road")
        payload.setdefault(
            "blueprint",
            DEFAULT_VEHICLE_BLUEPRINT if payload["kind"] == "vehicle" else DEFAULT_WALKER_BLUEPRINT,
        )
        payload.setdefault("spawn", {"road_id": self.selected_roads[0].id if self.selected_roads else "0", "s_fraction": 0.5})
        payload.setdefault("speed_kph", 25.0 if payload["kind"] == "vehicle" else 5.0)
        payload.setdefault("autopilot", payload["kind"] == "vehicle" and payload.get("placement_mode") == "road")
        payload.setdefault("timeline", [])
        lane_type = self._lane_type_for_spawn_payload(payload)
        if (
            payload.get("kind") == "vehicle"
            and payload.get("placement_mode") == "road"
            and lane_type == "parking"
            and str(payload.get("role") or "traffic") != "ego"
        ):
            payload["is_static"] = True
            payload["speed_kph"] = 0.0
            payload["autopilot"] = False
        actor = self._normalize_actor(ActorDraft.model_validate(payload))
        self._ensure_selected_road(actor.spawn.road_id)
        if actor.destination is not None:
            self._ensure_selected_road(actor.destination.road_id)
        self.actors.append(actor)
        if select:
            self.selected_actor_id = actor.id
        return actor

    def _add_actor_row(self, operation: dict[str, Any], *, select: bool) -> list[ActorDraft]:
        template = deepcopy(operation.get("actor") or {})
        count = int(operation.get("count") or 0)
        if count <= 0:
            raise RuntimeError("add_actor_row requires count > 0.")

        template = self._coerce_actor_payload(template)
        spawn_template = dict(template.get("spawn") or {})
        road_id = str(operation.get("road_id") or spawn_template.get("road_id") or "")
        if not road_id:
            raise RuntimeError("add_actor_row requires a road_id.")
        section_id_raw = operation.get("section_id", spawn_template.get("section_id", 0))
        if section_id_raw is None:
            raise RuntimeError("add_actor_row requires a section_id.")
        section_id = int(section_id_raw)

        lane_ids_raw = operation.get("lane_ids")
        if lane_ids_raw is None:
            single_lane = operation.get("lane_id", spawn_template.get("lane_id"))
            lane_ids_raw = [single_lane] if single_lane is not None else []
        lane_ids = [int(lane_id) for lane_id in lane_ids_raw if lane_id is not None]
        if not lane_ids:
            raise RuntimeError("add_actor_row requires at least one lane id.")

        is_relative_s = bool(operation.get("is_relative_s", True))
        s_start = float(operation.get("s_start", 0.1))
        s_end = float(operation.get("s_end", 0.9))
        if not is_relative_s:
            road_length = self._road_length(road_id)
            if not road_length or road_length <= 0:
                raise RuntimeError(f"Road {road_id} length is unavailable for absolute spacing.")
            s_start = s_start / road_length
            s_end = s_end / road_length
        s_fractions = self._series_s_fractions(count, start=s_start, end=s_end)

        label_prefix = str(
            operation.get("label_prefix")
            or template.pop("label_prefix", None)
            or template.get("label")
            or ("Traffic Car" if template.get("kind", "vehicle") == "vehicle" else "Actor")
        ).strip()
        next_index = self._next_label_index(label_prefix)

        created: list[ActorDraft] = []
        for index, s_fraction in enumerate(s_fractions):
            actor_payload = deepcopy(template)
            actor_payload["id"] = str(uuid.uuid4())
            actor_payload["label"] = f"{label_prefix} {next_index + index}"
            actor_payload["spawn"] = {
                **spawn_template,
                "road_id": road_id,
                "section_id": section_id,
                "lane_id": lane_ids[index % len(lane_ids)],
                "s_fraction": s_fraction,
            }
            created.append(self._add_actor(actor_payload, select=False))

        if select and created:
            self.selected_actor_id = created[-1].id
        return created

    def _update_actor(self, actor_id: str, changes: dict[str, Any]) -> ActorDraft:
        actor = self._resolve_actor(actor_id)
        payload = actor.model_dump()
        for key, value in changes.items():
            if key in {"spawn", "destination", "spawn_point", "destination_point"} and value is not None:
                payload[key] = {**(payload.get(key) or {}), **dict(value)}
            else:
                payload[key] = value
        updated = self._normalize_actor(ActorDraft.model_validate(payload))
        if updated.spawn.road_id:
            self._ensure_selected_road(updated.spawn.road_id)
        if updated.destination is not None:
            self._ensure_selected_road(updated.destination.road_id)
        self.actors = [updated if item.id == actor.id else item for item in self.actors]
        return updated

    def _remove_actor(self, actor_id: str) -> ActorDraft:
        actor = self._resolve_actor(actor_id)
        self.actors = [item for item in self.actors if item.id != actor.id]
        if self.selected_actor_id == actor.id:
            self.selected_actor_id = self.actors[0].id if self.actors else None
        return actor

    def _replace_timeline(self, actor_id: str, timeline_payload: list[dict[str, Any]]) -> ActorDraft:
        return self._update_actor(actor_id, {"timeline": timeline_payload})

    def _add_timeline_clip(self, actor_id: str, clip_payload: dict[str, Any]) -> ActorDraft:
        actor = self._resolve_actor(actor_id)
        clip_data = deepcopy(clip_payload)
        clip_data.setdefault("id", str(clip_data.get("id") or uuid.uuid4()))
        clip = ActorTimelineClip.model_validate(clip_data)
        return self._update_actor(
            actor.id,
            {"timeline": [item.model_dump() for item in actor.timeline] + [clip.model_dump()]},
        )

    def _remove_timeline_clip(self, actor_id: str, clip_id: str) -> ActorDraft:
        actor = self._resolve_actor(actor_id)
        return self._update_actor(
            actor.id,
            {"timeline": [item.model_dump() for item in actor.timeline if item.id != clip_id]},
        )

    def _add_selected_roads(self, road_ids: list[str]) -> list[str]:
        added: list[str] = []
        for road_id in road_ids:
            if any(road.id == road_id for road in self.selected_roads):
                continue
            self._ensure_selected_road(road_id)
            added.append(road_id)
        return added

    def _remove_selected_roads(self, road_ids: list[str]) -> list[str]:
        road_id_set = set(road_ids)
        self.selected_roads = [road for road in self.selected_roads if road.id not in road_id_set]
        self.actors = [
            actor
            for actor in self.actors
            if actor.spawn.road_id not in road_id_set
            and (actor.destination is None or actor.destination.road_id not in road_id_set)
        ]
        if self.selected_actor_id and not any(actor.id == self.selected_actor_id for actor in self.actors):
            self.selected_actor_id = self.actors[0].id if self.actors else None
        return road_ids

    def _set_selected_roads(self, road_ids: list[str]) -> None:
        self.selected_roads = []
        self._add_selected_roads(road_ids)
        road_id_set = set(road_ids)
        self.actors = [
            actor
            for actor in self.actors
            if actor.spawn.road_id in road_id_set
            and (actor.destination is None or actor.destination.road_id in road_id_set)
        ]
        if self.selected_actor_id and not any(actor.id == self.selected_actor_id for actor in self.actors):
            self.selected_actor_id = self.actors[0].id if self.actors else None


class BedrockSceneAssistant:
    def __init__(self, carla_metadata: Any | None = None) -> None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = ""
        self.carla_metadata = carla_metadata

    def _system_prompt(self) -> str:
        return dedent(
            """
            You are a scene editing copilot for a CARLA scenario authoring UI.
            The user wants concrete scene changes, not high-level advice.

            Rules:
            - Inspect before you modify when the request depends on current actor positions, lanes, or road structure.
            - Use find_roads and search_maps_by_road when the user asks for a road type or map that matches certain characteristics.
            - Use switch_map only when changing the active CARLA world is necessary to satisfy the request.
            - Only use road ids, section ids, and lane ids returned by the tools.
            - Apply edits through the mutation tool; do not describe a change as complete unless the mutation tool succeeded.
            - Prefer small, safe edits when the request is underspecified.
            - When a scenario is running, use live actor state when discussing current position. Otherwise use authored positions.
            - Default vehicle blueprint: vehicle.tesla.model3
            - Default walker blueprint: walker.pedestrian.0001
            - When adding many similar actors on one road, prefer one add_actor_row edit instead of many add_actor edits.
            - When the user asks for an ego vehicle, set role=ego in the initial add_actor operation. Do not rely on a later update to convert a traffic actor into ego.
            - Parked vehicles must be authored as static in the initial operation: is_static=true, speed_kph=0, autopilot=false, no lane changes.
            - For vehicles placed in parking lanes, prefer static parked cars unless the user explicitly asks for movement.
            - Do not stack multiple actors at the same s_fraction unless the user explicitly asks for overlap.
            - When using add_actor_row, set s_start and s_end so repeated actors are distributed along the road.
            - If the user asks for cars on both sides of a road, inspect the road first and use the parking lane ids returned by the tools.
            - For road vehicles, timeline clips can use Route follow, Traffic Manager, chase_actor, ram_actor, and next-intersection turn instructions.
            - Route-follow vehicles may also set lane_facing=against_lane for wrong-way spawn orientation and route_direction=reverse to back along the route.
            - Use ram_actor when the user wants an intentional collision; autopilot alone will not do that reliably.
            - Use turn_left_at_next_intersection or turn_right_at_next_intersection when the user wants a road vehicle to take a junction turn under Traffic Manager.
            - Keep the final reply short and state exactly what changed or what blocked the change.
            """
        ).strip()

    def _scene_capsule(self, request: SceneAssistantRequest) -> str:
        selected_actor_line = request.selected_actor_id or "none"
        return dedent(
            f"""
            Scene capsule:
            - map: {request.map_name}
            - selected roads: {len(request.selected_roads)}
            - actors: {len(request.actors)}
            - live actors: {len(request.live_actors)}
            - selected actor id: {selected_actor_line}

            Use tools to inspect the scene and apply edits.
            """
        ).strip()

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_scene_overview",
                "description": "Return a compact summary of selected roads, current actors, and authored or live positions.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_actor",
                "description": "Return detailed information about a specific actor by id or label.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "actor_ref": {"type": "string"},
                    },
                    "required": ["actor_ref"],
                },
            },
            {
                "name": "get_road",
                "description": "Return road summary and runtime lane segment details for a road id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "road_id": {"type": "string"},
                    },
                    "required": ["road_id"],
                },
            },
            {
                "name": "get_adjacent_lanes",
                "description": "Return left and right drivable lane information for a specific road/section/lane.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "road_id": {"type": "string"},
                        "section_id": {"type": "integer"},
                        "lane_id": {"type": "integer"},
                    },
                    "required": ["road_id", "section_id", "lane_id"],
                },
            },
            {
                "name": "find_nearest_lane",
                "description": "Project an XY point in frontend map coordinates to the nearest runtime lane.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "find_roads",
                "description": "Search roads in the current map by tags, lane types, parking, and lane-count characteristics.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "lane_types": {"type": "array", "items": {"type": "string"}},
                        "is_intersection": {"type": "boolean"},
                        "has_parking": {"type": "boolean"},
                        "driving_left": {"type": "integer"},
                        "driving_right": {"type": "integer"},
                        "total_driving": {"type": "integer"},
                        "parking_left_min": {"type": "integer"},
                        "parking_right_min": {"type": "integer"},
                        "require_parking_on_both_sides": {"type": "boolean"},
                        "limit": {"type": "integer"},
                    },
                },
            },
            {
                "name": "search_maps_by_road",
                "description": "Search across all supported dataset maps for roads with specific tags, parking, and lane-count characteristics.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "lane_types": {"type": "array", "items": {"type": "string"}},
                        "is_intersection": {"type": "boolean"},
                        "has_parking": {"type": "boolean"},
                        "driving_left": {"type": "integer"},
                        "driving_right": {"type": "integer"},
                        "total_driving": {"type": "integer"},
                        "parking_left_min": {"type": "integer"},
                        "parking_right_min": {"type": "integer"},
                        "require_parking_on_both_sides": {"type": "boolean"},
                        "map_limit": {"type": "integer"},
                        "roads_per_map_limit": {"type": "integer"},
                    },
                },
            },
            {
                "name": "switch_map",
                "description": "Switch the active CARLA world to another supported map and reset the authored scene to that map.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "map_name": {"type": "string"},
                    },
                    "required": ["map_name"],
                },
            },
            {
                "name": "apply_scene_edits",
                "description": (
                    "Apply validated scene edits. Supported operation types: "
                    "add_actor, add_actor_row, update_actor, remove_actor, replace_timeline, add_timeline_clip, "
                    "remove_timeline_clip, select_actor, add_selected_roads, remove_selected_roads, set_selected_roads. "
                    "Actor payloads may include role, is_static, speed_kph, autopilot, placement_mode, blueprint, "
                    "spawn {road_id, section_id, lane_id, s_fraction}, route_direction, lane_facing, destination, spawn_point, destination_point, "
                    "and timeline. Timeline clips support follow_route, set_speed, stop, hold_position, enable_autopilot, disable_autopilot, "
                    "lane_change_left, lane_change_right, turn_left_at_next_intersection, turn_right_at_next_intersection, "
                    "chase_actor, and ram_actor. Chase and ram clips require target_actor_id and may include target_speed_kph. "
                    "Set role=ego in add_actor when the user asks for an ego vehicle. For parked cars, "
                    "set is_static=true, speed_kph=0, autopilot=false in the initial add_actor or add_actor_row actor template."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "actor": {"type": "object"},
                                    "count": {"type": "integer"},
                                    "road_id": {"type": "string"},
                                    "section_id": {"type": "integer"},
                                    "lane_id": {"type": "integer"},
                                    "lane_ids": {"type": "array", "items": {"type": "integer"}},
                                    "s_start": {"type": "number"},
                                    "s_end": {"type": "number"},
                                    "is_relative_s": {"type": "boolean"},
                                    "label_prefix": {"type": "string"},
                                    "actor_id": {"type": "string"},
                                    "changes": {"type": "object"},
                                    "timeline": {"type": "array", "items": {"type": "object"}},
                                    "clip": {"type": "object"},
                                    "clip_id": {"type": "string"},
                                    "road_ids": {"type": "array", "items": {"type": "string"}},
                                    "select": {"type": "boolean"},
                                },
                                "required": ["type"],
                            },
                        }
                    },
                    "required": ["operations"],
                },
            },
        ]

    def _invoke(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())

    def _tool_result_content(self, result: dict[str, Any]) -> list[dict[str, str]]:
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=True)}]

    def _text_from_content(self, content: list[dict[str, Any]]) -> str:
        parts = [str(part.get("text", "")).strip() for part in content if part.get("type") == "text"]
        return "\n".join(part for part in parts if part).strip()

    def _run_tool(self, state: SceneEditorState, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if name == "get_scene_overview":
            return state.scene_overview()
        if name == "get_actor":
            return state.actor_details(str(tool_input.get("actor_ref") or ""))
        if name == "get_road":
            return state.road_details(str(tool_input.get("road_id") or ""))
        if name == "get_adjacent_lanes":
            return state.adjacent_lanes(
                str(tool_input.get("road_id") or ""),
                int(tool_input.get("section_id")),
                int(tool_input.get("lane_id")),
            )
        if name == "find_nearest_lane":
            return state.nearest_lane(float(tool_input.get("x")), float(tool_input.get("y")))
        if name == "find_roads":
            return state.find_roads(tool_input)
        if name == "search_maps_by_road":
            return state.search_maps_by_road(tool_input)
        if name == "switch_map":
            if self.carla_metadata is None:
                raise RuntimeError("Map switching is unavailable because no CARLA metadata service is attached.")
            target_map = str(tool_input.get("map_name") or "").strip()
            if not target_map:
                raise RuntimeError("map_name is required.")
            status = self.carla_metadata.load_map(target_map)
            runtime_map = self.carla_metadata.get_runtime_map()
            state.replace_runtime_context(runtime_map.map_name, runtime_map)
            return {
                "status": status.model_dump() if hasattr(status, "model_dump") else status,
                "runtime_map": {
                    "map_name": runtime_map.map_name,
                    "normalized_map_name": runtime_map.normalized_map_name,
                    "road_segment_count": len(runtime_map.road_segments),
                    "road_summary_count": len(runtime_map.road_summaries),
                },
                "scene_reset": True,
            }
        if name == "apply_scene_edits":
            return state.apply_operations([dict(item) for item in tool_input.get("operations") or []])
        raise RuntimeError(f"Unsupported assistant tool {name}.")

    def _langchain_tools(self, state: SceneEditorState, trace: list[SceneAssistantToolTrace]) -> dict[str, Any]:
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain tool support is unavailable.")

        def make_tool(name: str, description: str, args_schema: type[BaseModel]):
            def run_tool(**kwargs: Any) -> str:
                result = self._run_tool(state, name, kwargs)
                trace.append(SceneAssistantToolTrace(name=name, input=dict(kwargs), result=result))
                return json.dumps(result, ensure_ascii=True)

            run_tool.__name__ = name
            return StructuredTool.from_function(
                func=run_tool,
                name=name,
                description=description,
                args_schema=args_schema,
            )

        tools = [
            make_tool(
                "get_scene_overview",
                "Return a compact summary of selected roads, current actors, and authored or live positions.",
                SceneAssistantGetSceneOverviewInput,
            ),
            make_tool(
                "get_actor",
                "Return detailed information about a specific actor by id or label.",
                SceneAssistantGetActorInput,
            ),
            make_tool(
                "get_road",
                "Return road summary and runtime lane segment details for a road id.",
                SceneAssistantGetRoadInput,
            ),
            make_tool(
                "get_adjacent_lanes",
                "Return left and right drivable lane information for a specific road, section, and lane.",
                SceneAssistantGetAdjacentLanesInput,
            ),
            make_tool(
                "find_nearest_lane",
                "Project an XY point in frontend map coordinates to the nearest runtime lane.",
                SceneAssistantFindNearestLaneInput,
            ),
            make_tool(
                "find_roads",
                "Search roads in the current map by tags, parking, and lane-count characteristics.",
                SceneAssistantFindRoadsInput,
            ),
            make_tool(
                "search_maps_by_road",
                "Search across all supported dataset maps for roads with specific tags, parking, and lane-count characteristics.",
                SceneAssistantSearchMapsByRoadInput,
            ),
            make_tool(
                "switch_map",
                "Switch the active CARLA world to another supported map and reset the authored scene to that map.",
                SceneAssistantSwitchMapInput,
            ),
            make_tool(
                "apply_scene_edits",
                (
                    "Apply validated scene edits. Supported operation types: add_actor, add_actor_row, "
                    "update_actor, remove_actor, replace_timeline, add_timeline_clip, remove_timeline_clip, "
                    "select_actor, add_selected_roads, remove_selected_roads, set_selected_roads."
                ),
                SceneAssistantApplyEditsInput,
            ),
        ]
        return {tool.name: tool for tool in tools}

    def _langchain_message_history(self, request: SceneAssistantRequest) -> list[Any]:
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain message support is unavailable.")
        messages: list[Any] = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=self._scene_capsule(request)),
        ]
        for message in request.messages:
            if message.role == "assistant":
                messages.append(AIMessage(content=message.content))
            else:
                messages.append(HumanMessage(content=message.content))
        return messages

    def _langchain_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part.strip())
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")).strip())
            return "\n".join(part for part in parts if part).strip()
        return str(content or "").strip()

    def _chat_langchain(self, model_id: str, request: SceneAssistantRequest) -> SceneAssistantResponse:
        trace: list[SceneAssistantToolTrace] = []
        state = SceneEditorState(request)
        messages = self._langchain_message_history(request)
        tool_map = self._langchain_tools(state, trace)
        model = create_chat_model(model_id, temperature=0.2, max_tokens=1400).bind_tools(list(tool_map.values()))
        raw_response: dict[str, Any] = {}

        for _ in range(8):
            ai_message = model.invoke(
                messages,
                config=langsmith_run_config(
                    "carla_scene_assistant_turn",
                    tags=["carla", "scene-assistant", model_id],
                    metadata={
                        "map_name": request.map_name,
                        "selected_road_count": len(request.selected_roads),
                        "actor_count": len(request.actors),
                        "live_actor_count": len(request.live_actors),
                    },
                ),
            )
            raw_response = serialize_ai_message(ai_message)
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                reply = self._langchain_text(getattr(ai_message, "content", "")) or "I inspected the scene but do not have a textual reply."
                self.model_id = model_id
                return state.response(model_id, reply, trace, raw_response)

            for tool_call in tool_calls:
                name = str(tool_call.get("name") or "")
                if name not in tool_map:
                    raise RuntimeError(f"Unsupported assistant tool {name}.")
                tool_output = tool_map[name].invoke(tool_call.get("args") or {})
                messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=str(tool_call.get("id") or ""),
                        name=name,
                    )
                )
        raise RuntimeError("Assistant exceeded the maximum tool-use rounds.")

    @traceable(run_type="chain", name="carla_scene_assistant")
    def chat(self, request: SceneAssistantRequest) -> SceneAssistantResponse:
        if LANGCHAIN_AVAILABLE:
            errors: list[str] = []
            for model_id in [candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate]:
                try:
                    return self._chat_langchain(model_id, request)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
            raise RuntimeError("Scene assistant failed. " + " | ".join(errors))

        trace: list[SceneAssistantToolTrace] = []
        base_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self._scene_capsule(request)}],
            }
        ]
        for message in request.messages:
            base_messages.append(
                {
                    "role": message.role,
                    "content": [{"type": "text", "text": message.content}],
                }
            )

        errors: list[str] = []
        for model_id in [candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate]:
            state = SceneEditorState(request)
            messages = deepcopy(base_messages)
            raw_response: dict[str, Any] = {}
            try:
                for _ in range(8):
                    payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1400,
                        "temperature": 0.2,
                        "system": self._system_prompt(),
                        "tools": self._tool_definitions(),
                        "messages": messages,
                    }
                    raw_response = self._invoke(model_id, payload)
                    content = raw_response.get("content") or []
                    messages.append({"role": "assistant", "content": content})

                    tool_uses = [part for part in content if part.get("type") == "tool_use"]
                    if not tool_uses:
                        reply = self._text_from_content(content) or "I inspected the scene but do not have a textual reply."
                        self.model_id = model_id
                        return state.response(model_id, reply, trace, raw_response)

                    tool_results: list[dict[str, Any]] = []
                    for tool_use in tool_uses:
                        name = str(tool_use.get("name") or "")
                        tool_input = dict(tool_use.get("input") or {})
                        result = self._run_tool(state, name, tool_input)
                        trace.append(SceneAssistantToolTrace(name=name, input=tool_input, result=result))
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use["id"],
                                "content": self._tool_result_content(result),
                            }
                        )
                    messages.append({"role": "user", "content": tool_results})
                raise RuntimeError("Assistant exceeded the maximum tool-use rounds.")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Scene assistant failed. " + " | ".join(errors))
