from __future__ import annotations

import json
import math
import os
import re
import uuid
from copy import deepcopy
from textwrap import dedent
from typing import Any

import anthropic
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
from ..road_corridors import build_corridor_lookup, resolve_corridor_distance, parse_lane_spec
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



class InspectStreetInput(BaseModel):
    ref: str = Field(default="", description="A road_id, actor_id, or corridor_id to look up.")


class FindStreetsInput(BaseModel):
    query: str = Field(default="", description="Free text search (road name, tags)")
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
    limit: int = 5


class InspectActorInput(BaseModel):
    ref: str = Field(default="", description="Actor id or label.")


class EditSceneInput(BaseModel):
    operations: list[dict[str, Any]] = Field(default_factory=list)


class ManageScenarioInput(BaseModel):
    action: str = Field(description="One of: list, read, create, duplicate, rename, delete, switch")
    scenario_id: str | None = None
    name: str | None = None
    source_scenario_id: str | None = None


class RunSimulationInput(BaseModel):
    weather: dict[str, Any] | None = Field(default=None, description="Weather preset name or params dict")
    traffic_lights: list[dict[str, Any]] | None = Field(default=None, description="Traffic light overrides")
    duration_seconds: int | None = None
    record: bool = True


class SceneAssistantSearchRoadsInput(BaseModel):
    scope: str = Field(default="current_map", description="Search scope: 'current_map' or 'all_maps'")
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
    limit: int = 12


class SceneAssistantSwitchMapInput(BaseModel):
    map_name: str


class SceneAssistantGetCorridorsInput(BaseModel):
    pass


class SceneAssistantGetStreetContextInput(BaseModel):
    ref: str = Field(default="", description="A road_id or actor_id to look up the street for.")


class SceneAssistantGetMapContextInput(BaseModel):
    carla_map_name: str = Field(default="", description="CARLA map name. Leave empty for the current map.")


class SceneAssistantSetWeatherInput(BaseModel):
    cloudiness: float | None = Field(default=None, ge=0.0, le=100.0, description="Cloud coverage 0-100")
    precipitation: float | None = Field(default=None, ge=0.0, le=100.0, description="Rain intensity 0-100")
    precipitation_deposits: float | None = Field(default=None, ge=0.0, le=100.0, description="Puddles on ground 0-100")
    wind_intensity: float | None = Field(default=None, ge=0.0, le=100.0, description="Wind strength 0-100")
    fog_density: float | None = Field(default=None, ge=0.0, le=100.0, description="Fog density 0-100")
    fog_distance: float | None = Field(default=None, ge=0.0, le=500.0, description="Fog start distance in meters")
    sun_altitude_angle: float | None = Field(default=None, ge=-90.0, le=90.0, description="Sun elevation: -90 midnight, 0 horizon, 90 noon")
    sun_azimuth_angle: float | None = Field(default=None, ge=0.0, le=360.0, description="Sun compass bearing 0-360")


class SceneAssistantGetStreetFurnitureInput(BaseModel):
    pass


class SceneAssistantSetTrafficLightStateInput(BaseModel):
    traffic_light_id: int = Field(description="CARLA actor id of the traffic light")
    state: str = Field(description="Desired state: red, yellow, or green")
    duration_seconds: float | None = Field(default=None, ge=0.0, le=300.0, description="How long to hold this state")


class SceneAssistantRoadPositionToWorldInput(BaseModel):
    road_id: int = Field(description="OpenDRIVE road id")
    s_fraction: float = Field(ge=0.0, le=1.0, description="Position along road 0.0-1.0")
    lane_id: int = Field(description="OpenDRIVE lane id")


class SceneAssistantListDatasetScenariosInput(BaseModel):
    pass


class SceneAssistantReadScenarioInput(BaseModel):
    scenario_id: str = Field(description="Scenario ID to read")


class SceneAssistantCreateScenarioInput(BaseModel):
    display_name: str = Field(description="Name for the new scenario")
    map_name: str = Field(default="", description="CARLA map name (defaults to current map)")


class SceneAssistantDuplicateScenarioInput(BaseModel):
    source_scenario_id: str = Field(description="ID of the scenario to copy")
    new_name: str = Field(description="Display name for the copy")


class SceneAssistantRenameScenarioInput(BaseModel):
    scenario_id: str
    new_name: str


class SceneAssistantDeleteScenarioInput(BaseModel):
    scenario_id: str


class SceneAssistantSwitchActiveScenarioInput(BaseModel):
    scenario_id: str = Field(description="Scenario ID to switch to")
    reason: str = Field(default="", description="Why switching")


class SceneAssistantRunSimulationInput(BaseModel):
    duration_seconds: float = Field(default=10.0, ge=1.0, le=120.0, description="Duration in seconds")



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
        self.corridors = []
        self.corridors_by_id = {}
        self.road_to_corridor = {}
        self.callback_url = request.callback_url
        self.callback_auth = request.callback_auth
        self.dataset_id = request.dataset_id
        self.switch_to_scenario_id: str | None = None
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


    def set_corridors(self, corridors_data: list) -> None:
        from ..road_corridors import RoadCorridor
        self.corridors = corridors_data
        self.corridors_by_id = {}
        self.road_to_corridor = {}
        for cd in corridors_data:
            cid = cd.id if hasattr(cd, "id") else cd.get("id", "")
            rids = cd.road_ids if hasattr(cd, "road_ids") else cd.get("road_ids", [])
            self.corridors_by_id[cid] = cd
            for rid in rids:
                self.road_to_corridor[str(rid)] = cid

    def get_corridors_summary(self) -> dict:
        results = []
        for cd in self.corridors:
            if hasattr(cd, "id"):
                results.append({"id": cd.id, "description": cd.description, "total_length_m": round(cd.total_length, 1), "road_count": len(cd.road_ids), "road_ids": cd.road_ids, "has_parking": cd.has_parking, "lane_config": cd.lane_config})
            else:
                results.append({"id": cd["id"], "description": cd.get("description", ""), "total_length_m": cd.get("total_length", 0), "road_count": cd.get("road_count", 1), "road_ids": cd.get("road_ids", []), "has_parking": cd.get("has_parking", False), "lane_config": cd.get("lane_config", "")})
        multi = sum(1 for r in results if r["road_count"] > 1)
        return {"total_corridors": len(results), "multi_road_corridors": multi, "corridors": results}

    def _get_corridor_attr(self, cd, attr, default=None):
        return getattr(cd, attr, None) if hasattr(cd, attr) else cd.get(attr, default)

    def get_street_context(self, ref: str) -> dict:
        road_id = None
        for actor in self.actors:
            if actor.id == ref or actor.label.lower() == ref.lower():
                if actor.spawn and actor.spawn.road_id:
                    road_id = str(actor.spawn.road_id)
                break
        if road_id is None:
            road_id = str(ref)
        corridor_id = self.road_to_corridor.get(road_id)
        if not corridor_id:
            return {"street": {"id": "road-" + road_id, "description": "Single road segment", "total_length_m": 0, "segments": [{"road_id": road_id, "length_m": 0, "start_m": 0, "lanes": self._get_lanes_for_road(road_id)}]}, "connected_streets": [], "actors_on_street": self._actors_on_roads([road_id]), "placement_hint": "Single road segment."}
        cd = self.corridors_by_id.get(corridor_id)
        if not cd:
            return {"error": "Corridor " + corridor_id + " not found"}
        rids = self._get_corridor_attr(cd, "road_ids", [])
        total = self._get_corridor_attr(cd, "total_length", 0)
        seg_l = self._get_corridor_attr(cd, "segment_lengths", [])
        seg_o = self._get_corridor_attr(cd, "segment_offsets", [])
        desc = self._get_corridor_attr(cd, "description", "")
        cid = self._get_corridor_attr(cd, "id", "")
        segs = []
        for i, rid in enumerate(rids):
            segs.append({"road_id": rid, "length_m": round(seg_l[i], 1) if i < len(seg_l) else 0, "start_m": round(seg_o[i], 1) if i < len(seg_o) else 0, "lanes": self._get_lanes_for_road(rid)})
        actors = self._actors_on_roads(rids)
        positions = sorted([a["distance_m"] for a in actors])
        if positions:
            gaps = []
            prev = 0
            for p in positions:
                if p - prev > 5:
                    gaps.append((prev, p, p - prev))
                prev = p
            if total - prev > 5:
                gaps.append((prev, total, total - prev))
            biggest = max(gaps, key=lambda g: g[2]) if gaps else (0, total, total)
            hint = f"{total:.0f}m street, {len(actors)} actors. Largest gap: {biggest[2]:.0f}m ({biggest[0]:.0f}m to {biggest[1]:.0f}m)."
        else:
            hint = f"{total:.0f}m street, no actors placed yet."
        return {"street": {"id": cid, "description": desc, "total_length_m": round(total, 1), "segments": segs}, "connected_streets": [], "actors_on_street": actors, "placement_hint": hint}

    def _get_lanes_for_road(self, road_id: str) -> dict:
        segments = self.segments_by_road.get(str(road_id), [])
        left, right, seen = [], [], set()
        for seg in segments:
            lt = getattr(seg, "lane_type", "driving").lower()
            lid = getattr(seg, "lane_id", 0)
            if (lid, lt) in seen:
                continue
            seen.add((lid, lt))
            if lid < 0:
                right.append(lt)
            elif lid > 0:
                left.append(lt)
        return {"right": sorted(set(right)), "left": sorted(set(left))}

    def _actors_on_roads(self, road_ids: list) -> list:
        road_set = {str(r) for r in road_ids}
        offset_map = {}
        cid = self.road_to_corridor.get(str(road_ids[0])) if road_ids else None
        if cid:
            cd = self.corridors_by_id.get(cid)
            if cd:
                crids = self._get_corridor_attr(cd, "road_ids", [])
                coffsets = self._get_corridor_attr(cd, "segment_offsets", [])
                clengths = self._get_corridor_attr(cd, "segment_lengths", [])
                for i, rid in enumerate(crids):
                    offset_map[rid] = (coffsets[i] if i < len(coffsets) else 0, clengths[i] if i < len(clengths) else 0)
        results = []
        for actor in self.actors:
            if not actor.spawn or not actor.spawn.road_id:
                continue
            rid = str(actor.spawn.road_id)
            if rid not in road_set:
                continue
            off, length = offset_map.get(rid, (0, 0))
            dist = off + (actor.spawn.s_fraction or 0) * length
            lid = actor.spawn.lane_id
            side = "right" if (lid is not None and lid < 0) else "left" if (lid is not None and lid > 0) else "center"
            results.append({"id": actor.id, "label": actor.label, "distance_m": round(dist, 1), "lane": side})
        return results

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
            "actors": [
                {
                    "id": actor.id,
                    "label": actor.label,
                    "kind": actor.kind,
                    "role": actor.role,
                    "road_id": actor.spawn.road_id if actor.spawn else None,
                    "s_fraction": actor.spawn.s_fraction if actor.spawn else None,
                    "is_static": actor.is_static,
                    "speed_kph": actor.speed_kph,
                }
                for actor in self.actors
            ],
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
            return {
                "road_id": road_key,
                "summary": None,
                "segments": [],
                "note": "This road has no CARLA runtime lanes. Actors placed here will not spawn.",
            }
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
            return {
                "road_id": str(road_id),
                "section_id": section_id,
                "lane_id": lane_id,
                "left_lane_id": None,
                "right_lane_id": None,
                "left_lane": None,
                "right_lane": None,
                "note": "No runtime lane data available. Actor placement may still work.",
            }
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
            raise RuntimeError("Runtime map has no lane geometry. Use find_streets instead -- it returns suggested_spawn with road_id, section_id, lane_id, and s_fraction ready for edit_scene.")
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
        runtime_road_ids = {str(k) for k in self.segments_by_road.keys()}
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
            runtime_road_ids=runtime_road_ids,
        )
        return {
            "map_name": self.map_name,
            "match_count": len(matches),
            "roads": matches,
        }

    def search_maps_by_road(self, criteria: dict[str, Any]) -> dict[str, Any]:
        return {"match_count": 0, "maps": [], "note": "Cross-map search is not available."}

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
            switch_to_scenario_id=self.switch_to_scenario_id,
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
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = "claude-sonnet-4-6"
        self.carla_metadata = carla_metadata

    def _editor_request(self, state: SceneEditorState, method: str, path: str, body: dict | None = None) -> dict:
        """Make HTTP request back to the editor API."""
        import requests as http_requests
        if not state.callback_url:
            raise RuntimeError("callback_url is not set — dataset tools are unavailable without editor callback.")
        url = f"{state.callback_url}{path}"
        headers = {
            "Cookie": f"{"__Secure-" if state.callback_url and state.callback_url.startswith("https") else ""}better-auth.session_token={state.callback_auth}",
            "Content-Type": "application/json",
        }
        if method == "GET":
            r = http_requests.get(url, headers=headers, timeout=15)
        elif method == "POST":
            r = http_requests.post(url, headers=headers, json=body or {}, timeout=15)
        elif method == "PUT":
            r = http_requests.put(url, headers=headers, json=body or {}, timeout=15)
        elif method == "PATCH":
            r = http_requests.patch(url, headers=headers, json=body or {}, timeout=15)
        elif method == "DELETE":
            r = http_requests.delete(url, headers=headers, timeout=15)
        else:
            raise RuntimeError(f"Unsupported HTTP method: {method}")
        r.raise_for_status()
        return r.json() if r.content else {}

    def _system_prompt(self) -> str:
        return dedent(
            """
            You are a scene editing copilot for a CARLA scenario authoring UI.
            The user wants concrete scene changes, not high-level advice.

            Workflow for placing actors:
            1. Check the scene capsule for available streets (corridors). Each corridor chains connected road segments into one logical street.
            2. Use inspect_street(ref) to see lane details, existing actors, and placement hints for any street.
            3. Call edit_scene with ALL changes in a single call. Use abstract placement: {street: "corridor-id", distance_m: N, lane: "right driving"}.
            4. Reply with a short summary of what was placed.
            Aim for 2-3 tool calls total. The corridor system handles multi-road placement automatically.

            Available tools (6 total):
            - inspect_street: full context for any street/actor/corridor
            - find_streets: search by tags, parking, lane types
            - inspect_actor: actor detail (position, timeline)
            - edit_scene: all mutations (actors, weather, traffic lights)
            - manage_scenario: list/read/create/duplicate/rename/delete/switch scenarios
            - run_simulation: execute with optional weather + traffic light config

            Rules:
            - find_streets returns sections with driving_lanes and parking_lanes arrays plus a suggested_spawn. Use these directly.
            - Only call get_road if you need specific lane details not available from find_streets.
            - Only use road ids, section ids, and lane ids returned by the tools.
            - Apply edits through edit_scene; do not describe a change as complete unless it succeeded.
            - Batch all actor placements into ONE edit_scene call with multiple operations.
            - Use switch_map only when changing the active CARLA world is necessary.
            - Default vehicle blueprint: vehicle.tesla.model3
            - Default walker blueprint: walker.pedestrian.0001
            - When the user asks for an ego vehicle, set role=ego in the initial add_actor operation.
            - Parked vehicles: is_static=true, speed_kph=0, autopilot=false.
            - Space actors by distance along the street. Use get_street_context to see existing actors and find gaps.
            - When adding many similar actors on one road, prefer add_actor_row over many add_actor edits.
            - Route-follow vehicles may set lane_facing=against_lane for wrong-way or route_direction=reverse for backing.
            - Use ram_actor for intentional collisions; autopilot alone will not do that reliably.
            - Use turn_left/right_at_next_intersection for junction turns under Traffic Manager.
            - Keep the final reply short and state exactly what changed.

            Timeline actions for vehicles:
            - follow_route, set_speed, stop, hold_position, enable_autopilot, disable_autopilot
            - lane_change_left, lane_change_right, turn_left/right_at_next_intersection
            - chase_actor, ram_actor (target_actor_id required)
            - drive_reverse: applies CARLA vehicle.apply_control(reverse=True). This is physical reverse gear, NOT route_direction='reverse' which is route-follow backward along a path.
            - creep_forward: slow cautious movement at 1-10 kph (target_speed_kph clamped to 10 max)
            - yield_to_actor: wait for target actor to pass then proceed (target_actor_id + following_distance_m)
            - swerve: sudden lateral movement (swerve_direction=left/right, swerve_magnitude=0.5-3.0m)

            Weather and environment tools:
            - set_weather: control cloudiness, precipitation, fog, wind, sun angle
            - get_street_furniture: query traffic lights with positions and stop waypoints
            - set_traffic_light_state: freeze and set a traffic light to red/yellow/green
            - road_position_to_world: convert (road_id, s_fraction, lane_id) to (x, y, yaw)

            Dataset and scenario management tools:
            - list_dataset_scenarios: list all scenarios in the current dataset
            - read_scenario: read full draft of a scenario (actors, roads, map, duration)
            - create_scenario: create a new empty scenario in the dataset
            - duplicate_scenario: copy an existing scenario (actors, roads, settings) into a new one
            - rename_scenario: change a scenario's display name
            - delete_scenario: permanently remove a scenario from the dataset
            - switch_active_scenario: switch the editor to view/edit a different scenario
            - run_simulation: run the current scenario as a CARLA simulation

            When working with datasets:
            - Use list_dataset_scenarios first to see what exists before making changes.
            - Use read_scenario to inspect a scenario before duplicating or modifying it.
            - After creating or duplicating a scenario, use switch_active_scenario so the user can see it.
            - When asked to create variations, duplicate the source scenario and modify the copy.
            """
        ).strip()

    def _scene_capsule(self, request: SceneAssistantRequest) -> str:
        selected_actor_line = request.selected_actor_id or "none"
        # Fetch map context from Aurora DB (enriched data)
        map_context_lines = ""
        try:
            from ..map_context import get_map_context as _get_map_ctx
            ctx = _get_map_ctx(request.map_name)
            if ctx:
                desc = ctx.get("description", "")
                place = ctx.get("place_context", {})
                city = place.get("city", "")
                state = place.get("state", "")
                if desc:
                    map_context_lines += "\n- location: " + desc[:150]
                if city:
                    map_context_lines += "\n- area: " + city + (", " + state if state else "")
                enrichment = ctx.get("enrichment", {})
                features = enrichment.get("feature_counts", {})
                if features:
                    feat_parts = [f"{v} {k}" for k, v in features.items() if v]
                    if feat_parts:
                        map_context_lines += "\n- nearby: " + ", ".join(feat_parts)
                tags = ctx.get("tags", [])
                if tags:
                    map_context_lines += "\n- features: " + ", ".join(tags[:8])
                candidates = ctx.get("candidate_locations", [])
                if candidates:
                    map_context_lines += "\n- scenario locations:"
                    for cl in candidates[:5]:
                        label = cl.get("label", "")
                        kind = cl.get("kind", "")
                        if label:
                            map_context_lines += "\n    " + label + " (" + kind + ")"
        except Exception:
            pass  # Map context is optional enrichment

        corridor_lines = ""
        if hasattr(self, '_cached_corridors') and self._cached_corridors:
            corridors = self._cached_corridors
            multi = [cd for cd in corridors if (len(cd.road_ids) if hasattr(cd, 'road_ids') else cd.get('road_count', 1)) > 1]
            corridor_lines = "\n- streets: " + str(len(corridors)) + " (" + str(len(multi)) + " multi-segment)"
            sorted_c = sorted(corridors, key=lambda cd: -(cd.total_length if hasattr(cd, 'total_length') else cd.get('total_length', 0)))
            for cd in sorted_c[:5]:
                cid = cd.id if hasattr(cd, 'id') else cd.get('id', '')
                desc = cd.description if hasattr(cd, 'description') else cd.get('description', '')
                corridor_lines += "\n  " + cid + ": " + desc
            if len(corridors) > 5:
                corridor_lines += "\n  ...and " + str(len(corridors) - 5) + " more. Use get_corridors for full list."
        return dedent(
            f"""
            Scene capsule:
            - map: {request.map_name}
            - selected roads: {len(request.selected_roads)}
            - actors: {len(request.actors)}
            - live actors: {len(request.live_actors)}
            - selected actor id: {selected_actor_line}{map_context_lines}{corridor_lines}

            Place actors with: {{street: "corridor-id", distance_m: N, lane: "right driving"}}
            Use inspect_street(ref) for full street details.
            """
        ).strip()

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_corridors",
                "description": "Return all road corridors (logical streets) with total length, lane config, and road IDs. Use to understand the road network topology.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_street_context",
                "description": "Get full street context: corridor details, lanes per segment, connected streets, existing actors, placement hints. Pass a road_id or actor_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref": {"type": "string", "description": "A road_id or actor_id"}},
                    "required": ["ref"],
                },
            },
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
                "name": "find_streets",
                "description": "Search roads and get placement-ready results. Each result includes lane IDs and a suggested_spawn (road_id, section_id, lane_id, s_fraction) — pass these directly to edit_scene. Do NOT call get_road afterward, the lane info is already here. Set scope='current_map' for active map, scope='all_maps' for all maps.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": ["current_map", "all_maps"], "default": "current_map"},
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
                "name": "get_map_context",
                "description": "Get real-world context for a CARLA map: location description, geographic tags, nearby points of interest, and candidate scenario locations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "carla_map_name": {"type": "string", "description": "CARLA map name. Leave empty for the current map."},
                    },
                },
            },
            {
                "name": "search_maps",
                "description": "Search all available maps by scenario tags or keywords. Use this to find the best map for a scenario (e.g., find maps with school zones, parking, traffic lights, highways). Returns map name, CARLA map name, tags, and description for each match.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by map-level tags. Maps must contain ALL specified tags. Examples: SCHOOL_ZONE_BOUNDARY, TRANSIT_BUS_STOP, TURN_UNPROTECTED_LEFT, PARKING_LANE_NEAR_INTERSECTION_OCCLUSION, INTERSECTION_SIGNALIZED, NARROW_RESIDENTIAL_STREET_WITH_PARKING, FREEWAY_HIGHWAY_MAIN_CORRIDOR, WORK_ZONE_LANE_CLOSURE_STAGING"
                        },
                        "query": {
                            "type": "string",
                            "description": "Free-text search on map name and description. Case-insensitive partial match."
                        }
                    }
                }
            },
            
            {
                "name": "edit_scene",
                "description": "Apply scene mutations. Operation types: add_actor, add_actor_row, update_actor, remove_actor, replace_timeline, add_timeline_clip, remove_timeline_clip, select_actor, add_selected_roads, remove_selected_roads, set_selected_roads.",
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
            {
                "name": "set_weather",
                "description": "Set CARLA weather parameters. All params optional: cloudiness (0-100), precipitation (0-100), precipitation_deposits (0-100), wind_intensity (0-100), fog_density (0-100), fog_distance (0-500m), sun_altitude_angle (-90 to 90), sun_azimuth_angle (0-360).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cloudiness": {"type": "number", "minimum": 0, "maximum": 100},
                        "precipitation": {"type": "number", "minimum": 0, "maximum": 100},
                        "precipitation_deposits": {"type": "number", "minimum": 0, "maximum": 100},
                        "wind_intensity": {"type": "number", "minimum": 0, "maximum": 100},
                        "fog_density": {"type": "number", "minimum": 0, "maximum": 100},
                        "fog_distance": {"type": "number", "minimum": 0, "maximum": 500},
                        "sun_altitude_angle": {"type": "number", "minimum": -90, "maximum": 90},
                        "sun_azimuth_angle": {"type": "number", "minimum": 0, "maximum": 360},
                    },
                },
            },
            {
                "name": "get_street_furniture",
                "description": "Query CARLA traffic lights and poles. Returns traffic_light_id, type, position, stop_waypoints for each traffic light.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "set_traffic_light_state",
                "description": "Set a traffic light to a specific state. Freezes all traffic lights first, then sets the target. Use get_street_furniture first to find traffic_light_id values.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "traffic_light_id": {"type": "integer", "description": "CARLA actor id of the traffic light"},
                        "state": {"type": "string", "enum": ["red", "yellow", "green"]},
                        "duration_seconds": {"type": "number", "minimum": 0, "maximum": 300},
                    },
                    "required": ["traffic_light_id", "state"],
                },
            },
            {
                "name": "road_position_to_world",
                "description": "Convert an OpenDRIVE road position (road_id, s_fraction, lane_id) to world coordinates (x, y, yaw).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "road_id": {"type": "integer"},
                        "s_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                        "lane_id": {"type": "integer"},
                    },
                    "required": ["road_id", "s_fraction", "lane_id"],
                },
            },
            {
                "name": "list_dataset_scenarios",
                "description": "List all scenarios in the current dataset with id, display_name, actor_count, and map_name.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_scenario",
                "description": "Read the full draft of a scenario including actors, selected_roads, map_name, and duration.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string", "description": "The scenario ID to read"},
                    },
                    "required": ["scenario_id"],
                },
            },
            {
                "name": "create_scenario",
                "description": "Create a new empty scenario in the current dataset.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "display_name": {"type": "string", "description": "Name for the new scenario"},
                        "map_name": {"type": "string", "description": "CARLA map name (defaults to current map if omitted)"},
                    },
                    "required": ["display_name"],
                },
            },
            {
                "name": "duplicate_scenario",
                "description": "Duplicate an existing scenario: copies all actors, roads, and settings into a new scenario.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_scenario_id": {"type": "string", "description": "ID of the scenario to copy"},
                        "new_name": {"type": "string", "description": "Display name for the new copy"},
                    },
                    "required": ["source_scenario_id", "new_name"],
                },
            },
            {
                "name": "rename_scenario",
                "description": "Change the display name of an existing scenario.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string"},
                        "new_name": {"type": "string"},
                    },
                    "required": ["scenario_id", "new_name"],
                },
            },
            {
                "name": "delete_scenario",
                "description": "Permanently delete a scenario from the dataset.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string"},
                    },
                    "required": ["scenario_id"],
                },
            },
            {
                "name": "switch_active_scenario",
                "description": "Switch the editor to view and edit a different scenario. Does not make an HTTP call — the editor client handles the switch.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string", "description": "ID of the scenario to switch to"},
                        "reason": {"type": "string", "description": "Why the switch is happening"},
                    },
                    "required": ["scenario_id"],
                },
            },
            {
                "name": "run_simulation",
                "description": "Run the current scenario as a CARLA simulation on the GPU orchestrator.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "duration_seconds": {"type": "number", "default": 10, "description": "Simulation duration in seconds"},
                    },
                },
            },
        ]

    def _invoke(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Wrap system prompt with cache_control for prompt caching
        system_content = payload.get("system", "")
        if isinstance(system_content, str):
            system_content = [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]

        response = self.client.messages.create(
            model=model_id,
            max_tokens=payload.get("max_tokens", 32000),
            temperature=payload.get("temperature", 1),
            system=system_content,
            tools=payload.get("tools", []),
            messages=payload.get("messages", []),
            thinking={"type": "disabled"},
        )
        return {
            "content": [block.model_dump() for block in response.content],
            "stop_reason": response.stop_reason,
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            },
        }


    def _invoke_stream(self, model_id: str, payload: dict[str, Any]):
        """Streaming version of _invoke. Yields (event_type, data) tuples. Retries on transient errors."""
        import time as _time

        system_content = payload.get("system", "")
        if isinstance(system_content, str):
            system_content = [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.client.messages.stream(
                    model=model_id,
                    max_tokens=payload.get("max_tokens", 32000),
                    temperature=payload.get("temperature", 1),
                    system=system_content,
                    tools=payload.get("tools", []),
                    messages=payload.get("messages", []),
                    thinking={"type": "disabled"},
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "thinking_delta":
                                yield ("thinking", delta.thinking)
                            elif delta.type == "text_delta":
                                yield ("text", delta.text)

                    response = stream.get_final_message()
                    yield ("done", {
                        "content": [block.model_dump() for block in response.content],
                        "stop_reason": response.stop_reason,
                        "model": response.model,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                        },
                    })
                    return  # Success
            except Exception as e:
                error_str = str(e).lower()
                is_transient = "overloaded" in error_str or "rate" in error_str or "529" in error_str or "503" in error_str
                if is_transient and attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"[_invoke_stream] Transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                    _time.sleep(wait)
                    continue
                raise

    def _tool_result_content(self, result: dict[str, Any]) -> list[dict[str, str]]:
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=True)}]

    def _text_from_content(self, content: list[dict[str, Any]]) -> str:
        parts = [str(part.get("text", "")).strip() for part in content if part.get("type") == "text"]
        return "\n".join(part for part in parts if part).strip()


    def _sanitize_content_for_api(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip extra fields from content blocks that the API rejects (parsed_output, citations, signature)."""
        allowed_keys = {
            "thinking": {"type", "thinking", "signature"},
            "text": {"type", "text"},
            "tool_use": {"type", "id", "name", "input"},
            "tool_result": {"type", "tool_use_id", "content", "is_error"},
        }
        sanitized = []
        for block in content:
            block_type = block.get("type", "")
            keys = allowed_keys.get(block_type)
            if keys:
                sanitized.append({k: v for k, v in block.items() if k in keys})
            else:
                sanitized.append(block)
        return sanitized

    def _run_tool(self, state: SceneEditorState, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        # === Consolidated 6-tool handler ===
        if name == "inspect_street":
            return state.get_street_context(str(tool_input.get("ref") or ""))
        if name == "find_streets":
            return state.find_roads(tool_input)
        if name == "inspect_actor":
            return state.actor_details(str(tool_input.get("ref") or ""))
        if name == "edit_scene":
            ops = [dict(item) for item in tool_input.get("operations") or []]
            # Resolve abstract placements (street + distance_m + lane) to concrete anchors
            for op in ops:
                actor = op.get("actor", {})
                placement = actor.get("placement") or op.get("placement")
                if placement and "street" in placement:
                    try:
                        from ..road_corridors import resolve_corridor_distance, parse_lane_spec
                        corridor_id = str(placement["street"])
                        distance_m = float(placement.get("distance_m", 0))
                        lane_spec = str(placement.get("lane", "right driving"))
                        # Resolve corridor distance to road_id + s_fraction
                        corridor = state.corridors_by_id.get(corridor_id)
                        if corridor:
                            road_id, s_fraction = resolve_corridor_distance(corridor, distance_m)
                            # Resolve lane spec to lane_id
                            side, lane_type = parse_lane_spec(lane_spec)
                            segments = state.segments_by_road.get(str(road_id), [])
                            lane_id = None
                            for seg in segments:
                                seg_type = getattr(seg, "lane_type", "").lower()
                                seg_lid = getattr(seg, "lane_id", 0)
                                if seg_type == lane_type:
                                    if (side == "right" and seg_lid < 0) or (side == "left" and seg_lid > 0):
                                        lane_id = seg_lid
                                        break
                            # If no exact match, pick first driving lane on the right side
                            if lane_id is None:
                                for seg in segments:
                                    seg_lid = getattr(seg, "lane_id", 0)
                                    if side == "right" and seg_lid < 0:
                                        lane_id = seg_lid
                                        break
                                    elif side == "left" and seg_lid > 0:
                                        lane_id = seg_lid
                                        break
                            # Check if this is a parking placement — use world coordinates
                            if lane_type == "parking":
                                # Compute synthetic parking lane_id (editor convention)
                                # Right parking: -(drivingRight + index), Left parking: (drivingLeft + index)
                                road_sections = []
                                for rd in state.runtime_map.road_summaries:
                                    if str(getattr(rd, "id", "")) == str(road_id):
                                        road_sections = getattr(rd, "section_summaries", [])
                                        break
                                driving_count = 0
                                if road_sections:
                                    sec = road_sections[0]
                                    if side == "right":
                                        driving_count = getattr(sec, "driving_right", 1)
                                    else:
                                        driving_count = getattr(sec, "driving_left", 1)
                                if driving_count == 0:
                                    driving_count = 1
                                parking_lane_id = -(driving_count + 1) if side == "right" else (driving_count + 1)
                                # Set road anchor with synthetic parking lane_id
                                if "spawn" not in actor:
                                    actor["spawn"] = {}
                                actor["spawn"]["road_id"] = str(road_id)
                                actor["spawn"]["s_fraction"] = s_fraction
                                actor["spawn"]["lane_id"] = parking_lane_id
                                actor["placement_mode"] = "road"
                                actor["is_static"] = True
                                actor["speed_kph"] = 0
                                actor["autopilot"] = False
                                op["actor"] = actor
                                continue
                            # Set the resolved spawn on the actor (driving lane path)
                            if "spawn" not in actor:
                                actor["spawn"] = {}
                            actor["spawn"]["road_id"] = str(road_id)
                            actor["spawn"]["s_fraction"] = s_fraction
                            if lane_id is not None:
                                actor["spawn"]["lane_id"] = lane_id
                            op["actor"] = actor
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning("Abstract placement resolution failed: %s", exc)
            return state.apply_operations(ops)
        if name == "manage_scenario":
            return self._handle_manage_scenario(state, tool_input)
        if name == "run_simulation":
            return self._handle_run_simulation(state, tool_input)
        # Legacy tool names (backward compat during transition)
        if name == "get_street_context":
            return state.get_street_context(str(tool_input.get("ref") or ""))
        if name == "get_corridors":
            return state.get_corridors_summary()
        if name == "get_scene_overview":
            return state.scene_overview()
        if name == "get_actor":
            return state.actor_details(str(tool_input.get("actor_ref") or ""))
        if name == "get_road":
            return state.road_details(str(tool_input.get("road_id") or ""))
        if name == "get_adjacent_lanes":
            return state.adjacent_lanes(str(tool_input.get("road_id") or ""), int(tool_input.get("section_id")), int(tool_input.get("lane_id")))
        if name == "find_nearest_lane":
            return state.nearest_lane(float(tool_input.get("x")), float(tool_input.get("y")))
        if name == "find_streets":
            scope = str(tool_input.pop("scope", "current_map"))
            if scope == "all_maps":
                return state.search_maps_by_road(tool_input)
            return state.find_roads(tool_input)
        if name == "switch_map":
            return {"error": "Map switching is not available. Please switch maps from the editor UI."}
        if name == "get_map_context":
            from ..map_context import get_map_context as get_map_context_from_db
            target = str(tool_input.get("carla_map_name") or "").strip() or state.map_name
            result = get_map_context_from_db(target)
            if result is None:
                return {"error": f"No map asset found for CARLA map '{target}'"}
            return result
        if name == "edit_scene":
            ops = [dict(item) for item in tool_input.get("operations") or []]
            # Resolve abstract placements (street + distance_m + lane) to concrete anchors
            for op in ops:
                actor = op.get("actor", {})
                placement = actor.get("placement") or op.get("placement")
                if placement and "street" in placement:
                    try:
                        from ..road_corridors import resolve_corridor_distance, parse_lane_spec
                        corridor_id = str(placement["street"])
                        distance_m = float(placement.get("distance_m", 0))
                        lane_spec = str(placement.get("lane", "right driving"))
                        # Resolve corridor distance to road_id + s_fraction
                        corridor = state.corridors_by_id.get(corridor_id)
                        if corridor:
                            road_id, s_fraction = resolve_corridor_distance(corridor, distance_m)
                            # Resolve lane spec to lane_id
                            side, lane_type = parse_lane_spec(lane_spec)
                            segments = state.segments_by_road.get(str(road_id), [])
                            lane_id = None
                            for seg in segments:
                                seg_type = getattr(seg, "lane_type", "").lower()
                                seg_lid = getattr(seg, "lane_id", 0)
                                if seg_type == lane_type:
                                    if (side == "right" and seg_lid < 0) or (side == "left" and seg_lid > 0):
                                        lane_id = seg_lid
                                        break
                            # If no exact match, pick first driving lane on the right side
                            if lane_id is None:
                                for seg in segments:
                                    seg_lid = getattr(seg, "lane_id", 0)
                                    if side == "right" and seg_lid < 0:
                                        lane_id = seg_lid
                                        break
                                    elif side == "left" and seg_lid > 0:
                                        lane_id = seg_lid
                                        break
                            # Check if this is a parking placement — use world coordinates
                            if lane_type == "parking":
                                # Compute synthetic parking lane_id (editor convention)
                                # Right parking: -(drivingRight + index), Left parking: (drivingLeft + index)
                                road_sections = []
                                for rd in state.runtime_map.road_summaries:
                                    if str(getattr(rd, "id", "")) == str(road_id):
                                        road_sections = getattr(rd, "section_summaries", [])
                                        break
                                driving_count = 0
                                if road_sections:
                                    sec = road_sections[0]
                                    if side == "right":
                                        driving_count = getattr(sec, "driving_right", 1)
                                    else:
                                        driving_count = getattr(sec, "driving_left", 1)
                                if driving_count == 0:
                                    driving_count = 1
                                parking_lane_id = -(driving_count + 1) if side == "right" else (driving_count + 1)
                                # Set road anchor with synthetic parking lane_id
                                if "spawn" not in actor:
                                    actor["spawn"] = {}
                                actor["spawn"]["road_id"] = str(road_id)
                                actor["spawn"]["s_fraction"] = s_fraction
                                actor["spawn"]["lane_id"] = parking_lane_id
                                actor["placement_mode"] = "road"
                                actor["is_static"] = True
                                actor["speed_kph"] = 0
                                actor["autopilot"] = False
                                op["actor"] = actor
                                continue
                            # Set the resolved spawn on the actor (driving lane path)
                            if "spawn" not in actor:
                                actor["spawn"] = {}
                            actor["spawn"]["road_id"] = str(road_id)
                            actor["spawn"]["s_fraction"] = s_fraction
                            if lane_id is not None:
                                actor["spawn"]["lane_id"] = lane_id
                            op["actor"] = actor
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning("Abstract placement resolution failed: %s", exc)
            return state.apply_operations(ops)
        if name == "set_weather":
            if self.carla_metadata is None:
                return {"error": "Weather control unavailable."}
            return self.carla_metadata.set_weather(tool_input)
        if name == "get_street_furniture":
            if self.carla_metadata is None:
                return {"error": "Street furniture unavailable."}
            return self.carla_metadata.get_street_furniture()
        if name == "set_traffic_light_state":
            if self.carla_metadata is None:
                return {"error": "Traffic light control unavailable."}
            return self.carla_metadata.set_traffic_light_state(tool_input)
        if name == "road_position_to_world":
            if self.carla_metadata is None:
                return {"error": "Position conversion unavailable."}
            return self.carla_metadata.road_position_to_world(tool_input)
        # Dataset tools -> manage_scenario
        dataset_actions = {"list_dataset_scenarios": "list", "read_scenario": "read", "create_scenario": "create", "duplicate_scenario": "duplicate", "rename_scenario": "rename", "delete_scenario": "delete", "switch_active_scenario": "switch"}
        if name in dataset_actions:
            tool_input["action"] = dataset_actions[name]
            return self._handle_manage_scenario(state, tool_input)
        if name == "run_simulation":
            return self._handle_run_simulation(state, tool_input)
        raise RuntimeError(f"Unsupported tool: {name}")

    def _handle_manage_scenario(self, state: SceneEditorState, tool_input: dict[str, Any]) -> dict[str, Any]:
        if not state.callback_url:
            return {"error": "Scenario management requires the editor connection. This feature is not available in standalone mode."}
        action = str(tool_input.get("action", ""))
        if action == "list":
            return self._editor_request(state, "GET", "/api/datasets/" + str(getattr(state, "dataset_id", "") or "") + "/scenarios")
        if action == "read":
            sid = tool_input.get("scenario_id", "")
            return self._editor_request(state, "GET", "/api/scenarios/" + str(sid))
        if action == "create":
            return self._editor_request(state, "POST", "/api/datasets/" + str(getattr(state, "dataset_id", "") or "") + "/scenarios", {"name": tool_input.get("name", "New Scenario")})
        if action == "duplicate":
            return self._editor_request(state, "POST", "/api/scenarios/" + str(tool_input.get("source_scenario_id", "")) + "/duplicate", {"name": tool_input.get("name", "")})
        if action == "rename":
            return self._editor_request(state, "PATCH", "/api/scenarios/" + str(tool_input.get("scenario_id", "")), {"display_name": tool_input.get("name", "")})
        if action == "delete":
            return self._editor_request(state, "DELETE", "/api/scenarios/" + str(tool_input.get("scenario_id", "")))
        if action == "switch":
            return {"switch_to_scenario_id": tool_input.get("scenario_id", ""), "note": "Editor will switch to this scenario."}
        return {"error": f"Unknown action: {action}. Use: list, read, create, duplicate, rename, delete, switch"}

    def _handle_run_simulation(self, state: SceneEditorState, tool_input: dict[str, Any]) -> dict[str, Any]:
        # Apply weather if specified
        weather = tool_input.get("weather")
        if weather and self.carla_metadata:
            if isinstance(weather, str):
                self.carla_metadata.set_weather({"preset": weather})
            elif isinstance(weather, dict):
                self.carla_metadata.set_weather(weather)
        # Apply traffic light overrides
        tl = tool_input.get("traffic_lights")
        if tl and self.carla_metadata:
            for tl_config in tl:
                self.carla_metadata.set_traffic_light_state(tl_config)
        # Run simulation via editor callback
        return {"status": "simulation_requested", "weather_applied": weather is not None, "traffic_lights_set": tl is not None, "record": tool_input.get("record", True)}


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
                "inspect_street",
                "Get full context for a street: corridor details, available lanes per segment, connected streets at intersections, existing actors, and placement hints. Pass a road_id, actor_id, or corridor_id.",
                InspectStreetInput,
            ),
            make_tool(
                "find_streets",
                "Search for streets by tags, parking, lane types, and lane-count characteristics. Returns matching streets with corridor context.",
                FindStreetsInput,
            ),
            make_tool(
                "inspect_actor",
                "Return detailed information about a specific actor by id or label, including position, timeline, and properties.",
                InspectActorInput,
            ),
            make_tool(
                "edit_scene",
                "Apply scene mutations. Operation types: add_actor, add_actor_row, update_actor, remove_actor, replace_timeline, add_timeline_clip, remove_timeline_clip, select_actor, add_selected_roads, remove_selected_roads, set_selected_roads, set_weather, set_traffic_light. Use placement format: {street: corridor_id, distance_m: N, lane: 'right driving'} for actor placement.",
                EditSceneInput,
            ),
            make_tool(
                "manage_scenario",
                "Manage scenarios in the current dataset. Actions: list (list all), read (read full draft), create (new empty), duplicate (copy existing), rename (change name), delete (remove), switch (change active scenario). Pass action + relevant params.",
                ManageScenarioInput,
            ),
            make_tool(
                "run_simulation",
                "Run the current scenario as a CARLA simulation. Optionally set weather (preset name or params), traffic light states, and duration. Records by default.",
                RunSimulationInput,
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
        # Inject corridor data from carla_metadata cache
        if self.carla_metadata is not None:
            try:
                with self.carla_metadata._lock:
                    cached_corridors = self.carla_metadata._corridor_cache.get(request.map_name) or self.carla_metadata._corridor_cache.get(getattr(self.carla_metadata, "_current_map_name", ""))
                if cached_corridors:
                    state.set_corridors(cached_corridors)
                    self._cached_corridors = cached_corridors
                # Also populate runtime segments from carla_metadata if editor sent empty
                if not state.runtime_map.road_segments:
                    try:
                        rt = self.carla_metadata.get_runtime_map(force_refresh=False)
                        state.runtime_map = rt
                        state._rebuild_runtime_indexes()
                        import logging
                        logging.getLogger(__name__).info("Populated %d runtime segments from carla_metadata", len(rt.road_segments))
                    except Exception as _exc:
                        import logging
                        logging.getLogger(__name__).warning("Failed to populate runtime segments: %s", _exc)
            except Exception:
                pass  # Corridors are optional
        messages = self._langchain_message_history(request)
        tool_map = self._langchain_tools(state, trace)
        model = create_chat_model(
            model_id,
            temperature=1,
            max_tokens=32000,
            thinking={"type": "disabled"},
        ).bind_tools(list(tool_map.values()))
        raw_response: dict[str, Any] = {}

        for _ in range(30):
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
                # Skip thinking blocks, only extract text
                content = getattr(ai_message, "content", "")
                if isinstance(content, list):
                    content = [part for part in content if not (isinstance(part, dict) and part.get("type") == "thinking")]
                reply = self._langchain_text(content) or "I inspected the scene but do not have a textual reply."
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
            return self._chat_langchain(self.model_id, request)

        trace: list[SceneAssistantToolTrace] = []
        state = SceneEditorState(request)
        # Inject corridor data from carla_metadata cache
        if self.carla_metadata is not None:
            try:
                with self.carla_metadata._lock:
                    cached_corridors = self.carla_metadata._corridor_cache.get(request.map_name) or self.carla_metadata._corridor_cache.get(getattr(self.carla_metadata, "_current_map_name", ""))
                if cached_corridors:
                    state.set_corridors(cached_corridors)
                    self._cached_corridors = cached_corridors
                # Also populate runtime segments from carla_metadata if editor sent empty
                if not state.runtime_map.road_segments:
                    try:
                        rt = self.carla_metadata.get_runtime_map(force_refresh=False)
                        state.runtime_map = rt
                        state._rebuild_runtime_indexes()
                        import logging
                        logging.getLogger(__name__).info("Populated %d runtime segments from carla_metadata", len(rt.road_segments))
                    except Exception as _exc:
                        import logging
                        logging.getLogger(__name__).warning("Failed to populate runtime segments: %s", _exc)
            except Exception:
                pass  # Corridors are optional
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

        messages = deepcopy(base_messages)
        raw_response: dict[str, Any] = {}
        try:
            for _ in range(30):
                payload = {
                    "max_tokens": 32000,
                    "temperature": 1,
                    "system": self._system_prompt(),
                    "tools": self._tool_definitions(),
                    "messages": messages,
                }
                raw_response = self._invoke(self.model_id, payload)
                content = raw_response.get("content") or []
                messages.append({"role": "assistant", "content": self._sanitize_content_for_api(content)})

                # Filter: only look at tool_use blocks for tool execution
                tool_uses = [part for part in content if part.get("type") == "tool_use"]
                if not tool_uses:
                    # Get text from non-thinking blocks
                    text_parts = [part for part in content if part.get("type") == "text"]
                    reply = self._text_from_content(text_parts) or "I inspected the scene but do not have a textual reply."
                    self.model_id = self.model_id
                    return state.response(self.model_id, reply, trace, raw_response)

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
        except Exception as exc:
            raise RuntimeError(f"Scene assistant failed: {exc}") from exc

    def chat_stream(self, request: SceneAssistantRequest):
        """Generator that yields SSE-formatted events for each step of the tool-use loop.

        Events:
          - thinking: LLM reasoning / thinking blocks
          - tool_call: tool name + input when Claude calls a tool
          - tool_result: tool name + output after execution
          - actor_update: current actors + roads after edit_scene
          - complete: final response (same payload as sync endpoint)
          - error: on any error
        """
        import traceback as tb

        trace: list[SceneAssistantToolTrace] = []
        state = SceneEditorState(request)
        # Inject corridor data from carla_metadata cache
        if self.carla_metadata is not None:
            try:
                with self.carla_metadata._lock:
                    cached_corridors = self.carla_metadata._corridor_cache.get(request.map_name) or self.carla_metadata._corridor_cache.get(getattr(self.carla_metadata, "_current_map_name", ""))
                if cached_corridors:
                    state.set_corridors(cached_corridors)
                    self._cached_corridors = cached_corridors
                # Also populate runtime segments from carla_metadata if editor sent empty
                if not state.runtime_map.road_segments:
                    try:
                        rt = self.carla_metadata.get_runtime_map(force_refresh=False)
                        state.runtime_map = rt
                        state._rebuild_runtime_indexes()
                        import logging
                        logging.getLogger(__name__).info("Populated %d runtime segments from carla_metadata", len(rt.road_segments))
                    except Exception as _exc:
                        import logging
                        logging.getLogger(__name__).warning("Failed to populate runtime segments: %s", _exc)
            except Exception:
                pass  # Corridors are optional
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

        messages = deepcopy(base_messages)

        def _sse(event: str, data: Any) -> str:
            payload = json.dumps(data, ensure_ascii=True, default=str)
            return f"event: {event}\ndata: {payload}\n\n"

        try:
            system_content = self._system_prompt()
            if isinstance(system_content, str):
                system_content = [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]

            for turn in range(30):
                # Use streaming API for real-time text/thinking deltas
                raw_response = None
                # Retry transient API errors (overloaded, rate limit)
                import time as _time
                _stream_ok = False
                for _retry in range(3):
                    try:
                        for evt_type, evt_data in self._invoke_stream(self.model_id, {
                            "max_tokens": 32000,
                            "temperature": 1,
                            "system": system_content,
                            "tools": self._tool_definitions(),
                            "messages": messages,
                        }):
                            if evt_type == "thinking":
                                yield _sse("thinking", {"thinking": evt_data})
                            elif evt_type == "text":
                                yield _sse("text", {"text": evt_data})
                            elif evt_type == "done":
                                raw_response = evt_data
                        _stream_ok = True
                        break
                    except Exception as _retry_err:
                        _err_str = str(_retry_err).lower()
                        if ("overloaded" in _err_str or "529" in _err_str or "503" in _err_str) and _retry < 2:
                            _wait = (_retry + 1) * 10
                            print(f"[chat_stream] Transient API error (attempt {_retry+1}/3), retrying in {_wait}s")
                            _time.sleep(_wait)
                            continue
                        raise

                if raw_response is None:
                    yield _sse("error", {"error": "Stream ended without final response"})
                    return

                content = raw_response.get("content") or []
                messages.append({"role": "assistant", "content": self._sanitize_content_for_api(content)})

                # Check for tool uses
                tool_uses = [part for part in content if part.get("type") == "tool_use"]

                if not tool_uses:
                    # Final text response - no more tool calls
                    text_parts = [part for part in content if part.get("type") == "text"]
                    reply = self._text_from_content(text_parts) or "I inspected the scene but do not have a textual reply."
                    response = state.response(self.model_id, reply, trace, raw_response)
                    yield _sse("complete", response.model_dump())
                    return

                # Process each tool call
                tool_results: list[dict[str, Any]] = []
                for tool_use in tool_uses:
                    name = str(tool_use.get("name") or "")
                    tool_input = dict(tool_use.get("input") or {})

                    # Emit tool_call event
                    yield _sse("tool_call", {"name": name, "input": tool_input, "tool_use_id": tool_use["id"]})

                    try:
                        result = self._run_tool(state, name, tool_input)
                    except Exception as tool_exc:
                        result = {"error": str(tool_exc)}

                    trace.append(SceneAssistantToolTrace(name=name, input=tool_input, result=result))

                    # Emit tool_result event
                    yield _sse("tool_result", {"name": name, "result": result, "tool_use_id": tool_use["id"]})

                    # Emit actor_update after edit_scene
                    if name == "edit_scene":
                        yield _sse("actor_update", {
                            "actors": [actor.model_dump() for actor in state.actors],
                            "selected_roads": [road.model_dump() for road in state.selected_roads],
                            "selected_actor_id": state.selected_actor_id,
                            "map_name": state.map_name,
                        })

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": self._tool_result_content(result),
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

            yield _sse("error", {"error": "Assistant exceeded the maximum tool-use rounds."})
        except Exception as exc:
            yield _sse("error", {"error": str(exc), "traceback": tb.format_exc()})
