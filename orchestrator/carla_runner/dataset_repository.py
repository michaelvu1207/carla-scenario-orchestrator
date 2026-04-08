from __future__ import annotations

import logging
from typing import Any

from .models import RuntimeRoadSectionSummary, RuntimeRoadSummary, SelectedRoad

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level generated map cache (replaces static maps.generated.json)
# ---------------------------------------------------------------------------
_generated_map_cache: dict[str, dict[str, Any]] = {}


def normalize_map_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/").split("/")[-1].strip()
    if normalized.endswith(".xodr"):
        normalized = normalized[:-5]
    return normalized


def set_generated_map_cache(map_name: str, generated_map: dict[str, Any]) -> None:
    key = normalize_map_name(map_name)
    if not key:
        return
    _generated_map_cache[key] = generated_map
    logger.debug("set_generated_map_cache: cached %s (%d roads)", key, len(generated_map.get("roads", [])))


def _get_generated_roads(map_name: str) -> list[dict[str, Any]]:
    cached = _generated_map_cache.get(normalize_map_name(map_name))
    return cached.get("roads", []) if cached else []


def _get_generated_stats(map_name: str) -> dict[str, Any]:
    cached = _generated_map_cache.get(normalize_map_name(map_name))
    return cached.get("stats", {}) if cached else {}


def list_supported_maps() -> list[str]:
    return sorted(_generated_map_cache.keys())


def build_selected_roads(map_name: str, road_ids: list[str]) -> list[SelectedRoad]:
    roads_data = _get_generated_roads(map_name)
    if not roads_data:
        return []
    wanted = {str(road_id) for road_id in road_ids}
    roads: list[SelectedRoad] = []
    for road in roads_data:
        if str(road.get("id")) not in wanted:
            continue
        roads.append(
            SelectedRoad(
                id=str(road.get("id")),
                name=road.get("name") or f"Road {road.get('id')}",
                length=float(road.get("length") or 0.0),
                tags=[str(tag) for tag in road.get("tags", [])],
                section_labels=[str(section.get("label")) for section in road.get("sections", [])],
            )
        )
    return roads


def build_runtime_road_summaries(map_name: str) -> list[RuntimeRoadSummary]:
    roads_data = _get_generated_roads(map_name)
    if not roads_data:
        return []

    summaries: list[RuntimeRoadSummary] = []
    for road in roads_data:
        section_summaries = [
            RuntimeRoadSectionSummary(
                index=int(section.get("index") or 0),
                label=str(section.get("label") or ""),
                s=float(section.get("s") or 0.0),
                driving_left=int(section.get("drivingLeft") or 0),
                driving_right=int(section.get("drivingRight") or 0),
                parking_left=int(section.get("parkingLeft") or 0),
                parking_right=int(section.get("parkingRight") or 0),
                total_driving=int(section.get("totalDriving") or 0),
                total_width=float(section.get("totalWidth") or 0.0),
                lane_types=[str(lane_type) for lane_type in section.get("laneTypes", [])],
                tags=[str(tag) for tag in section.get("tags", [])],
            )
            for section in road.get("sections", [])
        ]
        lane_types = sorted(
            {
                lane_type
                for section in section_summaries
                for lane_type in section.lane_types
            }
        )
        summaries.append(
            RuntimeRoadSummary(
                id=str(road.get("id")),
                name=str(road.get("name") or f"Road {road.get('id')}"),
                is_intersection=bool(road.get("isIntersection")),
                tags=[str(tag) for tag in road.get("tags", [])],
                lane_types=lane_types,
                has_parking="parking" in lane_types,
                has_shoulder="shoulder" in lane_types,
                has_sidewalk="sidewalk" in lane_types,
                section_summaries=section_summaries,
            )
        )
    return summaries


def dataset_lane_type_counts(map_name: str) -> dict[str, int]:
    stats = _get_generated_stats(map_name)
    counts = stats.get("laneTypes", {})
    return {str(key): int(value) for key, value in counts.items()}


def _road_sections_for_search(road: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(section) for section in road.get("sections", [])]


def _road_lane_types(road: dict[str, Any]) -> set[str]:
    lane_types: set[str] = set()
    for section in _road_sections_for_search(road):
        lane_types.update(str(lane_type).lower() for lane_type in section.get("laneTypes", []))
    return lane_types


def _road_matches_query_text(road: dict[str, Any], query: str) -> bool:
    if not query.strip():
        return True
    haystack_parts = [
        str(road.get("id") or ""),
        str(road.get("name") or ""),
        " ".join(str(tag) for tag in road.get("tags", [])),
    ]
    for section in _road_sections_for_search(road):
        haystack_parts.append(str(section.get("label") or ""))
        haystack_parts.append(" ".join(str(tag) for tag in section.get("tags", [])))
        haystack_parts.append(" ".join(str(lane_type) for lane_type in section.get("laneTypes", [])))
    haystack = " ".join(haystack_parts).lower()
    return all(token in haystack for token in query.lower().split())


def _section_matches_filters(
    section: dict[str, Any],
    *,
    driving_left: int | None,
    driving_right: int | None,
    total_driving: int | None,
    parking_left_min: int | None,
    parking_right_min: int | None,
    require_parking_on_both_sides: bool | None,
) -> bool:
    if driving_left is not None and int(section.get("drivingLeft") or 0) != driving_left:
        return False
    if driving_right is not None and int(section.get("drivingRight") or 0) != driving_right:
        return False
    if total_driving is not None and int(section.get("totalDriving") or 0) != total_driving:
        return False
    if parking_left_min is not None and int(section.get("parkingLeft") or 0) < parking_left_min:
        return False
    if parking_right_min is not None and int(section.get("parkingRight") or 0) < parking_right_min:
        return False
    if require_parking_on_both_sides is True:
        if int(section.get("parkingLeft") or 0) <= 0 or int(section.get("parkingRight") or 0) <= 0:
            return False
    return True


def _road_matches_filters(
    road: dict[str, Any],
    *,
    query: str = "",
    tags: list[str] | None = None,
    lane_types: list[str] | None = None,
    is_intersection: bool | None = None,
    has_parking: bool | None = None,
    driving_left: int | None = None,
    driving_right: int | None = None,
    total_driving: int | None = None,
    parking_left_min: int | None = None,
    parking_right_min: int | None = None,
    require_parking_on_both_sides: bool | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not _road_matches_query_text(road, query):
        return False, []
    road_tags = {str(tag).lower() for tag in road.get("tags", [])}
    wanted_tags = {str(tag).lower() for tag in tags or [] if str(tag).strip()}
    if wanted_tags and not wanted_tags.issubset(road_tags):
        return False, []
    road_lane_types = _road_lane_types(road)
    wanted_lane_types = {str(lane_type).lower() for lane_type in lane_types or [] if str(lane_type).strip()}
    if wanted_lane_types and not wanted_lane_types.issubset(road_lane_types):
        return False, []
    if is_intersection is not None and bool(road.get("isIntersection")) != is_intersection:
        return False, []
    if has_parking is True and "parking" not in road_lane_types:
        return False, []
    if has_parking is False and "parking" in road_lane_types:
        return False, []

    matching_sections = [
        section
        for section in _road_sections_for_search(road)
        if _section_matches_filters(
            section,
            driving_left=driving_left,
            driving_right=driving_right,
            total_driving=total_driving,
            parking_left_min=parking_left_min,
            parking_right_min=parking_right_min,
            require_parking_on_both_sides=require_parking_on_both_sides,
        )
    ]
    if any(
        value is not None
        for value in (
            driving_left,
            driving_right,
            total_driving,
            parking_left_min,
            parking_right_min,
            require_parking_on_both_sides,
        )
    ) and not matching_sections:
        return False, []
    return True, matching_sections or _road_sections_for_search(road)


def _road_search_result(map_name: str, road: dict[str, Any], matching_sections: list[dict[str, Any]]) -> dict[str, Any]:
    # Build lane info from sections so the LLM can place actors without calling get_road
    sections_with_lanes = []
    suggested_spawn = None
    for section in matching_sections:
        section_idx = int(section.get("index") or 0)
        driving_right = int(section.get("drivingRight") or 0)
        driving_left = int(section.get("drivingLeft") or 0)
        parking_right = int(section.get("parkingRight") or 0)
        parking_left = int(section.get("parkingLeft") or 0)
        # Driving lanes: negative IDs go right, positive go left (OpenDRIVE convention)
        driving_lane_ids = [-(i + 1) for i in range(driving_right)] + [(i + 1) for i in range(driving_left)]
        parking_lane_ids = []
        if parking_right > 0:
            parking_lane_ids.append(-(driving_right + 1))
        if parking_left > 0:
            parking_lane_ids.append(driving_left + 1)
        sections_with_lanes.append({
            "section_id": section_idx,
            "label": str(section.get("label") or ""),
            "driving_lanes": driving_lane_ids,
            "parking_lanes": parking_lane_ids,
            "total_driving": int(section.get("totalDriving") or 0),
            "tags": [str(tag) for tag in section.get("tags", [])],
        })
        # Suggest the first driving lane at mid-road as default spawn
        if suggested_spawn is None and driving_lane_ids:
            suggested_spawn = {
                "road_id": str(road.get("id")),
                "section_id": section_idx,
                "lane_id": driving_lane_ids[0],
                "s_fraction": 0.3,
            }
    return {
        "map_name": normalize_map_name(map_name),
        "road_id": str(road.get("id")),
        "name": str(road.get("name") or f"Road {road.get('id')}"),
        "length": float(road.get("length") or 0.0),
        "is_intersection": bool(road.get("isIntersection")),
        "tags": [str(tag) for tag in road.get("tags", [])],
        "sections": sections_with_lanes,
        "suggested_spawn": suggested_spawn,
    }


def search_roads(
    map_name: str,
    *,
    query: str = "",
    tags: list[str] | None = None,
    lane_types: list[str] | None = None,
    is_intersection: bool | None = None,
    has_parking: bool | None = None,
    driving_left: int | None = None,
    driving_right: int | None = None,
    total_driving: int | None = None,
    parking_left_min: int | None = None,
    parking_right_min: int | None = None,
    require_parking_on_both_sides: bool | None = None,
    limit: int = 5,
    runtime_road_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    roads_data = _get_generated_roads(map_name)
    if not roads_data:
        return []
    results: list[dict[str, Any]] = []
    for road in roads_data:
        # Skip roads not present in CARLA runtime (phantom roads)
        if runtime_road_ids is not None and str(road.get("id")) not in runtime_road_ids:
            continue
        matches, matching_sections = _road_matches_filters(
            road,
            query=query,
            tags=tags,
            lane_types=lane_types,
            is_intersection=is_intersection,
            has_parking=has_parking,
            driving_left=driving_left,
            driving_right=driving_right,
            total_driving=total_driving,
            parking_left_min=parking_left_min,
            parking_right_min=parking_right_min,
            require_parking_on_both_sides=require_parking_on_both_sides,
        )
        if not matches:
            continue
        results.append(_road_search_result(map_name, road, matching_sections))
        if len(results) >= max(1, limit):
            break
    return results


