"""Road corridor detection from OpenDRIVE xodr XML.

Chains connected non-junction road segments into logical streets (corridors)
using junction connectivity + heading alignment + lane count matching.
"""
from __future__ import annotations

import hashlib
import logging
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

HEADING_THRESHOLD_DEG = 25.0
AMBIGUITY_GUARD_DEG = 5.0
MAX_CHAIN_LENGTH = 20
BOUNDARY_BUFFER_M = 2.0


@dataclass
class RoadCorridor:
    id: str
    road_ids: list[str]
    junction_ids: list[str]
    total_length: float
    segment_lengths: list[float]
    segment_offsets: list[float]
    has_parking: bool
    lane_config: str
    bearing_deg: float
    description: str
    is_bidirectional: bool = True


def _as_float(value: str | None, fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _endpoint_heading(road: ET.Element, end: str) -> float | None:
    """Get heading at start or end of a road from planView geometry."""
    geometries = road.findall("./planView/geometry")
    if not geometries:
        return None
    if end == "start":
        return _as_float(geometries[0].get("hdg"))
    # end == "end": use the last geometry segment
    g = geometries[-1]
    hdg = _as_float(g.get("hdg"))
    length = _as_float(g.get("length"))
    arc = g.find("arc")
    if arc is not None:
        curv = _as_float(arc.get("curvature"))
        if abs(curv) > 1e-9:
            return hdg + curv * length
    return hdg


def _endpoint_position(road: ET.Element, end: str) -> tuple[float, float] | None:
    """Get (x, y) at start or end of a road."""
    geometries = road.findall("./planView/geometry")
    if not geometries:
        return None
    if end == "start":
        g = geometries[0]
        return (_as_float(g.get("x")), _as_float(g.get("y")))
    g = geometries[-1]
    x0 = _as_float(g.get("x"))
    y0 = _as_float(g.get("y"))
    hdg = _as_float(g.get("hdg"))
    length = _as_float(g.get("length"))
    arc = g.find("arc")
    if arc is not None:
        curv = _as_float(arc.get("curvature"))
        if abs(curv) > 1e-9:
            x = x0 + (math.sin(hdg + curv * length) - math.sin(hdg)) / curv
            y = y0 - (math.cos(hdg + curv * length) - math.cos(hdg)) / curv
            return (x, y)
    return (x0 + length * math.cos(hdg), y0 + length * math.sin(hdg))


def _driving_lane_count(road: ET.Element) -> int:
    """Count total driving lanes from the first lane section."""
    sections = road.findall("./lanes/laneSection")
    if not sections:
        return 0
    section = sections[0]
    count = 0
    for lane in section.findall("./left/lane"):
        if lane.get("type") == "driving":
            count += 1
    for lane in section.findall("./right/lane"):
        if lane.get("type") == "driving":
            count += 1
    return count


def _road_has_parking(road: ET.Element) -> bool:
    """Check if any lane section has parking lanes."""
    for section in road.findall("./lanes/laneSection"):
        for lane in section.findall("./left/lane"):
            if lane.get("type") == "parking":
                return True
        for lane in section.findall("./right/lane"):
            if lane.get("type") == "parking":
                return True
    return False


def _dominant_lane_config(road: ET.Element) -> str:
    """Get lane config label from first section (e.g., '1L/1R', '2L/2R')."""
    sections = road.findall("./lanes/laneSection")
    if not sections:
        return "0L/0R"
    section = sections[0]
    left = sum(1 for l in section.findall("./left/lane") if l.get("type") == "driving")
    right = sum(1 for l in section.findall("./right/lane") if l.get("type") == "driving")
    return f"{left}L/{right}R"


def _heading_diff(h1: float, h2: float, same_end: bool) -> float:
    """Compute heading difference in degrees between two road endpoints at a junction.

    If both roads have the same contact end (both 'start' or both 'end'),
    they approach from opposite directions — check anti-parallel alignment.
    If different contact ends (one 'start', one 'end'), check parallel alignment.
    """
    if same_end:
        # Anti-parallel: headings should differ by ~180°
        diff = abs(((h1 - h2 + math.pi) % (2 * math.pi)) - math.pi)
    else:
        # Parallel: headings should be similar
        diff = abs(((h1 - h2) % (2 * math.pi)))
        if diff > math.pi:
            diff = 2 * math.pi - diff
    return math.degrees(diff)


def _bearing_between(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute compass bearing (0-360) from p1 to p2 in OpenDRIVE coordinates."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # OpenDRIVE: x is east, y is north (but y is often flipped)
    angle = math.degrees(math.atan2(dx, -dy)) % 360
    return angle


def _bearing_to_compass(deg: float) -> str:
    """Convert bearing degrees to compass direction."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45) % 8
    return directions[idx]


def _stable_corridor_id(road_ids: list[str]) -> str:
    """Generate a deterministic corridor ID from sorted road IDs."""
    key = ",".join(sorted(road_ids))
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"corridor-{h}"


def build_corridors(xodr_text: str) -> list[RoadCorridor]:
    """Build road corridors from OpenDRIVE XML text.

    Algorithm:
    Step A: Build road→junction adjacency from <road><link> elements
    Step B: Resolve junction connections to road-to-road pairs using heading alignment
    Step C: Chain continuations into corridors
    """
    try:
        root = ET.fromstring(xodr_text)
    except ET.ParseError as exc:
        logger.warning("Failed to parse xodr XML for corridor building: %s", exc)
        return []

    roads = {r.get("id", ""): r for r in root.findall("road")}
    if not roads:
        return []

    # Non-junction roads only
    nj_roads = {
        rid: r for rid, r in roads.items()
        if r.get("junction", "-1") == "-1"
    }
    if not nj_roads:
        return []

    # --- Step A: Build road→junction adjacency ---
    junction_to_roads: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for rid, road in nj_roads.items():
        link = road.find("link")
        if link is None:
            continue
        for elem, end in [(link.find("predecessor"), "start"), (link.find("successor"), "end")]:
            if elem is not None and elem.get("elementType") == "junction":
                jid = elem.get("elementId", "")
                if jid:
                    junction_to_roads[jid].append((rid, end))

    # --- Step B: Detect continuation pairs at each junction ---
    continuations: dict[tuple[str, str], tuple[str, str]] = {}

    for jid, touches in junction_to_roads.items():
        if len(touches) < 2:
            continue

        # Compute all candidate pairs with heading + lane matching
        candidates: list[tuple[float, str, str, str, str]] = []
        for i in range(len(touches)):
            for j in range(i + 1, len(touches)):
                r1, e1 = touches[i]
                r2, e2 = touches[j]
                h1 = _endpoint_heading(nj_roads[r1], e1)
                h2 = _endpoint_heading(nj_roads[r2], e2)
                if h1 is None or h2 is None:
                    continue
                same_end = (e1 == e2)
                deg_diff = _heading_diff(h1, h2, same_end)
                if deg_diff >= HEADING_THRESHOLD_DEG:
                    continue
                # Lane count matching
                lc1 = _driving_lane_count(nj_roads[r1])
                lc2 = _driving_lane_count(nj_roads[r2])
                if abs(lc1 - lc2) > 1:
                    continue
                candidates.append((deg_diff, r1, e1, r2, e2))

        # Greedy best-first matching with ambiguity guard
        candidates.sort()
        used: set[tuple[str, str]] = set()
        for idx, (deg_diff, r1, e1, r2, e2) in enumerate(candidates):
            k1, k2 = (r1, e1), (r2, e2)
            if k1 in used or k2 in used:
                continue
            # Ambiguity guard: if the next candidate for the same road-end is within 5°
            ambiguous = False
            for other_idx in range(idx + 1, len(candidates)):
                od, or1, oe1, or2, oe2 = candidates[other_idx]
                if od - deg_diff > AMBIGUITY_GUARD_DEG:
                    break
                if (or1, oe1) == k1 or (or2, oe2) == k1 or (or1, oe1) == k2 or (or2, oe2) == k2:
                    ambiguous = True
                    break
            if ambiguous:
                continue
            continuations[k1] = (r2, e2)
            continuations[k2] = (r1, e1)
            used.add(k1)
            used.add(k2)

    # --- Step C: Chain into corridors ---
    visited: set[str] = set()
    corridors: list[RoadCorridor] = []

    for rid in nj_roads:
        if rid in visited:
            continue
        chain: list[str] = [rid]
        chain_junctions: list[str] = []
        visited.add(rid)

        # Walk backward from start
        current = rid
        walk_count = 0
        while walk_count < MAX_CHAIN_LENGTH:
            key = (current, "start")
            if key not in continuations:
                break
            next_rid, next_end = continuations[key]
            if next_rid in visited:
                break
            # Find the junction between current and next_rid
            for jid, touches in junction_to_roads.items():
                rids_in_junction = {t[0] for t in touches}
                if current in rids_in_junction and next_rid in rids_in_junction:
                    chain_junctions.insert(0, jid)
                    break
            chain.insert(0, next_rid)
            visited.add(next_rid)
            current = next_rid
            # Continue from the other end of next_rid
            if next_end == "end":
                walk_count += 1
                continue
            break

        # Walk forward from end
        current = rid
        walk_count = 0
        while walk_count < MAX_CHAIN_LENGTH:
            key = (current, "end")
            if key not in continuations:
                break
            next_rid, next_end = continuations[key]
            if next_rid in visited:
                break
            for jid, touches in junction_to_roads.items():
                rids_in_junction = {t[0] for t in touches}
                if current in rids_in_junction and next_rid in rids_in_junction:
                    chain_junctions.append(jid)
                    break
            chain.append(next_rid)
            visited.add(next_rid)
            current = next_rid
            if next_end == "start":
                walk_count += 1
                continue
            break

        # Build corridor metadata
        segment_lengths = [_as_float(nj_roads[r].get("length")) for r in chain]
        total_length = sum(segment_lengths)
        offsets: list[float] = []
        cumulative = 0.0
        for sl in segment_lengths:
            offsets.append(cumulative)
            cumulative += sl

        has_parking = any(_road_has_parking(nj_roads[r]) for r in chain)
        lane_config = _dominant_lane_config(nj_roads[chain[0]])

        # Bearing from start to end
        start_pos = _endpoint_position(nj_roads[chain[0]], "start")
        end_pos = _endpoint_position(nj_roads[chain[-1]], "end")
        if start_pos and end_pos and (abs(start_pos[0] - end_pos[0]) > 1 or abs(start_pos[1] - end_pos[1]) > 1):
            bearing = _bearing_between(start_pos, end_pos)
        else:
            hdg = _endpoint_heading(nj_roads[chain[0]], "start")
            bearing = (math.degrees(hdg) % 360) if hdg is not None else 0.0

        compass = _bearing_to_compass(bearing)
        parking_tag = ", parking" if has_parking else ""
        desc = f"{compass} {lane_config}, {total_length:.0f}m, {len(chain)} segment{'s' if len(chain) > 1 else ''}{parking_tag}"

        corridors.append(RoadCorridor(
            id=_stable_corridor_id(chain),
            road_ids=chain,
            junction_ids=chain_junctions,
            total_length=total_length,
            segment_lengths=segment_lengths,
            segment_offsets=offsets,
            has_parking=has_parking,
            lane_config=lane_config,
            bearing_deg=round(bearing, 1),
            description=desc,
        ))

    corridors.sort(key=lambda c: -c.total_length)

    multi = sum(1 for c in corridors if len(c.road_ids) > 1)
    logger.info(
        "Built %d corridors (%d multi-road, %d single) from %d non-junction roads",
        len(corridors), multi, len(corridors) - multi, len(nj_roads),
    )
    return corridors


def corridor_for_road(corridors: list[RoadCorridor], road_id: str) -> RoadCorridor | None:
    """Find the corridor containing a given road_id. O(n) — build a dict for O(1) if needed."""
    road_str = str(road_id)
    for corridor in corridors:
        if road_str in corridor.road_ids:
            return corridor
    return None


def build_corridor_lookup(corridors: list[RoadCorridor]) -> dict[str, RoadCorridor]:
    """Build road_id → corridor lookup dict for O(1) access."""
    lookup: dict[str, RoadCorridor] = {}
    for corridor in corridors:
        for road_id in corridor.road_ids:
            lookup[road_id] = corridor
    return lookup





def adjust_corridors_for_junction_gaps(
    corridors: list[RoadCorridor],
    runtime_segments: list[dict],
) -> list[RoadCorridor]:
    """Adjust corridor lengths and offsets to include physical junction gap distances.

    Called after runtime data is available. Uses centerline endpoints to measure
    the actual gap between consecutive roads through junctions.
    """
    import math as _math

    # Build segment lookup: road_id -> segment with lane closest to center
    seg_by_road: dict[str, dict] = {}
    for seg in runtime_segments:
        rid = str(seg.get("road_id", ""))
        lid = seg.get("lane_id", 0)
        if rid not in seg_by_road or abs(lid) < abs(seg_by_road[rid].get("lane_id", 999)):
            seg_by_road[rid] = seg

    adjusted: list[RoadCorridor] = []
    for corridor in corridors:
        if len(corridor.road_ids) < 2:
            adjusted.append(corridor)
            continue

        # Measure junction gaps from runtime centerlines
        junction_gaps: list[float] = []
        for i in range(len(corridor.road_ids) - 1):
            seg_a = seg_by_road.get(corridor.road_ids[i])
            seg_b = seg_by_road.get(corridor.road_ids[i + 1])
            if seg_a and seg_b and seg_a.get("centerline") and seg_b.get("centerline"):
                end_a = seg_a["centerline"][-1]
                start_b = seg_b["centerline"][0]
                gap = _math.sqrt(
                    (end_a["x"] - start_b["x"])**2 + (end_a["y"] - start_b["y"])**2
                )
                junction_gaps.append(gap)
            else:
                junction_gaps.append(0.0)

        # Rebuild offsets: road_length + junction_gap + road_length + junction_gap + ...
        new_offsets: list[float] = []
        new_total = 0.0
        for i, road_id in enumerate(corridor.road_ids):
            new_offsets.append(new_total)
            new_total += corridor.segment_lengths[i]
            if i < len(junction_gaps):
                new_total += junction_gaps[i]

        # Update corridor
        adjusted.append(RoadCorridor(
            id=corridor.id,
            road_ids=corridor.road_ids,
            junction_ids=corridor.junction_ids,
            total_length=round(new_total, 1),
            segment_lengths=corridor.segment_lengths,
            segment_offsets=[round(o, 1) for o in new_offsets],
            has_parking=corridor.has_parking,
            lane_config=corridor.lane_config,
            bearing_deg=corridor.bearing_deg,
            description=corridor.description.replace(
                f"{corridor.total_length:.0f}m",
                f"{new_total:.0f}m",
            ),
        ))

    return adjusted


def resolve_corridor_distance(corridor, distance_m: float) -> tuple[str, float]:
    """Resolve a distance along a corridor to (road_id, local_s_fraction).

    Handles junction dead zones: if distance falls between road segments
    (in a junction gap), snaps to the nearest road endpoint.
    """
    total = corridor.total_length if isinstance(corridor, RoadCorridor) else corridor.get("total_length", 0)
    seg_lengths = corridor.segment_lengths if isinstance(corridor, RoadCorridor) else corridor.get("segment_lengths", [])
    seg_offsets = corridor.segment_offsets if isinstance(corridor, RoadCorridor) else corridor.get("segment_offsets", [])
    road_ids = corridor.road_ids if isinstance(corridor, RoadCorridor) else corridor.get("road_ids", [])

    distance = max(BOUNDARY_BUFFER_M, min(distance_m, total - BOUNDARY_BUFFER_M))

    for i, (road_id, offset, length) in enumerate(zip(road_ids, seg_offsets, seg_lengths)):
        road_end = offset + length
        if distance <= road_end:
            # Distance falls within this road segment
            local_s = distance - offset
            local_s = max(BOUNDARY_BUFFER_M, min(local_s, length - BOUNDARY_BUFFER_M))
            s_fraction = local_s / length if length > 0 else 0.5
            return road_id, round(s_fraction, 4)

        # Check if distance falls in the junction gap after this road
        if i + 1 < len(road_ids):
            next_offset = seg_offsets[i + 1]
            if distance < next_offset:
                # In a junction gap — snap to nearest road endpoint
                dist_to_end = distance - road_end
                dist_to_next_start = next_offset - distance
                if dist_to_end <= dist_to_next_start:
                    # Closer to end of current road
                    return road_id, round(1.0 - BOUNDARY_BUFFER_M / length, 4) if length > 0 else 0.9
                else:
                    # Closer to start of next road
                    return road_ids[i + 1], round(BOUNDARY_BUFFER_M / seg_lengths[i + 1], 4) if seg_lengths[i + 1] > 0 else 0.1

    # Fallback: last road
    return road_ids[-1], 0.9


def parse_lane_spec(spec: str) -> tuple[str, str]:
    """Parse a lane spec like 'right parking' into (side, lane_type)."""
    parts = spec.strip().lower().split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        # Default: if only type given, assume right side
        return "right", parts[0]
    return "right", "driving"
