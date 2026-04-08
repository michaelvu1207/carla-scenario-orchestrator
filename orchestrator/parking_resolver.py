"""Server-side parking position resolver.
Computes world coordinates for parked cars by offsetting from the driving lane centerline.

Algorithm (matches editor's buildParkingPlacementGuides):
1. Find the outermost driving lane centerline for the target road
2. At the target s_fraction, get the centerline point (x, y, yaw)
3. Compute perpendicular normal (yaw + 90 for right, yaw - 90 for left)
4. Offset by 3.075m (half driving width + half parking width)
5. Return world coordinates (x, y, z, yaw)
"""
import math
from typing import Any


# Lane width constants (matching editor road-utils.ts)
DRIVING_WIDTH = 3.25
PARKING_WIDTH = 2.9
PARKING_OFFSET = (DRIVING_WIDTH + PARKING_WIDTH) / 2  # = 3.075m


def compute_parking_position(
    runtime_segments: list[dict[str, Any]],
    road_id: str,
    s_fraction: float,
    side: str = "right",
    parking_index: int = 1,
) -> dict[str, float] | None:
    """Compute world coordinates for a parked car position.

    Args:
        runtime_segments: All runtime road segments from CARLA
        road_id: The road to park on
        s_fraction: Position along the road (0.0-1.0)
        side: "right" or "left"
        parking_index: Which parking lane (1 = closest to driving, 2 = further out)

    Returns:
        {x, y, z, yaw} world coordinates, or None if road not found
    """
    # Find the outermost driving lane on the requested side
    road_segs = [
        s for s in runtime_segments
        if str(s.get("road_id", "")) == str(road_id)
        and not s.get("is_junction", False)
    ]
    if not road_segs:
        return None

    # Find the outermost driving lane on the requested side
    if side == "right":
        # Negative lane_ids, find the one with largest absolute value (outermost)
        driving_segs = [s for s in road_segs if s.get("lane_id", 0) < 0]
        if not driving_segs:
            return None
        outermost = min(driving_segs, key=lambda s: s["lane_id"])  # most negative = outermost right
    else:
        # Positive lane_ids, find the one with largest value (outermost)
        driving_segs = [s for s in road_segs if s.get("lane_id", 0) > 0]
        if not driving_segs:
            return None
        outermost = max(driving_segs, key=lambda s: s["lane_id"])  # most positive = outermost left

    centerline = outermost.get("centerline", [])
    if not centerline or len(centerline) < 2:
        return None

    # Get the point at s_fraction
    idx = min(int(s_fraction * (len(centerline) - 1)), len(centerline) - 1)
    idx = max(0, idx)
    pt = centerline[idx]

    # Get heading (yaw) at this point
    # Use the yaw from the centerline point, or compute from adjacent points
    yaw_rad = math.radians(pt.get("yaw", 0))

    # Compute perpendicular offset direction
    # In CARLA coordinates: x=east, y=south (y is flipped from standard math)
    # The centerline yaw points along the road direction
    # Right perpendicular: yaw + 90 degrees
    # Left perpendicular: yaw - 90 degrees
    offset_distance = PARKING_OFFSET + (parking_index - 1) * PARKING_WIDTH

    if side == "right":
        # Right side: perpendicular to the right of travel direction
        offset_angle = yaw_rad + math.pi / 2
    else:
        # Left side: perpendicular to the left of travel direction
        offset_angle = yaw_rad - math.pi / 2

    # Apply offset
    park_x = pt["x"] + offset_distance * math.cos(offset_angle)
    park_y = pt["y"] + offset_distance * math.sin(offset_angle)
    park_z = pt.get("z", 0) + 0.1  # Slight Z offset to avoid ground clipping

    # Parked car faces along the road (same yaw as driving lane)
    park_yaw = pt.get("yaw", 0)

    return {
        "x": round(park_x, 3),
        "y": round(park_y, 3),
        "z": round(park_z, 3),
        "yaw": round(park_yaw, 2),
    }


def resolve_parking_placement(
    runtime_segments: list[dict[str, Any]],
    road_id: str,
    s_fraction: float,
    lane_spec: str = "right parking",
) -> dict[str, Any] | None:
    """High-level: resolve abstract 'right parking' to world coordinates.

    Returns dict with spawn_point for CARLA and anchor info for the editor.
    """
    parts = lane_spec.strip().lower().split()
    side = parts[0] if parts else "right"

    parking_pos = compute_parking_position(
        runtime_segments, road_id, s_fraction, side=side,
    )
    if not parking_pos:
        return None

    return {
        "spawn_point": parking_pos,
        "placement_mode": "point",
        "road_id": road_id,
        "s_fraction": s_fraction,
        "side": side,
    }
