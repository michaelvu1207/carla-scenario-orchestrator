"""
GT sensor spawner and 2D bounding box extraction for CARLA simulations.

GT sensors are fixed-config cameras attached to the ego vehicle for ground-truth data.
Sensor types:
  - semantic_seg  -> sensor.camera.semantic_segmentation (PNG)
  - depth         -> sensor.camera.depth (PNG)
  - instance_seg  -> sensor.camera.instance_segmentation (PNG)

Bboxes are projected using the semantic_seg (or first GT) camera intrinsics + extrinsics.
"""
from __future__ import annotations

import json
import logging
import math
import queue
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GT_WIDTH = 1280
GT_HEIGHT = 720
GT_FOV = 90.0
# Chase-camera pose (same as topdown recording camera)
GT_X, GT_Y, GT_Z = -5.5, 0.0, 2.8
GT_PITCH, GT_YAW, GT_ROLL = -15.0, 0.0, 0.0

GT_SENSOR_BLUEPRINTS: dict[str, str] = {
    "semantic_seg": "sensor.camera.semantic_segmentation",
    "depth": "sensor.camera.depth",
    "instance_seg": "sensor.camera.instance_segmentation",
}


class GtSpawnedSensor:
    def __init__(self, sensor_type: str, actor: Any, frame_queue: queue.Queue, output_dir: Path):
        self.sensor_type = sensor_type
        self.actor = actor
        self.frame_queue = frame_queue
        self.output_dir = output_dir
        self.saved_frame_count = 0

    def destroy(self) -> None:
        try:
            if self.actor is not None and self.actor.is_alive:
                self.actor.stop()
                self.actor.destroy()
        except Exception as e:
            logger.warning(f"Error destroying GT sensor {self.sensor_type}: {e}")
        self.actor = None


def spawn_gt_sensors(
    world: Any,
    blueprint_library: Any,
    gt_sensor_names: list[str],
    ego_vehicle: Any,
    gt_dir: Path,
) -> list[GtSpawnedSensor]:
    """Spawn GT cameras attached to ego vehicle."""
    if not gt_sensor_names or ego_vehicle is None:
        return []

    import carla

    transform = carla.Transform(
        carla.Location(x=GT_X, y=GT_Y, z=GT_Z),
        carla.Rotation(pitch=GT_PITCH, yaw=GT_YAW, roll=GT_ROLL),
    )

    spawned: list[GtSpawnedSensor] = []
    for sensor_type in gt_sensor_names:
        bp_name = GT_SENSOR_BLUEPRINTS.get(sensor_type)
        if bp_name is None:
            logger.warning(f"Unknown GT sensor type: {sensor_type!r}, skipping")
            continue
        try:
            bp = blueprint_library.find(bp_name)
            bp.set_attribute("image_size_x", str(GT_WIDTH))
            bp.set_attribute("image_size_y", str(GT_HEIGHT))
            bp.set_attribute("fov", str(GT_FOV))

            actor = world.spawn_actor(
                bp, transform, attach_to=ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            out_dir = gt_dir / sensor_type
            out_dir.mkdir(parents=True, exist_ok=True)

            fq: queue.Queue = queue.Queue()
            actor.listen(fq.put)
            spawned.append(GtSpawnedSensor(sensor_type, actor, fq, out_dir))
            logger.info(f"Spawned GT sensor: {sensor_type}")
        except Exception as e:
            logger.error(f"Failed to spawn GT sensor {sensor_type}: {e}")

    return spawned


def collect_gt_frames(
    gt_sensors: list[GtSpawnedSensor],
    frame_num: int,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Drain one frame from each GT sensor queue and save as PNG. Returns {type: data}."""
    collected: dict[str, Any] = {}
    for sensor in gt_sensors:
        try:
            data = sensor.frame_queue.get(timeout=timeout)
            dest = sensor.output_dir / f"frame_{data.frame:06d}.png"
            data.save_to_disk(str(dest))
            sensor.saved_frame_count += 1
            collected[sensor.sensor_type] = data
        except queue.Empty:
            logger.debug(f"GT sensor {sensor.sensor_type} timeout at frame {frame_num}")
    return collected


def destroy_gt_sensors(gt_sensors: list[GtSpawnedSensor]) -> None:
    for sensor in gt_sensors:
        sensor.destroy()
    if gt_sensors:
        logger.info(f"Destroyed {len(gt_sensors)} GT sensors")


def _camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array(
        [[focal, 0, width / 2.0],
         [0, focal, height / 2.0],
         [0, 0, 1.0]],
        dtype=np.float64,
    )


_CARLA_TO_CAM = np.array(
    [[0, 1, 0, 0],
     [0, 0, -1, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.float64,
)


def extract_bboxes(
    world_actors: list[Any],
    camera_sensor: Any,
    frame_num: int,
    bbox_dir: Path,
    width: int = GT_WIDTH,
    height: int = GT_HEIGHT,
    fov: float = GT_FOV,
) -> list[dict]:
    """
    Project 3D bounding boxes of all alive world actors onto the GT camera image plane.
    Writes per-frame JSON in an OrchestratorFrameGT-compatible shape.
    """
    bbox_dir.mkdir(parents=True, exist_ok=True)

    K = _camera_intrinsic(width, height, fov)
    world_to_cam = np.array(camera_sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
    extrinsic = _CARLA_TO_CAM @ world_to_cam
    camera_loc = camera_sensor.get_transform().location

    detections: list[dict[str, Any]] = []
    for actor in world_actors:
        try:
            if not actor.is_alive:
                continue
            bb = actor.bounding_box
            verts = bb.get_world_vertices(actor.get_transform())
        except Exception:
            continue

        pts_2d = []
        for vertex in verts:
            p_world = np.array([vertex.x, vertex.y, vertex.z, 1.0], dtype=np.float64)
            p_cam = extrinsic @ p_world
            if p_cam[2] <= 0:
                continue
            u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            v_c = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
            pts_2d.append((float(u), float(v_c)))

        if len(pts_2d) < 2:
            continue

        xs = [point[0] for point in pts_2d]
        ys = [point[1] for point in pts_2d]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(width), max(xs))
        y2 = min(float(height), max(ys))

        if x2 <= x1 or y2 <= y1:
            continue

        actor_loc = actor.get_transform().location
        dx = float(actor_loc.x) - float(camera_loc.x)
        dy = float(actor_loc.y) - float(camera_loc.y)
        dz = float(actor_loc.z) - float(camera_loc.z)
        distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)

        detections.append({
            "actor_id": str(int(actor.id)),
            "blueprint": str(actor.type_id),
            "bbox_2d": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
            "distance_m": round(distance_m, 3),
        })

    payload = {
        "frame_id": f"{frame_num:06d}",
        "timestamp_s": float(frame_num),
        "width": width,
        "height": height,
        "detections": detections,
    }
    bbox_path = bbox_dir / f"frame_{frame_num:06d}.json"
    bbox_path.write_text(json.dumps(payload), encoding="utf-8")
    return detections
