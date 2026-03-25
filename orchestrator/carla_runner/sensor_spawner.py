"""
Multi-sensor spawner for CARLA simulations.

Spawns cameras, LiDAR, and radar sensors based on SensorConfig list
from the frontend. Each sensor gets its own frame queue and output directory.
"""
from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any

from .models import SensorConfig, SensorPose

logger = logging.getLogger(__name__)

ATTACHMENT_TYPE_MAP = {
    "rigid": "Rigid",
    "spring_arm": "SpringArm",
    "spring_arm_ghost": "SpringArmGhost",
}

SENSOR_BLUEPRINT_MAP = {
    ("camera", "rgb"): "sensor.camera.rgb",
    ("camera", "depth"): "sensor.camera.depth",
    ("camera", "semantic_segmentation"): "sensor.camera.semantic_segmentation",
    ("camera", "instance_segmentation"): "sensor.camera.instance_segmentation",
    ("lidar", "point_cloud"): "sensor.lidar.ray_cast",
    ("lidar", "semantic_point_cloud"): "sensor.lidar.ray_cast_semantic",
    ("radar", "radar_data"): "sensor.other.radar",
    ("imu", "imu_data"): "sensor.other.imu",
    ("gnss", "gnss_fix"): "sensor.other.gnss",
}


class SpawnedSensor:
    def __init__(self, config, actor, frame_queue, output_dir):
        self.config = config
        self.actor = actor
        self.frame_queue = frame_queue
        self.output_dir = output_dir
        self.saved_frame_count = 0
        self.timeout_count = 0
        self.status = "recording"

    def destroy(self):
        try:
            if self.actor is not None and self.actor.is_alive:
                self.actor.stop()
                self.actor.destroy()
        except Exception as e:
            logger.warning(f"Error destroying sensor {self.config.label}: {e}")
        self.actor = None


def _require_carla():
    import carla
    return carla


def _build_transform(pose):
    carla = _require_carla()
    return carla.Transform(
        carla.Location(x=pose.x, y=pose.y, z=pose.z),
        carla.Rotation(pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll),
    )


def _get_attachment_type(attachment_str):
    carla = _require_carla()
    type_name = ATTACHMENT_TYPE_MAP.get(attachment_str, "Rigid")
    return getattr(carla.AttachmentType, type_name)


def spawn_sensors(world, blueprint_library, sensor_configs, actor_map, ego_vehicle, job_dir):
    spawned = []
    for config in sensor_configs:
        sensor_dir = job_dir / "sensors" / config.id
        sensor_dir.mkdir(parents=True, exist_ok=True)
        try:
            blueprint_key = (config.sensor_category, config.output_modality)
            bp_name = SENSOR_BLUEPRINT_MAP.get(blueprint_key)
            if bp_name is None:
                logger.warning(f"Unknown sensor type: {blueprint_key}, skipping {config.label}")
                continue

            bp = blueprint_library.find(bp_name)

            if config.sensor_category == "camera":
                bp.set_attribute("image_size_x", str(config.width))
                bp.set_attribute("image_size_y", str(config.height))
                bp.set_attribute("fov", str(config.fov))
                if config.update_rate > 0:
                    bp.set_attribute("sensor_tick", str(1.0 / config.update_rate))
            elif config.sensor_category == "lidar":
                bp.set_attribute("channels", str(config.channels))
                bp.set_attribute("range", str(config.range_m))
                bp.set_attribute("points_per_second", str(config.points_per_second))
                bp.set_attribute("rotation_frequency", str(config.rotation_frequency))
            elif config.sensor_category == "radar":
                bp.set_attribute("horizontal_fov", str(config.horizontal_fov))
                bp.set_attribute("vertical_fov", str(config.vertical_fov))
                bp.set_attribute("range", str(config.radar_range))

            transform = _build_transform(config.pose)
            attachment_type = _get_attachment_type(config.attachment_type)

            if config.attach_to == "world" and config.world_position:
                carla = _require_carla()
                wp = config.world_position
                wr = config.world_rotation
                world_transform = carla.Transform(
                    carla.Location(x=wp.x, y=wp.y, z=getattr(wr, "z", 8.0) if wr else 8.0),
                    carla.Rotation(
                        pitch=getattr(wr, "pitch", -15.0) if wr else -15.0,
                        yaw=getattr(wr, "yaw", 0.0) if wr else 0.0,
                        roll=getattr(wr, "roll", 0.0) if wr else 0.0,
                    ),
                )
                sensor_actor = world.spawn_actor(bp, world_transform)
            elif config.attach_to == "ego" and ego_vehicle:
                sensor_actor = world.spawn_actor(bp, transform, attach_to=ego_vehicle, attachment_type=attachment_type)
            elif config.attach_to in actor_map:
                parent = actor_map[config.attach_to]
                sensor_actor = world.spawn_actor(bp, transform, attach_to=parent, attachment_type=attachment_type)
            else:
                logger.warning(f"Cannot find parent for sensor {config.label}, skipping")
                continue

            frame_queue_obj = queue.Queue()
            sensor_actor.listen(frame_queue_obj.put)
            spawned.append(SpawnedSensor(config, sensor_actor, frame_queue_obj, sensor_dir))
            logger.info(f"Spawned sensor: {config.label} ({config.sensor_category}/{config.output_modality})")

        except Exception as e:
            logger.error(f"Failed to spawn sensor {config.label}: {e}")
            continue

    logger.info(f"Spawned {len(spawned)}/{len(sensor_configs)} sensors")
    return spawned


def _save_sensor_frame(sensor, data):
    """Save a single sensor frame to disk (runs in thread pool)."""
    try:
        if sensor.config.sensor_category == "camera":
            dest = sensor.output_dir / f"{data.frame:06d}.jpg"
            data.save_to_disk(str(dest))
        elif sensor.config.sensor_category == "lidar":
            dest = sensor.output_dir / f"{data.frame:06d}.ply"
            data.save_to_disk(str(dest))
        elif sensor.config.sensor_category == "radar":
            import csv
            dest = sensor.output_dir / f"{data.frame:06d}.csv"
            with open(dest, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["velocity", "azimuth", "altitude", "depth"])
                for detection in data:
                    writer.writerow([detection.velocity, detection.azimuth, detection.altitude, detection.depth])
        sensor.saved_frame_count += 1
    except Exception as e:
        logger.warning(f"Failed to save frame for {sensor.config.label}: {e}")
        sensor.timeout_count += 1


# Shared thread pool for parallel frame saves (created once, reused)
_frame_save_pool = None


def _get_frame_save_pool(max_workers=8):
    global _frame_save_pool
    if _frame_save_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _frame_save_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _frame_save_pool


def collect_sensor_frames(spawned_sensors, timeout=2.0):
    """Collect frames from all sensors in parallel.

    Phase 1: Drain all sensor queues concurrently (non-blocking after initial wait).
    Phase 2: Save all frames to disk in parallel via thread pool.
    """
    active = [s for s in spawned_sensors if s.status == "recording"]
    if not active:
        return

    # Phase 1: Collect frames from queues (queues are already populated by CARLA callbacks)
    frames = []
    for sensor in active:
        try:
            data = sensor.frame_queue.get(timeout=timeout)
            frames.append((sensor, data))
        except queue.Empty:
            sensor.timeout_count += 1

    # Phase 2: Save all collected frames in parallel
    if len(frames) <= 1:
        for sensor, data in frames:
            _save_sensor_frame(sensor, data)
    else:
        pool = _get_frame_save_pool()
        futures = [pool.submit(_save_sensor_frame, sensor, data) for sensor, data in frames]
        for future in futures:
            future.result()  # Wait for all saves to complete


def destroy_all_sensors(spawned_sensors):
    for sensor in spawned_sensors:
        sensor.destroy()
    logger.info(f"Destroyed {len(spawned_sensors)} sensors")
