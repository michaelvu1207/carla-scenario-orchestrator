"""
Per-sensor video encoder.

After simulation completes, encodes each camera sensor's JPEG frames
into MP4 videos using ffmpeg. LiDAR/radar data is left as-is (raw files).
"""
from __future__ import annotations

from typing import Callable

import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def encode_sensor_video(
    sensor_dir: Path,
    fps: int,
    width: int,
    height: int,
    output_filename: str = "recording.mp4",
) -> Path | None:
    """
    Encode JPEG frames in sensor_dir into an MP4 video.

    Returns the output path on success, None on failure.
    """
    output_path = sensor_dir / output_filename

    # Check if there are frames to encode
    frame_files = sorted(sensor_dir.glob("*.jpg"))
    frame_count = len(frame_files)
    if frame_count == 0:
        logger.warning(f"No frames to encode in {sensor_dir}")
        return None

    # CARLA uses global frame numbers (e.g., 920707.jpg), not sequential.
    # Use glob pattern for ffmpeg input so it works with any filenames.
    glob_pattern = str(sensor_dir / "*.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", glob_pattern,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg failed for {sensor_dir}: {result.stderr[-500:]}")
            return None

        logger.info(f"Encoded {frame_count} frames -> {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out for {sensor_dir}")
        return None
    except Exception as e:
        logger.error(f"Encoding failed for {sensor_dir}: {e}")
        return None


def encode_all_sensors(
    spawned_sensors: list,
    fps: int = 30,
    max_workers: int = 4,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> dict[str, Path | None]:
    """
    Encode all camera sensors in parallel.

    Returns a dict mapping sensor_id -> output_path (or None if failed).
    Non-camera sensors are skipped (their data is already saved as raw files).
    """
    results: dict[str, Path | None] = {}

    camera_sensors = [s for s in spawned_sensors if s.config.sensor_category == "camera"]
    other_sensors = [s for s in spawned_sensors if s.config.sensor_category != "camera"]

    # Render LiDAR sensors as BEV videos, skip other non-camera sensors
    from .lidar_renderer import render_lidar_frames
    lidar_sensors = [s for s in other_sensors if s.config.sensor_category == "lidar"]
    skip_sensors = [s for s in other_sensors if s.config.sensor_category != "lidar"]

    for sensor in skip_sensors:
        results[sensor.config.id] = sensor.output_dir
        sensor.status = "complete"
        logger.info(f"Sensor {sensor.config.label} ({sensor.config.sensor_category}) - raw data saved, no encoding needed")

    # Render LiDAR PLY files to BEV JPEGs, then encode as video
    for sensor in lidar_sensors:
        rendered = render_lidar_frames(sensor.output_dir)
        if rendered > 0:
            logger.info(f"Rendered {rendered} BEV frames for {sensor.config.label}, queuing for video encoding")
            # Move to camera_sensors list so it gets encoded to MP4
            camera_sensors.append(sensor)
            # Use BEV resolution for encoding (512x512)
            sensor._bev_width = 512
            sensor._bev_height = 512
        else:
            results[sensor.config.id] = sensor.output_dir
            sensor.status = "complete"
            logger.info(f"Sensor {sensor.config.label} - no LiDAR frames to render")

    if not camera_sensors:
        return results

    logger.info(f"Encoding {len(camera_sensors)} camera sensors in parallel (max {max_workers} workers)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sensor in camera_sensors:
            sensor.status = "encoding"
            # Use BEV dimensions for LiDAR sensors (rendered as 512x512 images)
            width = getattr(sensor, '_bev_width', None) or sensor.config.width
            height = getattr(sensor, '_bev_height', None) or sensor.config.height
            future = executor.submit(
                encode_sensor_video,
                sensor.output_dir,
                fps,
                width,
                height,
            )
            futures[future] = sensor

        completed_count = 0
        for future in as_completed(futures):
            sensor = futures[future]
            completed_count += 1
            try:
                output_path = future.result()
                if output_path:
                    results[sensor.config.id] = output_path
                    sensor.status = "complete"
                else:
                    results[sensor.config.id] = None
                    sensor.status = "failed"
            except Exception as e:
                logger.error(f"Encoding error for {sensor.config.label}: {e}")
                results[sensor.config.id] = None
                sensor.status = "failed"
            if on_progress:
                try:
                    on_progress(completed_count, len(camera_sensors))
                except Exception:
                    pass

    complete_count = sum(1 for v in results.values() if v is not None)
    logger.info(f"Encoding complete: {complete_count}/{len(results)} sensors succeeded")
    return results
