"""
LiDAR Bird's Eye View (BEV) renderer.

Converts PLY point cloud files into top-down BEV images for video encoding.
Each point is colored by height (z-axis) with intensity modulating brightness.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# BEV rendering parameters
BEV_RANGE_M = 80.0       # meters from center in each direction
BEV_RESOLUTION = 512      # output image size (pixels)
BEV_HEIGHT_MIN = -2.0     # meters (ground level)
BEV_HEIGHT_MAX = 8.0      # meters (above vehicle)

# Height-based colormap (blue=low, green=mid, red=high, white=very high)
def _height_to_rgb(z_norm: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Map normalized height [0,1] and intensity [0,1] to RGB."""
    r = np.clip(z_norm * 2.0, 0, 1)
    g = np.clip(1.0 - np.abs(z_norm - 0.5) * 2.0, 0, 1)
    b = np.clip(1.0 - z_norm * 2.0, 0, 1)
    # Intensity modulates brightness
    brightness = 0.3 + 0.7 * intensity
    rgb = np.stack([r * brightness, g * brightness, b * brightness], axis=-1)
    return (rgb * 255).astype(np.uint8)


def render_ply_to_bev(ply_path: Path, output_path: Path,
                      range_m: float = BEV_RANGE_M,
                      resolution: int = BEV_RESOLUTION) -> bool:
    """Render a single PLY file as a BEV image.

    Returns True on success, False on failure.
    """
    try:
        # Parse ASCII PLY
        with open(ply_path, "r") as f:
            # Skip header
            vertex_count = 0
            for line in f:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line.strip() == "end_header":
                    break

            if vertex_count == 0:
                return False

            # Read points (x, y, z, intensity)
            points = np.loadtxt(f, max_rows=vertex_count, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] < 3:
            return False

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] > 3 else np.ones_like(z)

        # Filter to BEV range
        mask = (np.abs(x) < range_m) & (np.abs(y) < range_m)
        x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

        if len(x) == 0:
            # No points in range — create blank image
            img = Image.new("RGB", (resolution, resolution), (0, 0, 0))
            img.save(str(output_path))
            return True

        # Map to pixel coordinates
        px = ((x + range_m) / (2 * range_m) * (resolution - 1)).astype(int)
        py = ((y + range_m) / (2 * range_m) * (resolution - 1)).astype(int)
        px = np.clip(px, 0, resolution - 1)
        py = np.clip(py, 0, resolution - 1)

        # Normalize height and intensity
        z_norm = np.clip((z - BEV_HEIGHT_MIN) / (BEV_HEIGHT_MAX - BEV_HEIGHT_MIN), 0, 1)
        i_norm = np.clip(intensity, 0, 1)

        # Create BEV image (dark background)
        img_array = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        colors = _height_to_rgb(z_norm, i_norm)

        # Paint points (later points overwrite — highest z wins due to sorting)
        order = np.argsort(z)
        img_array[py[order], px[order]] = colors[order]

        img = Image.fromarray(img_array)
        img.save(str(output_path))
        return True

    except Exception as e:
        logger.warning(f"Failed to render BEV for {ply_path}: {e}")
        return False


def render_lidar_frames(sensor_dir: Path, output_dir: Path | None = None,
                        range_m: float = BEV_RANGE_M,
                        resolution: int = BEV_RESOLUTION) -> int:
    """Render all PLY files in a sensor directory to BEV PNGs.

    If output_dir is None, PNGs are written next to PLY files as {frame}.jpg
    (to match the camera frame naming convention for ffmpeg encoding).

    Returns the number of frames rendered.
    """
    ply_files = sorted(sensor_dir.glob("*.ply"))
    if not ply_files:
        return 0

    target_dir = output_dir or sensor_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for ply_file in ply_files:
        # Use .jpg extension to match camera frame convention
        out_name = ply_file.stem + ".jpg"
        out_path = target_dir / out_name
        if render_ply_to_bev(ply_file, out_path, range_m, resolution):
            rendered += 1

    logger.info(f"Rendered {rendered}/{len(ply_files)} LiDAR BEV frames in {sensor_dir.name}")
    return rendered
