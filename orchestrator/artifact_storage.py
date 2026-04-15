from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .models import JobRecord, StoredArtifact

GT_SENSOR_LABELS = {
    "semantic_seg": "GT Semantic",
    "depth": "GT Depth",
    "instance_seg": "GT Instance",
}
GT_MODALITY_MAP = {
    "semantic_seg": "semantic_segmentation",
    "depth": "depth",
    "instance_seg": "instance_segmentation",
}
GT_WIDTH = 1280
GT_HEIGHT = 720
GT_FOV = 90.0
GT_POSE = {
    "x": -5.5,
    "y": 0.0,
    "z": 2.8,
    "roll": 0.0,
    "pitch": -15.0,
    "yaw": 0.0,
}


def _safe_segment(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    cleaned = cleaned.strip("-._")
    return cleaned or "artifact"


def _file_ext(path: Path) -> str | None:
    suffix = path.suffix.strip().lower()
    if not suffix:
        return None
    return suffix.lstrip(".") or None


def _checksum_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_type_for_path(path: Path, default: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or default


def _frame_index_from_path(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _camera_intrinsics(width: int, height: int, fov: float) -> dict[str, Any]:
    import math

    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return {
        "width": width,
        "height": height,
        "fov": fov,
        "fx": focal,
        "fy": focal,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }


def _load_request_payload(job: JobRecord) -> dict[str, Any]:
    request_file = Path(job.artifacts.request_file)
    if not request_file.is_file():
        return {}
    try:
        return json.loads(request_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ArtifactStorage(Protocol):
    def upload_job_artifacts(self, job: JobRecord) -> list[StoredArtifact]: ...


class NullArtifactStorage:
    def upload_job_artifacts(self, job: JobRecord) -> list[StoredArtifact]:
        return []


class S3ArtifactStorage:
    def __init__(self, settings: Settings) -> None:
        if not settings.storage_bucket:
            raise RuntimeError("ORCH_STORAGE_BUCKET must be configured to enable S3 artifact uploads.")
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError("boto3 must be installed to enable S3 artifact uploads.") from exc
        self.settings = settings
        self.bucket = settings.storage_bucket
        self.client = boto3.client("s3", region_name=settings.storage_region)

    def upload_job_artifacts(self, job: JobRecord) -> list[StoredArtifact]:
        prefix = self._key_prefix(job)
        uploaded: list[StoredArtifact] = []
        run_dir = Path(job.artifacts.output_dir) / (job.run_id or "")
        if not run_dir.is_dir():
            return uploaded

        request_payload = _load_request_payload(job)
        sensor_configs = {
            str(sensor.get("id")): sensor
            for sensor in request_payload.get("sensors", [])
            if isinstance(sensor, dict) and sensor.get("id")
        }
        gt_sensor_names = [str(name) for name in request_payload.get("gt_sensors", []) if isinstance(name, str)]

        self._ensure_calibration_files(run_dir, sensor_configs, gt_sensor_names)
        self._ensure_gt_index_file(run_dir, job, prefix)

        def upload_file(
            *,
            path: Path,
            key: str,
            kind: str,
            label: str | None = None,
            artifact_class: str | None = None,
            sensor_id: str | None = None,
            sensor_label: str | None = None,
            sensor_category: str | None = None,
            output_modality: str | None = None,
            artifact_format: str | None = None,
            frame_index: int | None = None,
            sequence_id: str | None = None,
            is_raw: bool | None = None,
            metadata: dict[str, Any] | None = None,
            content_type: str | None = None,
        ) -> StoredArtifact:
            final_content_type = content_type or _content_type_for_path(path)
            self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": final_content_type})
            return StoredArtifact(
                kind=kind,
                label=label or path.name,
                local_path=str(path),
                content_type=final_content_type,
                file_ext=_file_ext(path),
                size_bytes=path.stat().st_size,
                checksum_sha256=_checksum_sha256(path),
                s3_bucket=self.bucket,
                s3_key=key,
                s3_uri=f"s3://{self.bucket}/{key}",
                artifact_class=artifact_class,
                sensor_id=sensor_id,
                sensor_label=sensor_label,
                sensor_category=sensor_category,
                output_modality=output_modality,
                artifact_format=artifact_format or _file_ext(path),
                frame_index=frame_index,
                sequence_id=sequence_id,
                is_raw=is_raw,
                metadata=metadata,
            )

        base_specs = [
            {
                "path": Path(job.artifacts.manifest_path) if job.artifacts.manifest_path else None,
                "key": f"{prefix}/manifest/run-manifest.json",
                "kind": "MANIFEST",
                "artifact_class": "metadata",
                "output_modality": "manifest",
                "artifact_format": "json",
                "is_raw": True,
                "content_type": "application/json",
            },
            {
                "path": Path(job.artifacts.request_file),
                "key": f"{prefix}/metadata/request.json",
                "kind": "REQUEST",
                "artifact_class": "metadata",
                "output_modality": "request",
                "artifact_format": "json",
                "is_raw": True,
                "content_type": "application/json",
            },
            {
                "path": Path(job.artifacts.runtime_settings_file),
                "key": f"{prefix}/metadata/runtime_settings.json",
                "kind": "RUNTIME_SETTINGS",
                "artifact_class": "metadata",
                "output_modality": "runtime_settings",
                "artifact_format": "json",
                "is_raw": True,
                "content_type": "application/json",
            },
            {
                "path": Path(job.artifacts.recording_path) if job.artifacts.recording_path else None,
                "key": f"{prefix}/recording/recording.mp4",
                "kind": "recording",
                "artifact_class": "recording",
                "output_modality": "rgb",
                "artifact_format": "mp4",
                "is_raw": False,
                "content_type": "video/mp4",
            },
            {
                "path": Path(job.artifacts.scenario_log_path) if job.artifacts.scenario_log_path else None,
                "key": f"{prefix}/logs/scenario.log",
                "kind": "SCENARIO_LOG",
                "artifact_class": "log",
                "output_modality": "scenario_log",
                "artifact_format": "log",
                "is_raw": True,
                "content_type": "text/plain; charset=utf-8",
            },
            {
                "path": Path(job.artifacts.debug_log_path) if job.artifacts.debug_log_path else None,
                "key": f"{prefix}/logs/run.log",
                "kind": "RUN_LOG",
                "artifact_class": "log",
                "output_modality": "run_log",
                "artifact_format": "log",
                "is_raw": True,
                "content_type": "text/plain; charset=utf-8",
            },
        ]

        for spec in base_specs:
            path = spec["path"]
            if not path or not path.is_file():
                continue
            uploaded.append(
                upload_file(
                    path=path,
                    key=spec["key"],
                    kind=spec["kind"],
                    artifact_class=spec["artifact_class"],
                    output_modality=spec["output_modality"],
                    artifact_format=spec["artifact_format"],
                    is_raw=spec["is_raw"],
                    content_type=spec["content_type"],
                )
            )

        calibration_dir = run_dir / "calibration"
        if calibration_dir.is_dir():
            for calibration_file in sorted(calibration_dir.rglob("*.json")):
                sensor_id = calibration_file.parent.name
                sensor_cfg = sensor_configs.get(sensor_id)
                uploaded.append(
                    upload_file(
                        path=calibration_file,
                        key=f"{prefix}/calibration/{sensor_id}/{calibration_file.name}",
                        kind="calibration",
                        artifact_class="calibration",
                        sensor_id=sensor_id,
                        sensor_label=(sensor_cfg or {}).get("label") if sensor_cfg else sensor_id,
                        sensor_category=(sensor_cfg or {}).get("sensor_category")
                        if sensor_cfg
                        else ("camera" if sensor_id.startswith("gt-") else None),
                        output_modality=(sensor_cfg or {}).get("output_modality") if sensor_cfg else None,
                        artifact_format="json",
                        sequence_id=sensor_id,
                        is_raw=True,
                        content_type="application/json",
                    )
                )

        sensors_dir = run_dir / "sensors"
        if sensors_dir.is_dir():
            for sensor_dir in sorted(sensors_dir.iterdir()):
                if not sensor_dir.is_dir():
                    continue
                sensor_id = sensor_dir.name
                sensor_cfg = sensor_configs.get(sensor_id, {})
                sensor_label = sensor_cfg.get("label") or sensor_id
                sensor_category = sensor_cfg.get("sensor_category")
                output_modality = sensor_cfg.get("output_modality")
                for file_path in sorted(sensor_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    frame_index = _frame_index_from_path(file_path)
                    suffix = _file_ext(file_path)
                    if file_path.name == "recording.mp4":
                        uploaded.append(
                            upload_file(
                                path=file_path,
                                key=f"{prefix}/sensors/{sensor_id}/recording/recording.mp4",
                                kind="recording",
                                artifact_class="recording",
                                sensor_id=sensor_id,
                                sensor_label=sensor_label,
                                sensor_category=sensor_category,
                                output_modality=output_modality,
                                artifact_format="mp4",
                                sequence_id=sensor_id,
                                is_raw=False,
                                content_type="video/mp4",
                            )
                        )
                        continue

                    modality_segment = output_modality or (suffix or "artifact")
                    uploaded.append(
                        upload_file(
                            path=file_path,
                            key=f"{prefix}/sensors/{sensor_id}/{modality_segment}/{file_path.name}",
                            kind=output_modality or suffix or "artifact",
                            artifact_class="raw_sensor_output",
                            sensor_id=sensor_id,
                            sensor_label=sensor_label,
                            sensor_category=sensor_category,
                            output_modality=output_modality,
                            artifact_format=suffix,
                            frame_index=frame_index,
                            sequence_id=sensor_id,
                            is_raw=True,
                        )
                    )

        gt_dir = run_dir / "gt"
        if gt_dir.is_dir():
            for gt_subdir in sorted(gt_dir.iterdir()):
                if not gt_subdir.is_dir():
                    continue
                gt_name = gt_subdir.name
                if gt_name == "scene":
                    for file_path in sorted(gt_subdir.glob("*.json")):
                        uploaded.append(
                            upload_file(
                                path=file_path,
                                key=f"{prefix}/gt/scene/{file_path.name}",
                                kind="gt_index",
                                artifact_class="ground_truth_index",
                                output_modality="gt_index",
                                artifact_format="json",
                                sequence_id="gt-scene",
                                is_raw=True,
                                content_type="application/json",
                            )
                        )
                    continue
                sensor_id = f"gt-{gt_name}"
                sensor_label = GT_SENSOR_LABELS.get(gt_name, sensor_id)
                output_modality = GT_MODALITY_MAP.get(gt_name, gt_name)
                for file_path in sorted(gt_subdir.iterdir()):
                    if not file_path.is_file():
                        continue
                    frame_index = _frame_index_from_path(file_path)
                    if gt_name == "bbox":
                        key = f"{prefix}/gt/bbox/{file_path.name}"
                        kind = "gt_bbox"
                        modality = "bbox_2d"
                    else:
                        key = f"{prefix}/gt/{gt_name}/{file_path.name}"
                        kind = f"gt_{gt_name}"
                        modality = output_modality
                    uploaded.append(
                        upload_file(
                            path=file_path,
                            key=key,
                            kind=kind,
                            artifact_class="ground_truth",
                            sensor_id=sensor_id,
                            sensor_label=sensor_label,
                            sensor_category="camera",
                            output_modality=modality,
                            artifact_format=_file_ext(file_path),
                            frame_index=frame_index,
                            sequence_id=sensor_id,
                            is_raw=True,
                        )
                    )

        return uploaded

    def upload_all_and_delete_local(self, job: JobRecord) -> list[StoredArtifact]:
        uploaded = self.upload_job_artifacts(job)
        job_dir = Path(job.artifacts.output_dir)
        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
        return uploaded

    def _key_prefix(self, job: JobRecord) -> str:
        scenario_id = _safe_segment(job.request.scenario_id or job.run_id or job.job_id)
        run_id = _safe_segment(job.run_id or "pending-run")
        parts = [self.settings.storage_prefix, scenario_id, "executions", _safe_segment(job.job_id), run_id]
        return "/".join(part for part in parts if part)

    def _ensure_calibration_files(
        self,
        run_dir: Path,
        sensor_configs: dict[str, dict[str, Any]],
        gt_sensor_names: list[str],
    ) -> None:
        calibration_dir = run_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        for sensor_id, sensor in sensor_configs.items():
            sensor_dir = calibration_dir / sensor_id
            sensor_dir.mkdir(parents=True, exist_ok=True)
            intrinsics_path = sensor_dir / "intrinsics.json"
            extrinsics_path = sensor_dir / "extrinsics.json"
            if not intrinsics_path.exists():
                if sensor.get("sensor_category") == "camera":
                    intrinsics = _camera_intrinsics(
                        int(sensor.get("width", 1920)),
                        int(sensor.get("height", 1080)),
                        float(sensor.get("fov", 90.0)),
                    )
                else:
                    intrinsics = {
                        "sensor_category": sensor.get("sensor_category"),
                        "output_modality": sensor.get("output_modality"),
                        "channels": sensor.get("channels"),
                        "range": sensor.get("range") or sensor.get("radar_range"),
                        "points_per_second": sensor.get("points_per_second"),
                        "rotation_frequency": sensor.get("rotation_frequency"),
                        "horizontal_fov": sensor.get("horizontal_fov"),
                        "vertical_fov": sensor.get("vertical_fov"),
                    }
                intrinsics_path.write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
            if not extrinsics_path.exists():
                extrinsics = {
                    "pose": sensor.get("pose", {}),
                    "attach_to": sensor.get("attach_to"),
                    "attachment_type": sensor.get("attachment_type"),
                    "tracking_target": sensor.get("tracking_target"),
                }
                extrinsics_path.write_text(json.dumps(extrinsics, indent=2), encoding="utf-8")

        for gt_name in gt_sensor_names:
            sensor_id = f"gt-{gt_name}"
            sensor_dir = calibration_dir / sensor_id
            sensor_dir.mkdir(parents=True, exist_ok=True)
            intrinsics_path = sensor_dir / "intrinsics.json"
            extrinsics_path = sensor_dir / "extrinsics.json"
            if not intrinsics_path.exists():
                intrinsics_path.write_text(
                    json.dumps(_camera_intrinsics(GT_WIDTH, GT_HEIGHT, GT_FOV), indent=2),
                    encoding="utf-8",
                )
            if not extrinsics_path.exists():
                extrinsics_path.write_text(
                    json.dumps(
                        {
                            "pose": GT_POSE,
                            "attach_to": "ego",
                            "attachment_type": "rigid",
                            "sensor_type": gt_name,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    def _ensure_gt_index_file(self, run_dir: Path, job: JobRecord, prefix: str) -> None:
        bbox_dir = run_dir / "gt" / "bbox"
        if not bbox_dir.is_dir():
            return
        frame_paths = sorted(bbox_dir.glob("frame_*.json"))
        if not frame_paths:
            return
        scene_dir = run_dir / "gt" / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        index_path = scene_dir / "index.json"
        frame_keys = [f"{prefix}/gt/bbox/{path.name}" for path in frame_paths]
        payload = {
            "scenario_id": job.request.scenario_id or job.job_id,
            "job_id": job.job_id,
            "run_id": job.run_id,
            "frame_count": len(frame_paths),
            "frame_keys": frame_keys,
        }
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
