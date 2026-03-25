from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .config import Settings
from .models import JobRecord, StoredArtifact


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
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency is required in deployed environments
            raise RuntimeError("boto3 must be installed to enable S3 artifact uploads.") from exc
        self.settings = settings
        self.bucket = settings.storage_bucket
        self.client = boto3.client("s3", region_name=settings.storage_region)

    def upload_job_artifacts(self, job: JobRecord) -> list[StoredArtifact]:
        prefix = self._key_prefix(job)
        uploaded: list[StoredArtifact] = []
        artifact_specs = (
            ("MANIFEST", job.artifacts.manifest_path, "application/json"),
            ("MP4", job.artifacts.recording_path, "video/mp4"),
            ("SCENARIO_LOG", job.artifacts.scenario_log_path, "text/plain; charset=utf-8"),
            ("RUN_LOG", job.artifacts.debug_log_path, "text/plain; charset=utf-8"),
        )
        for kind, local_path, content_type in artifact_specs:
            if not local_path:
                continue
            path = Path(local_path)
            if not path.is_file():
                continue
            key = f"{prefix}/{path.name}"
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            uploaded.append(
                StoredArtifact(
                    kind=kind,
                    label=path.name,
                    local_path=str(path),
                    content_type=content_type,
                    file_ext=_file_ext(path),
                    size_bytes=path.stat().st_size,
                    checksum_sha256=_checksum_sha256(path),
                    s3_bucket=self.bucket,
                    s3_key=key,
                    s3_uri=f"s3://{self.bucket}/{key}",
                )
            )

        # Upload per-sensor MP4 recordings as individually tracked artifacts
        job_dir = Path(job.artifacts.output_dir)
        # Sensor videos are at: {run_dir}/sensors/{sensor_id}/recording.mp4
        # The run_id subdirectory is the actual run output dir
        if job.run_id:
            sensors_dir = job_dir / job.run_id / "sensors"
            if sensors_dir.is_dir():
                # Read sensor labels from manifest if available
                sensor_labels: dict[str, str] = {}
                try:
                    import json as _json
                    manifest_path = job_dir / job.run_id / "manifest.json"
                    if manifest_path.exists():
                        manifest = _json.loads(manifest_path.read_text())
                        sensor_labels = manifest.get("sensor_labels") or {}
                except Exception:
                    pass
                for sensor_dir in sorted(sensors_dir.iterdir()):
                    if not sensor_dir.is_dir():
                        continue
                    mp4 = sensor_dir / "recording.mp4"
                    if not mp4.is_file():
                        continue
                    sensor_id = sensor_dir.name
                    label = sensor_labels.get(sensor_id, sensor_id)
                    key = f"{prefix}/sensors/{sensor_id}/recording.mp4"
                    self.client.upload_file(
                        str(mp4),
                        self.bucket,
                        key,
                        ExtraArgs={"ContentType": "video/mp4"},
                    )
                    uploaded.append(
                        StoredArtifact(
                            kind="MP4",
                            label=label,
                            local_path=str(mp4),
                            content_type="video/mp4",
                            file_ext="mp4",
                            size_bytes=mp4.stat().st_size,
                            checksum_sha256=_checksum_sha256(mp4),
                            s3_bucket=self.bucket,
                            s3_key=key,
                            s3_uri=f"s3://{self.bucket}/{key}",
                        )
                    )

        return uploaded


    def upload_all_and_delete_local(self, job: JobRecord) -> list[StoredArtifact]:
        """Upload all job artifacts to S3, then delete the entire local job directory."""
        import logging
        import shutil
        log = logging.getLogger(__name__)

        # First do the standard upload
        uploaded = self.upload_job_artifacts(job)

        # Also upload any remaining files in the job directory
        job_dir = Path(job.artifacts.output_dir)
        if job_dir.is_dir():
            prefix = self._key_prefix(job)
            already_uploaded = {a.local_path for a in uploaded}
            for file_path in job_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if str(file_path) in already_uploaded:
                    continue
                # Skip raw sensor frame files — only upload encoded MP4s, not individual frames
                if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.ply', '.csv'):
                    continue
                rel = file_path.relative_to(job_dir)
                key = f"{prefix}/{rel}"
                try:
                    self.client.upload_file(str(file_path), self.bucket, key)
                except Exception as exc:
                    log.warning(f"Failed to upload {file_path}: {exc}")

            # Delete local directory
            try:
                shutil.rmtree(job_dir)
                log.info(f"Deleted local job dir: {job_dir}")
            except Exception as exc:
                log.warning(f"Failed to delete {job_dir}: {exc}")

        return uploaded

    def _key_prefix(self, job: JobRecord) -> str:
        source_run_id = _safe_segment(job.request.source_run_id or job.run_id or job.job_id)
        run_id = _safe_segment(job.run_id or "pending-run")
        parts = [
            self.settings.storage_prefix,
            source_run_id,
            "executions",
            _safe_segment(job.job_id),
            run_id,
        ]
        return "/".join(part for part in parts if part)
