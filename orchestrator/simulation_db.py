"""
Simulation persistence — writes simulation records and artifacts to Aurora.

This is the single writer for the `simulations` and `simulation_artifacts` tables.
The orchestrator creates records as it runs, so the data is always consistent
with what actually happened.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from .db import execute, query_rows, param, json_param

logger = logging.getLogger(__name__)


def get_workspace_for_scenario(scenario_id: str) -> str | None:
    """Look up the workspace_id for a scenario from Aurora."""
    try:
        rows = query_rows(
            "SELECT workspace_id FROM scenarios WHERE id = :id OR legacy_run_id = :id LIMIT 1",
            [param("id", scenario_id)],
        )
        if rows:
            return rows[0].get("workspace_id")
    except Exception as exc:
        logger.warning(f"Could not look up workspace for scenario {scenario_id}: {exc}")
    return None


def create_simulation(
    scenario_id: str,
    workspace_id: str | None,
    map_name: str,
    orchestrator_job_id: str,
    orchestrator_base_url: str | None = None,
    request_payload: dict | None = None,
    user_id: str | None = None,
) -> str:
    """Create a simulation record when a render starts. Returns simulation ID."""
    # Auto-resolve workspace from scenario if not provided
    if not workspace_id and scenario_id:
        workspace_id = get_workspace_for_scenario(scenario_id)
    if not workspace_id:
        logger.warning("No workspace_id found, skipping simulation record")
        return ""

    sim_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    try:
        execute(
            """INSERT INTO simulations
                (id, workspace_id, scenario_id, scenario_legacy_id,
                 orchestrator_job_id, status, map_name,
                 orchestrator_base_url, request_payload,
                 created_by_user_id, created_at, updated_at)
            VALUES
                (:id, :workspace_id, :scenario_id, :scenario_legacy_id,
                 :orchestrator_job_id, :status, :map_name,
                 :orchestrator_base_url, :request_payload,
                 :created_by_user_id, :created_at::timestamptz, :updated_at::timestamptz)""",
            [
                param("id", sim_id),
                param("workspace_id", workspace_id),
                param("scenario_id", scenario_id),
                param("scenario_legacy_id", scenario_id),
                param("orchestrator_job_id", orchestrator_job_id),
                param("status", "running"),
                param("map_name", map_name),
                param("orchestrator_base_url", orchestrator_base_url),
                json_param("request_payload", request_payload or {}),
                param("created_by_user_id", user_id),
                param("created_at", now),
                param("updated_at", now),
            ],
        )
        logger.info(f"Created simulation {sim_id} for scenario {scenario_id} (workspace {workspace_id})")
        return sim_id
    except Exception as exc:
        logger.error(f"Failed to create simulation record: {exc}")
        raise


def update_simulation_status(
    simulation_id: str,
    status: str,
    backend_run_id: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update a simulation's status (running -> completed/failed)."""
    now = datetime.now(timezone.utc).isoformat()
    finished = now if status in ("completed", "failed", "cancelled") else None

    try:
        execute(
            """UPDATE simulations SET
                status = :status,
                backend_run_id = COALESCE(:backend_run_id, backend_run_id),
                error_message = :error_message,
                finished_at = CASE WHEN :finished_at IS NOT NULL
                    THEN :finished_at::timestamptz ELSE finished_at END,
                updated_at = :updated_at::timestamptz
            WHERE id = :id""",
            [
                param("id", simulation_id),
                param("status", status),
                param("backend_run_id", backend_run_id),
                param("error_message", error_message),
                param("finished_at", finished),
                param("updated_at", now),
            ],
        )
        logger.info(f"Updated simulation {simulation_id} -> {status}")
    except Exception as exc:
        logger.error(f"Failed to update simulation {simulation_id}: {exc}")


def create_artifact(
    simulation_id: str,
    scenario_id: str,
    workspace_id: str,
    kind: str,
    s3_bucket: str,
    s3_key: str,
    label: str | None = None,
    content_type: str | None = None,
    file_ext: str | None = None,
    size_bytes: int | None = None,
    checksum_sha256: str | None = None,
    user_id: str | None = None,
) -> str:
    """Create an artifact record after uploading to S3. Returns artifact ID."""
    artifact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    try:
        execute(
            """INSERT INTO simulation_artifacts
                (id, workspace_id, scenario_legacy_id, scenario_id, simulation_id,
                 kind, label, content_type, file_ext, size_bytes, checksum_sha256,
                 s3_bucket, s3_key, created_by_user_id, created_at)
            VALUES
                (:id, :workspace_id, :scenario_legacy_id, :scenario_id, :simulation_id,
                 :kind, :label, :content_type, :file_ext, :size_bytes, :checksum_sha256,
                 :s3_bucket, :s3_key, :created_by_user_id, :created_at::timestamptz)""",
            [
                param("id", artifact_id),
                param("workspace_id", workspace_id),
                param("scenario_legacy_id", scenario_id),
                param("scenario_id", scenario_id),
                param("simulation_id", simulation_id),
                param("kind", kind),
                param("label", label),
                param("content_type", content_type),
                param("file_ext", file_ext),
                param("size_bytes", size_bytes),
                param("checksum_sha256", checksum_sha256),
                param("s3_bucket", s3_bucket),
                param("s3_key", s3_key),
                param("created_by_user_id", user_id),
                param("created_at", now),
            ],
        )
        logger.info(f"Created artifact {artifact_id} ({kind}) for simulation {simulation_id}")
        return artifact_id
    except Exception as exc:
        logger.error(f"Failed to create artifact record: {exc}")
        raise
