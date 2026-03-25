"""
Aurora RDS Data API client for the orchestrator.

Uses the same shared Aurora cluster as the SvelteKit editor.
Credentials come from environment variables.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import boto3

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "rds-data",
            region_name=os.environ.get("AURORA_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
    return _client


def _get_arns():
    return {
        "resourceArn": os.environ["AURORA_CLUSTER_ARN"],
        "secretArn": os.environ["AURORA_SECRET_ARN"],
        "database": os.environ.get("AURORA_DATABASE", "simcloud"),
    }


def execute(sql: str, params: list[dict] | None = None) -> dict:
    """Execute a SQL statement against Aurora."""
    client = _get_client()
    kwargs = {**_get_arns(), "sql": sql, "includeResultMetadata": True}
    if params:
        kwargs["parameters"] = params
    return client.execute_statement(**kwargs)


def query_rows(sql: str, params: list[dict] | None = None) -> list[dict[str, Any]]:
    """Execute a query and return rows as dicts."""
    result = execute(sql, params)
    columns = [col["name"] for col in (result.get("columnMetadata") or [])]
    rows = []
    for record in result.get("records") or []:
        row = {}
        for i, field in enumerate(record):
            if field.get("isNull"):
                row[columns[i]] = None
            elif "stringValue" in field:
                row[columns[i]] = field["stringValue"]
            elif "longValue" in field:
                row[columns[i]] = field["longValue"]
            elif "doubleValue" in field:
                row[columns[i]] = field["doubleValue"]
            elif "booleanValue" in field:
                row[columns[i]] = field["booleanValue"]
            else:
                row[columns[i]] = None
        rows.append(row)
    return rows


def param(name: str, value: str | int | float | bool | None) -> dict:
    """Build a typed SQL parameter."""
    if value is None:
        return {"name": name, "value": {"isNull": True}}
    if isinstance(value, bool):
        return {"name": name, "value": {"booleanValue": value}}
    if isinstance(value, int):
        return {"name": name, "value": {"longValue": value}}
    if isinstance(value, float):
        return {"name": name, "value": {"doubleValue": value}}
    return {"name": name, "value": {"stringValue": str(value)}}


def json_param(name: str, value: Any) -> dict:
    """Build a JSON-typed SQL parameter."""
    import json
    return {
        "name": name,
        "value": {"stringValue": json.dumps(value)},
        "typeHint": "JSON",
    }
