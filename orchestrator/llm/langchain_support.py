from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LANGCHAIN_IMPORT_ERROR: Exception | None = None
LANGSMITH_IMPORT_ERROR: Exception | None = None

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

try:
    from langchain_aws import ChatBedrockConverse
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    LANGCHAIN_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    ChatBedrockConverse = None
    AIMessage = Any  # type: ignore[assignment]
    HumanMessage = Any  # type: ignore[assignment]
    SystemMessage = Any  # type: ignore[assignment]
    ToolMessage = Any  # type: ignore[assignment]
    StructuredTool = Any  # type: ignore[assignment]
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = exc

try:
    from langsmith import traceable, tracing_context

    LANGSMITH_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    LANGSMITH_AVAILABLE = False
    LANGSMITH_IMPORT_ERROR = exc

    def traceable(*decorator_args: Any, **decorator_kwargs: Any):  # type: ignore[override]
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator

    @contextmanager
    def tracing_context(**_: Any):
        yield


def langsmith_tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


def langsmith_run_config(name: str, *, tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "run_name": name,
        "tags": tags or [],
        "metadata": metadata or {},
    }


def create_chat_model(model_id: str, *, temperature: float, max_tokens: int):
    if not LANGCHAIN_AVAILABLE or ChatBedrockConverse is None:
        raise RuntimeError(f"LangChain Bedrock support is unavailable: {LANGCHAIN_IMPORT_ERROR}")
    region = os.environ.get("AWS_BEDROCK_REGION") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
    kwargs: dict = {
        "model": model_id,
        "region_name": region,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    bedrock_key = os.environ.get("AWS_BEDROCK_ACCESS_KEY_ID")
    bedrock_secret = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")
    if bedrock_key and bedrock_secret:
        import boto3
        kwargs["client"] = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=bedrock_key,
            aws_secret_access_key=bedrock_secret,
        )
    return ChatBedrockConverse(**kwargs)


def serialize_ai_message(message: Any) -> dict[str, Any]:
    if message is None:
        return {}
    return {
        "id": getattr(message, "id", None),
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "response_metadata": getattr(message, "response_metadata", None),
        "usage_metadata": getattr(message, "usage_metadata", None),
    }
