from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any

import boto3
from pydantic import BaseModel, Field

from .langchain_support import (
    LANGCHAIN_AVAILABLE,
    create_chat_model,
    langsmith_run_config,
    serialize_ai_message,
    traceable,
)
from ..carla_runner.models import ActorDraft, LLMGenerateRequest, LLMGenerateResponse


DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("BEDROCK_MODEL_ID", "").strip(),
    "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
]


class GeneratedScenarioPayload(BaseModel):
    summary: str = Field(default="Generated scenario draft.")
    actors: list[ActorDraft] = Field(default_factory=list)


class BedrockScenarioLLM:
    def __init__(self) -> None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = ""

    def _system_prompt(self) -> str:
        return dedent(
            """
            You are generating actor spawn drafts for a CARLA scenario editor.
            Return strict JSON with this shape:
            {
              "summary": "short sentence",
              "actors": [
                {
                  "id": "short-kebab-id",
                  "label": "Human readable name",
                  "kind": "vehicle" | "walker",
                  "role": "ego" | "traffic" | "pedestrian",
                  "blueprint": "valid Carla blueprint pattern",
                  "spawn": {
                    "road_id": "road id from the provided list",
                    "s_fraction": 0.0 to 1.0,
                    "lane_id": optional integer
                  },
                  "destination": optional same shape as spawn,
                  "speed_kph": number,
                  "autopilot": true or false,
                  "color": optional "R,G,B",
                  "notes": optional short text
                }
              ]
            }

            Rules:
            - Use only provided road ids.
            - Keep actor count modest and realistic.
            - Default vehicle actors to autopilot true unless the request explicitly implies a stopped vehicle.
            - Use walker blueprints with prefix walker.pedestrian.
            - Use vehicle blueprints with prefix vehicle.
            - Do not include markdown fences or prose outside the JSON object.
            """
        ).strip()

    def _user_prompt(self, request: LLMGenerateRequest) -> str:
        road_lines = []
        for road in request.selected_roads:
            road_lines.append(
                f"- road_id={road.id} name={road.name!r} length={road.length:.1f}m tags={','.join(road.tags) or 'none'} sections={','.join(road.section_labels) or 'none'}"
            )
        roads_blob = "\n".join(road_lines) or "- none"
        return dedent(
            f"""
            Map: {request.map_name}
            Max actors: {request.max_actors}

            Selected roads:
            {roads_blob}

            User request:
            {request.prompt}
            """
        ).strip()

    def _extract_json(self, body: str) -> dict[str, Any]:
        text = body.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        return json.loads(text)

    def _invoke(self, model_id: str, request: LLMGenerateRequest) -> dict[str, Any]:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1600,
            "temperature": 0.2,
            "system": self._system_prompt(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._user_prompt(request),
                        }
                    ],
                }
            ],
        }
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        decoded = json.loads(response["body"].read())
        content = decoded.get("content") or []
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return self._extract_json(text)

    def _invoke_langchain(self, model_id: str, request: LLMGenerateRequest) -> dict[str, Any]:
        chat = create_chat_model(model_id, temperature=0.2, max_tokens=1600)
        runnable = chat.with_structured_output(GeneratedScenarioPayload, include_raw=True)
        result = runnable.invoke(
            [
                ("system", self._system_prompt()),
                ("human", self._user_prompt(request)),
            ],
            config=langsmith_run_config(
                "carla_generate_scene",
                tags=["carla", "generate", model_id],
                metadata={
                    "map_name": request.map_name,
                    "selected_road_count": len(request.selected_roads),
                    "max_actors": request.max_actors,
                },
            ),
        )
        if result.get("parsing_error") is not None:
            raise RuntimeError(f"Structured output parsing failed: {result['parsing_error']}")
        parsed = result.get("parsed")
        if parsed is None:
            raise RuntimeError("LangChain returned no parsed scenario payload.")
        if isinstance(parsed, GeneratedScenarioPayload):
            payload = parsed
        else:
            payload = GeneratedScenarioPayload.model_validate(parsed)
        return {
            "summary": payload.summary,
            "actors": [actor.model_dump() for actor in payload.actors],
            "raw_message": serialize_ai_message(result.get("raw")),
        }

    @traceable(run_type="chain", name="carla_generate_scene")
    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResponse:
        errors: list[str] = []
        for model_id in [candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate]:
            try:
                raw = self._invoke_langchain(model_id, request) if LANGCHAIN_AVAILABLE else self._invoke(model_id, request)
                actors = [ActorDraft.model_validate(item) for item in raw.get("actors", [])]
                self.model_id = model_id
                return LLMGenerateResponse(
                    model=model_id,
                    summary=str(raw.get("summary") or "Generated scenario draft."),
                    actors=actors,
                    raw_json=raw,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Bedrock generation failed. " + " | ".join(errors))
