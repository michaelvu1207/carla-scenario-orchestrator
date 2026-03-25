from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """Allow Chrome Private Network Access (PNA) for Tailscale Funnel."""
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

from .config import Settings
from .models import CancelJobResponse, JobListResponse, JobRecord, JobSubmissionResponse
from .service import OrchestratorService
from .carla_runner.models import (
    LLMGenerateRequest,
    MapLoadRequest,
    SceneAssistantRequest,
    SimulationRunRequest,
)


settings = Settings.load()
service = OrchestratorService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.startup()
    yield


app = FastAPI(title="CARLA Scenario Orchestrator", lifespan=lifespan)

# Prometheus metrics endpoint at /metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(PrivateNetworkAccessMiddleware)



@app.get("/api/health")
async def health():
    return service.health()


@app.get("/api/carla/status")
async def carla_status():
    return service.carla_status()


@app.get("/api/carla/maps")
async def carla_maps():
    return service.carla_status()


@app.post("/api/carla/map/load")
async def carla_map_load(request: MapLoadRequest):
    try:
        return service.carla_load_map(request.map_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/capacity")
async def capacity():
    return service.capacity()


@app.get("/api/maps/supported")
async def supported_maps():
    return {"maps": service.supported_maps()}


@app.get("/api/map/info")
async def map_info():
    try:
        return service.map_info()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# DEPRECATED: runtime data is now included in GET /api/map/generated under the "runtime" key
@app.get("/api/map/runtime")
async def runtime_map():
    try:
        return service.runtime_map()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/map/xodr")
async def map_xodr():
    try:
        return Response(content=service.map_xodr(), media_type="application/xml")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/map/generated")
async def map_generated():
    try:
        return service.map_generated()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/actors/blueprints")
async def actor_blueprints():
    try:
        return service.actor_blueprints()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/llm/generate")
async def llm_generate(request: LLMGenerateRequest):
    if not request.selected_roads:
        raise HTTPException(status_code=400, detail="Select at least one road before generating actors.")
    try:
        return service.llm_generate(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/llm/scene-assistant")
async def llm_scene_assistant(request: SceneAssistantRequest):
    try:
        return service.llm_scene_assistant_chat(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs():
    return service.list_jobs()


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
async def get_job(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/jobs", response_model=JobSubmissionResponse)
async def submit_job(request: SimulationRunRequest):
    return service.submit_job(request)


@app.post("/api/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    try:
        return service.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc




class PreloadRequest(BaseModel):
    map_name: str

@app.post("/api/slots/preload")
async def preload_map(req: PreloadRequest):
    """Pre-load a map on an idle slot so the next simulation is instant."""
    from .carla_runner.dataset_repository import normalize_map_name
    target_map = normalize_map_name(req.map_name)

    # Check if any slot already has this map
    capacity = service.capacity()
    for slot in capacity.slots:
        if slot.role != "execution":
            continue
        if slot.current_map and normalize_map_name(slot.current_map) == target_map:
            return {"status": "already_loaded", "slot_index": slot.slot_index, "map": slot.current_map}

    # Find an idle slot to preload on
    for slot in capacity.slots:
        if slot.role != "execution" or slot.busy or slot.status != "ready":
            continue
        # Dispatch preload via worker pool
        if service.worker_pool:
            import asyncio
            asyncio.ensure_future(service.worker_pool.dispatch_preload(slot.slot_index, req.map_name))
            return {"status": "preloading", "slot_index": slot.slot_index}

    return {"status": "no_idle_slots"}

@app.post("/api/jobs/{job_id}/events")
async def push_job_event(job_id: str, request: StarletteRequest):
    """Internal: workers push simulation events here in real-time (single or batch)."""
    from .carla_runner.models import SimulationStreamMessage
    from .models import JobState
    data = await request.json()
    envelopes = data if isinstance(data, list) else [data]
    for envelope in envelopes:
        if envelope.get("kind") == "stream":
            payload = SimulationStreamMessage.model_validate(envelope.get("payload", {}))
            service.store.append_event(job_id, payload)
            job = service.get_job(job_id)
            if job and job.state == JobState.starting:
                service.store.update(job_id, state=JobState.running)
            if payload.error and job and job.state != JobState.cancelled:
                service.store.update(job_id, error=payload.error)
            if payload.recording is not None:
                service.store.update(job_id, run_id=payload.recording.run_id)
    return {"ok": True}

@app.websocket("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str, websocket: WebSocket):
    await websocket.accept()
    # Resume from client-specified event index to avoid replaying old events on reconnect
    after_str = websocket.query_params.get("after", "0")
    try:
        sent = max(0, int(after_str))
    except (ValueError, TypeError):
        sent = 0
    try:
        while True:
            job = service.get_job(job_id)
            if job is None:
                await websocket.close(code=4404)
                return
            new_events = job.events[sent:]
            if new_events:
                batch = [event.payload.model_dump() for event in new_events]
                await websocket.send_json({"events": batch, "next_index": sent + len(new_events)})
                sent += len(new_events)
            # Terminate after all events sent for completed jobs
            if job.state.value in ("succeeded", "failed", "cancelled") and sent >= len(job.events):
                await websocket.send_json({"stream_complete": True})
                await websocket.close()
                return
            if not new_events:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return













@app.get('/api/jobs/{job_id}/log')
async def job_log(job_id: str):
    log_text = service.get_job_log(job_id)
    if log_text is None:
        raise HTTPException(status_code=404, detail='Job or log not found.')
    return {'log': log_text}



@app.get("/api/jobs/{job_id}/diagnostics")
async def job_diagnostics(job_id: str):
    diagnostics = service.job_diagnostics(job_id)
    if diagnostics is None:
        raise HTTPException(status_code=404, detail="Diagnostics not found for this job.")
    return diagnostics


@app.get("/api/jobs/{job_id}/recordings")
async def job_recordings(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    recordings = service.list_recordings()
    matched = [r for r in recordings if r.run_id == job.run_id and job.run_id]
    return {"items": [item.model_dump() for item in matched]}



@app.get("/api/recordings")
async def list_all_recordings(source_run_id: str | None = None):
    """List all recordings, optionally filtered by source_run_id (scenario ID)."""
    recordings = service.list_recordings(source_run_id=source_run_id)
    return {"items": [r.model_dump() for r in recordings]}

# DEPRECATED: legacy path was /api/simulation/recordings/file
@app.get("/api/recordings/file")
async def recording_file(path: str):
    file_path = Path(path).resolve()
    jobs_root = settings.jobs_root.resolve()
    if not file_path.is_relative_to(jobs_root):
        raise HTTPException(status_code=403, detail="Access denied: path is outside the jobs directory.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Recording not found.")
    return FileResponse(file_path)
