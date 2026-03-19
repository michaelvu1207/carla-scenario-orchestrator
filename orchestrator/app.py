from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.websocket("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str, websocket: WebSocket):
    await websocket.accept()
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
                await websocket.send_json(batch)
                sent += len(new_events)
            else:
                await asyncio.sleep(0.01)
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
