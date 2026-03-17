from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import Settings
from .models import CancelJobResponse, CompatibilityRunResponse, JobListResponse, JobRecord, JobSubmissionResponse
from .service import OrchestratorService
from .carla_runner.models import (
    LLMGenerateRequest,
    MapLoadRequest,
    SceneAssistantRequest,
    SimulationRunRequest,
)


settings = Settings.load()
service = OrchestratorService(settings)
app = FastAPI(title="CARLA Scenario Orchestrator")

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


@app.get("/api/map/runtime")
async def runtime_map():
    try:
        return service.runtime_map()
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
            for event in new_events:
                await websocket.send_json(event.payload.model_dump())
                sent += 1
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


@app.websocket("/api/simulation/stream")
async def compatibility_stream(websocket: WebSocket, job_id: str | None = Query(default=None)):
    if not job_id:
        latest = service.latest_running_job()
        job_id = latest.job_id if latest else None
    if not job_id:
        await websocket.accept()
        await websocket.close(code=4404)
        return
    await job_stream(job_id, websocket)


@app.post("/api/simulation/run", response_model=CompatibilityRunResponse)
async def compatibility_run(request: SimulationRunRequest):
    return service.submit_compatibility_job(request)


@app.post("/api/simulation/stop")
async def compatibility_stop(job_id: str | None = Query(default=None)):
    if job_id:
        try:
            return service.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc
    latest = service.cancel_latest_running_job()
    if latest is None:
        return {"status": "idle", "running": False}
    return {"status": "stopping", "running": latest.state not in {"failed", "succeeded", "cancelled"}}


@app.get("/api/simulation/status")
async def compatibility_status(job_id: str | None = Query(default=None)):
    return service.simulation_status(job_id)


@app.post("/api/simulation/pause")
async def compatibility_pause(job_id: str | None = Query(default=None)):
    try:
        return service.pause_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/simulation/resume")
async def compatibility_resume(job_id: str | None = Query(default=None)):
    try:
        return service.resume_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/simulation/recordings")
async def simulation_recordings():
    return {"items": [item.model_dump() for item in service.list_recordings()]}


@app.get("/api/simulation/runs/latest")
async def simulation_latest_run():
    diagnostics = service.latest_run_diagnostics()
    if diagnostics is None:
        raise HTTPException(status_code=404, detail="No simulation runs found.")
    return diagnostics


@app.get("/api/simulation/runs/{run_id}")
async def simulation_run_details(run_id: str):
    diagnostics = service.job_diagnostics(run_id)
    if diagnostics is None:
        raise HTTPException(status_code=404, detail="Simulation run not found.")
    return diagnostics


@app.get("/api/simulation/recordings/file")
async def simulation_recording_file(path: str):
    file_path = Path(path).resolve()
    jobs_root = settings.jobs_root.resolve()
    if not file_path.is_relative_to(jobs_root):
        raise HTTPException(status_code=403, detail="Access denied: path is outside the jobs directory.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Recording not found.")
    return FileResponse(file_path)
