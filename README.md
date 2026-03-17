# CARLA Scenario Orchestrator

`carla-scenario-orchestrator` is the canonical repo for the SimCloud CARLA backend stack.

It now contains:

- the multi-job orchestrator API
- the shared CARLA scenario runner and authoring logic that previously lived in `carla-scenario-tool-server`
- the ops monitor UI under `monitor/`

The intent is to replace the old split between:

- `carla-scenario-orchestrator`
- `carla-scenario-tool-server`
- `simcloud-monitor`

with one source repository and one primary backend service.

## Architecture

The runtime model is:

- one FastAPI process receives jobs
- one GPU slot is reserved per active job
- one CARLA Docker container is launched per job
- one scenario runner subprocess connects to that CARLA instance and executes the scenario
- job events are captured and exposed over websocket streams
- artifacts are written under `runs/<job_id>/`
- artifacts can be uploaded to S3 for durable storage

The repo layout is:

- `orchestrator/`
  Python backend, job scheduler, runtime backend, artifact upload, compatibility API
- `orchestrator/carla_runner/`
  shared CARLA scenario execution models and worker logic migrated from the former tool server
- `orchestrator/llm/`
  Bedrock scenario generation and scene-assistant code
- `monitor/`
  Svelte monitor UI for host status, orchestrator status, jobs, and container inspection
- `data/maps.generated.json`
  static road metadata used to augment runtime CARLA map data
- `tests/`
  orchestrator unit tests

## Consolidation Changes

The following changes were made in this repo:

- moved the monitor app into `monitor/`
- folded the former tool-server functionality into the orchestrator API surface
- added compatibility endpoints so older frontends can migrate without talking to the old tool server
- added job-aware simulation status reporting
- added orchestrator-side pause and resume support for active jobs
- added a live jobs page to the monitor app
- updated monitor backend defaults to talk to the orchestrator directly
- documented and checked in the production systemd unit used on the host
- documented the current production port plan

The old repos are now legacy:

- `carla-scenario-tool-server`
  legacy single-runtime backend; functionality is now owned here
- `simcloud-monitor`
  legacy standalone monitor repo; active source of truth is now `monitor/` in this repo

## API

Primary orchestrator endpoints:

- `GET /api/health`
- `GET /api/capacity`
- `GET /api/carla/status`
- `GET /api/carla/maps`
- `POST /api/carla/map/load`
- `GET /api/map/info`
- `GET /api/map/runtime`
- `GET /api/maps/supported`
- `GET /api/actors/blueprints`
- `POST /api/llm/generate`
- `POST /api/llm/scene-assistant`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/simulation/status`
- `POST /api/simulation/run`
- `POST /api/simulation/stop`
- `POST /api/simulation/pause`
- `POST /api/simulation/resume`
- `GET /api/simulation/recordings`
- `GET /api/simulation/runs/latest`
- `GET /api/simulation/runs/{run_id}`
- `GET /api/simulation/recordings/file`
- `WS /api/jobs/{job_id}/stream`
- `WS /api/simulation/stream`

Compatibility behavior:

- `POST /api/simulation/run` accepts the legacy `SimulationRunRequest` payload and returns a job identifier
- `GET /api/simulation/status` exposes a tool-server-style status view backed by orchestrator job state
- `GET /api/map/info` exposes map name plus flattened waypoint data for older clients

## Current Production Setup

The currently documented host layout is:

- repo path: `/home/ubuntu/carla-scenario-orchestrator`
- service name: `carla-scenario-orchestrator.service`
- process manager: `systemd`
- orchestrator user: `ubuntu`
- orchestrator working directory: `/home/ubuntu/carla-scenario-orchestrator`

### GPU Allocation

Production is configured to use only GPUs `1-7`.

GPU `0` is intentionally excluded from the orchestrator scheduler.

Configured value:

```env
ORCH_GPU_DEVICES=1,2,3,4,5,6,7
```

### Port Plan

The production port sequence intentionally avoids very common service ranges.

Current production ports:

- orchestrator API: `18421`
- CARLA metadata CARLA host/port: `127.0.0.1:2000`

Per-job CARLA slot ports:

| Slot | GPU | CARLA RPC | Traffic Manager |
| --- | --- | --- | --- |
| 0 | 1 | 18467 | 19467 |
| 1 | 2 | 18504 | 19504 |
| 2 | 3 | 18541 | 19541 |
| 3 | 4 | 18578 | 19578 |
| 4 | 5 | 18615 | 19615 |
| 5 | 6 | 18652 | 19652 |
| 6 | 7 | 18689 | 19689 |

This comes from:

```env
ORCH_CARLA_RPC_PORT_BASE=18467
ORCH_TRAFFIC_MANAGER_PORT_BASE=19467
ORCH_PORT_STRIDE=37
```

Notes:

- `ORCH_CARLA_METADATA_PORT=2000` still points at the always-on local CARLA instance used for metadata queries such as map status, blueprint listing, and runtime map inspection
- per-job CARLA Docker instances do not use port `2000`; they use the uncommon port sequence above

## Systemd

The production unit is checked in at [deploy/systemd/carla-scenario-orchestrator.service](/Users/maikyon/Documents/Programming/carla-scenario-orchestrator/deploy/systemd/carla-scenario-orchestrator.service).

It currently does the following:

- starts the orchestrator on port `18421`
- forces `ORCH_GPU_DEVICES=1,2,3,4,5,6,7`
- restarts automatically on failure
- runs as `ubuntu`

Useful commands on the host:

```bash
sudo systemctl status carla-scenario-orchestrator.service
sudo systemctl restart carla-scenario-orchestrator.service
sudo journalctl -u carla-scenario-orchestrator.service -f
curl http://127.0.0.1:18421/api/health
curl http://127.0.0.1:18421/api/capacity
```

## Environment

See [`.env.example`](/Users/maikyon/Documents/Programming/carla-scenario-orchestrator/.env.example).

Important variables:

- `ORCH_GPU_DEVICES`
- `ORCH_CARLA_IMAGE`
- `ORCH_CARLA_CONTAINER_PREFIX`
- `ORCH_CARLA_STARTUP_TIMEOUT`
- `ORCH_CARLA_RPC_PORT_BASE`
- `ORCH_TRAFFIC_MANAGER_PORT_BASE`
- `ORCH_PORT_STRIDE`
- `ORCH_CARLA_TIMEOUT`
- `ORCH_CARLA_METADATA_HOST`
- `ORCH_CARLA_METADATA_PORT`
- `ORCH_CARLA_METADATA_TIMEOUT`
- `ORCH_DOCKER_NETWORK_MODE`
- `ORCH_CARLA_START_COMMAND`
- `ORCH_STORAGE_BUCKET`
- `ORCH_STORAGE_REGION`
- `ORCH_STORAGE_PREFIX`

Artifact upload layout:

```text
runs/<source_run_id>/executions/<job_id>/<backend_run_id>/
```

If `source_run_id` is present on the submitted request, uploaded artifacts line up with the originating SimCloud run.

## Backend Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn orchestrator.app:app --host 0.0.0.0 --port 18421
```

You still need a matching CARLA Python wheel installed in the same environment for the runner subprocess.

## Monitor

The monitor app lives in `monitor/`.

Build and run:

```bash
cd monitor
npm install
npm run build
PORT=3001 HOST=0.0.0.0 node build
```

Monitor environment example:

- `SSH_HOST=127.0.0.1`
- `SSH_USER=ubuntu`
- `ORCHESTRATOR_URL=http://127.0.0.1:18421`

The monitor currently provides:

- host status via shell and SSH-backed endpoints
- GPU utilization and GPU process views
- container list, log viewing, and restart actions
- orchestrator health and simulation status
- live orchestrator jobs listing

## Testing

Backend tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Monitor build verification:

```bash
cd monitor
npm run build
```
