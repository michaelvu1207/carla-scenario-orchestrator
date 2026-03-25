from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

TIMELINE_DURATION_LIMIT_SECONDS = 15.0
VEHICLE_SPEED_LIMIT_KPH = 240.0


class SelectedRoad(BaseModel):
    id: str
    name: str = ""
    length: float = 0.0
    tags: list[str] = Field(default_factory=list)
    section_labels: list[str] = Field(default_factory=list)


class ActorRoadAnchor(BaseModel):
    road_id: str
    s_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    lane_id: int | None = None
    section_id: int | None = None


class ActorMapPoint(BaseModel):
    x: float
    y: float


class TimedWaypoint(BaseModel):
    x: float
    y: float
    time: float


TimelineAction = Literal[
    "follow_route",
    "set_speed",
    "stop",
    "hold_position",
    "enable_autopilot",
    "disable_autopilot",
    "lane_change_left",
    "lane_change_right",
    "turn_left_at_next_intersection",
    "turn_right_at_next_intersection",
    "chase_actor",
    "ram_actor",
]


class ActorTimelineClip(BaseModel):
    id: str
    start_time: float = Field(default=0.0, ge=0.0, le=TIMELINE_DURATION_LIMIT_SECONDS)
    end_time: float | None = Field(default=None, ge=0.0, le=TIMELINE_DURATION_LIMIT_SECONDS)
    action: TimelineAction
    target_speed_kph: float | None = Field(default=None, ge=0.0, le=VEHICLE_SPEED_LIMIT_KPH)
    target_actor_id: str | None = None
    following_distance_m: float | None = Field(default=None, ge=0.0, le=50.0)
    enabled: bool = True


class ActorDraft(BaseModel):
    id: str
    label: str
    kind: Literal["vehicle", "walker", "prop"]
    role: Literal["ego", "traffic", "pedestrian", "prop"] = "traffic"
    is_static: bool = False
    placement_mode: Literal["road", "path", "point", "timed_path"] = "road"
    blueprint: str
    spawn: ActorRoadAnchor
    spawn_point: ActorMapPoint | None = None
    route: list[ActorRoadAnchor] = Field(default_factory=list)
    route_direction: Literal["forward", "reverse"] = "forward"
    lane_facing: Literal["with_lane", "against_lane"] = "with_lane"
    destination: ActorRoadAnchor | None = None
    destination_point: ActorMapPoint | None = None
    speed_kph: float = Field(default=100.0, ge=0.0, le=VEHICLE_SPEED_LIMIT_KPH)
    autopilot: bool = True
    color: str | None = None
    notes: str | None = None
    timed_waypoints: list[TimedWaypoint] = Field(default_factory=list)
    timeline: list[ActorTimelineClip] = Field(default_factory=list)


class LLMGenerateRequest(BaseModel):
    map_name: str
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    prompt: str = Field(min_length=1, max_length=6000)
    max_actors: int = Field(default=6, ge=1, le=12)


class LLMGenerateResponse(BaseModel):
    model: str
    summary: str
    actors: list[ActorDraft]
    raw_json: dict[str, Any]


class SceneAssistantMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class SceneAssistantToolTrace(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class SceneAssistantRequest(BaseModel):
    map_name: str
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    runtime_map: "RuntimeMapResponse"
    actors: list[ActorDraft] = Field(default_factory=list)
    live_actors: list["SimulationActorState"] = Field(default_factory=list)
    selected_actor_id: str | None = None
    messages: list[SceneAssistantMessage] = Field(default_factory=list, min_length=1)


class SceneAssistantResponse(BaseModel):
    model: str
    reply: str
    map_name: str
    normalized_map_name: str
    actors: list[ActorDraft] = Field(default_factory=list)
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    selected_actor_id: str | None = None
    tool_trace: list[SceneAssistantToolTrace] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class SimulationRunRequest(BaseModel):
    source_run_id: str | None = None
    map_name: str
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    actors: list[ActorDraft] = Field(default_factory=list)
    duration_seconds: float = Field(default=TIMELINE_DURATION_LIMIT_SECONDS, ge=1.0, le=TIMELINE_DURATION_LIMIT_SECONDS)
    fixed_delta_seconds: float = Field(default=0.05, ge=0.01, le=0.2)
    topdown_recording: bool = True
    recording_width: int = Field(default=1280, ge=320, le=3840)
    recording_height: int = Field(default=720, ge=240, le=2160)
    recording_fov: float = Field(default=90.0, ge=30.0, le=140.0)
    sensors: list[SensorConfig] = Field(default_factory=list)


class RecordingInfo(BaseModel):
    run_id: str
    label: str
    mp4_path: str | None = None
    frames_path: str | None = None
    created_at: str
    source_run_id: str | None = None
    s3_url: str | None = None


class SimulationRunDiagnostics(BaseModel):
    run_id: str
    map_name: str
    created_at: str
    source_run_id: str | None = None
    s3_url: str | None = None
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    actors: list[ActorDraft] = Field(default_factory=list)
    recording_path: str | None = None
    scenario_log_path: str | None = None
    debug_log_path: str | None = None
    worker_error: str | None = None
    saved_frame_count: int = 0
    sensor_timeout_count: int = 0
    last_sensor_frame: int | None = None
    skipped_actors: list[dict[str, Any]] = Field(default_factory=list)
    log_excerpt: str = ""


class CarlaMapInfo(BaseModel):
    name: str
    normalized_name: str


class CarlaStatusResponse(BaseModel):
    connected: bool
    current_map: str | None = None
    normalized_map_name: str | None = None
    server_version: str | None = None
    client_version: str | None = None
    available_maps: list[CarlaMapInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MapLoadRequest(BaseModel):
    map_name: str


class RuntimeRoadSegment(BaseModel):
    id: str
    road_id: int
    section_id: int
    lane_id: int
    lane_type: str | None = None
    is_junction: bool = False
    left_lane_id: int | None = None
    right_lane_id: int | None = None
    centerline: list[dict[str, float]]


class RuntimeRoadSectionSummary(BaseModel):
    index: int
    label: str = ""
    s: float = 0.0
    driving_left: int = 0
    driving_right: int = 0
    parking_left: int = 0
    parking_right: int = 0
    total_driving: int = 0
    total_width: float = 0.0
    lane_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RuntimeRoadSummary(BaseModel):
    id: str
    name: str = ""
    is_intersection: bool = False
    tags: list[str] = Field(default_factory=list)
    lane_types: list[str] = Field(default_factory=list)
    has_parking: bool = False
    has_shoulder: bool = False
    has_sidewalk: bool = False
    section_summaries: list[RuntimeRoadSectionSummary] = Field(default_factory=list)


class RuntimeMapResponse(BaseModel):
    map_name: str
    normalized_map_name: str
    road_segments: list[RuntimeRoadSegment]
    lane_type_counts: dict[str, int] = Field(default_factory=dict)
    dataset_lane_type_counts: dict[str, int] = Field(default_factory=dict)
    road_summaries: list[RuntimeRoadSummary] = Field(default_factory=list)
    dataset_augmented: bool = False


class SimulationActorState(BaseModel):
    id: int
    label: str
    kind: str
    role: str
    x: float
    y: float
    z: float
    yaw: float
    speed_mps: float
    road_id: int | None = None
    section_id: int | None = None
    lane_id: int | None = None


class SimulationStreamMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    frame: int
    timestamp: float
    event_kind: str = "frame"
    actors: list[SimulationActorState] = Field(default_factory=list)
    simulation_ended: bool = False
    error: str | None = None
    recording: RecordingInfo | None = None
    frame_jpeg: str | None = None


# ── Sensor Configuration (multi-sensor support) ──

class SensorPose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class SensorConfig(BaseModel):
    """Individual sensor configuration sent from the frontend."""
    id: str
    label: str
    sensor_category: Literal["camera", "lidar", "radar", "imu", "gnss"] = "camera"
    output_modality: str = "rgb"
    attachment_type: Literal["rigid", "spring_arm", "spring_arm_ghost"] = "rigid"
    pose: SensorPose = Field(default_factory=SensorPose)
    update_rate: float = Field(default=30.0, ge=0.0, le=60.0)
    attach_to: str = "ego"  # actor ID or "world"

    # Camera-specific
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=240, le=2160)
    fov: float = Field(default=90.0, ge=1.0, le=179.0)

    # LiDAR-specific
    channels: int = Field(default=64, ge=1, le=128)
    range_m: float = Field(default=100.0, ge=1.0, le=200.0)
    points_per_second: int = Field(default=1_200_000, ge=100)
    rotation_frequency: float = Field(default=20.0, ge=1.0, le=60.0)

    # Radar-specific
    horizontal_fov: float = Field(default=30.0, ge=1.0, le=179.0)
    vertical_fov: float = Field(default=30.0, ge=1.0, le=179.0)
    radar_range: float = Field(default=100.0, ge=1.0, le=300.0)

    # World-placed cameras
    world_position: ActorMapPoint | None = None
    world_rotation: SensorPose | None = None
