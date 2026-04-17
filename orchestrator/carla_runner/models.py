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
    id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
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
    spawn_yaw: float | None = None
    path_placement: list[ActorMapPoint] = Field(default_factory=list)
    path_spacing: float = Field(default=3.0, ge=0.5, le=50.0)
    timeline: list[ActorTimelineClip] = Field(default_factory=list)


class SimulationRunRequest(BaseModel):
    scenario_id: str | None = None
    correlation_id: str | None = None
    map_name: str
    selected_roads: list[SelectedRoad] = Field(default_factory=list)
    actors: list[ActorDraft] = Field(default_factory=list)
    duration_seconds: float = Field(default=TIMELINE_DURATION_LIMIT_SECONDS, ge=0.01, le=TIMELINE_DURATION_LIMIT_SECONDS)
    fixed_delta_seconds: float = Field(default=0.05, ge=0.01, le=0.2)
    topdown_recording: bool = True
    recording_width: int = Field(default=1280, ge=320, le=3840)
    recording_height: int = Field(default=720, ge=240, le=2160)
    recording_fov: float = Field(default=90.0, ge=30.0, le=140.0)
    sensors: list[SensorConfig] = Field(default_factory=list)
    gt_sensors: list[str] = Field(default_factory=list)  # e.g. ["semantic_seg", "depth", "instance_seg"]
    workspace_id: str | None = None
    submitted_by_agent_id: str | None = None
    test_run: bool = False
    artifact_ttl_hours: int | None = None
    priority: Literal["interactive", "batch"] = "interactive"


class RecordingInfo(BaseModel):
    run_id: str
    label: str
    mp4_path: str | None = None
    frames_path: str | None = None
    created_at: str
    scenario_id: str | None = None
    s3_url: str | None = None


class SimulationRunDiagnostics(BaseModel):
    run_id: str
    map_name: str
    created_at: str
    scenario_id: str | None = None
    s3_url: str | None = None
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
    warning: str | None = None
    recording: RecordingInfo | None = None


# ── Sensor Configuration (multi-sensor support) ──

class SensorPose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class SensorConfig(BaseModel):
    """Unified sensor configuration. Single flat type for all sensor categories.
    World sensors use attach_to='world' with pose as world position.
    Ego sensors use attach_to='ego' with pose as offset from vehicle."""
    id: str
    label: str = ""
    sensor_category: Literal["camera", "lidar", "radar", "imu", "gnss"] = "camera"
    output_modality: Literal[
        "rgb", "depth", "semantic_segmentation", "instance_segmentation",
        "point_cloud", "semantic_point_cloud", "radar_data", "imu_data", "gnss_fix",
    ] = "rgb"
    attachment_type: Literal["rigid", "spring_arm", "spring_arm_ghost"] = "rigid"
    pose: SensorPose = Field(default_factory=SensorPose)
    update_rate: float = Field(default=30.0, ge=0.0, le=60.0)
    attach_to: str = "ego"

    # Camera-specific
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=240, le=2160)
    fov: float = Field(default=90.0, ge=1.0, le=179.0)

    # LiDAR-specific
    channels: int = Field(default=64, ge=1, le=128)
    range: float = Field(default=100.0, ge=1.0, le=200.0)
    points_per_second: int = Field(default=1_200_000, ge=100)
    rotation_frequency: float = Field(default=20.0, ge=1.0, le=60.0)

    # Radar-specific
    horizontal_fov: float = Field(default=30.0, ge=1.0, le=179.0)
    vertical_fov: float = Field(default=30.0, ge=1.0, le=179.0)
    radar_range: float = Field(default=100.0, ge=1.0, le=300.0)

    # Tracking (world sensors that follow a named actor)
    tracking_target: str | None = None
