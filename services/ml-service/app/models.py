"""Shared dataclasses + Pydantic request/response models.

The Pydantic models bind EXACTLY to the AnalysisResult JSON shape in SPEC.md
(snake_case keys). Dataclasses are the light in-memory currency between
detector -> merge -> roles/events -> db, kept free of heavy imports so tests
can import them without pulling in torch/ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# In-memory dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """One person detection on one sampled frame."""

    video_ts_ms: int
    raw_track_id: int
    bbox: dict  # {x, y, w, h} normalized 0-1, top-left based
    conf: float
    standing: bool
    back_to_camera: bool
    track_no: Optional[int] = None  # merged identity, assigned post-merge


@dataclass
class VideoMeta:
    duration_ms: int
    fps: float
    width: int
    height: int


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


# One polygon vertex: exactly [x, y], finite floats only (NaN/Infinity are
# rejected to match the dashboard's parseZones validation). Without the
# inner-length bound a malformed point like [0.5] would pass validation and
# crash geometry.polygon_bbox with an IndexError 500 during derive.
PolygonPoint = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]],
    Field(min_length=2, max_length=2),
]


class ZoneIn(BaseModel):
    kind: Literal["board", "door"]
    polygon: list[PolygonPoint] = Field(min_length=3)


class AnalyzeRequest(BaseModel):
    video_id: str
    video_path: str
    sample_fps: float = 5.0
    zones: list[ZoneIn] = Field(default_factory=list)
    # Client-supplied dedup token. The dashboard's Workflow DevKit step runs
    # at-least-once: a retry after a lost HTTP response re-POSTs /analyze, and
    # without this key the service would enqueue a duplicate full YOLO job.
    # Same key -> the existing job (any status) is returned instead.
    idempotency_key: Optional[str] = None
    # Stale-run fence tokens (the workflow's reanalyze attempt id + run id).
    # Before rewriting detection_events the ML service verifies
    # videos.workflow_run_id is NULL or one of these, so a job whose run was
    # superseded by a newer reanalyze cannot clobber the current run's rows.
    # Empty list = fence disabled (tests / direct API use).
    run_tokens: list[str] = Field(default_factory=list)


class RederiveRequest(BaseModel):
    video_id: str
    zones: list[ZoneIn] = Field(default_factory=list)


class DetectBoardRequest(BaseModel):
    video_id: str
    video_path: str


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class AnalyzeAccepted(BaseModel):
    job_id: str


class JobStatusOut(BaseModel):
    status: Literal["queued", "running", "done", "failed"]
    progress: float
    stage: Literal["detecting", "merging", "deriving"]
    error: Optional[str] = None


class VideoInfoOut(BaseModel):
    duration_ms: int
    fps: float
    width: int
    height: int


class TrackOverlayOut(BaseModel):
    """Permanent overlay tier: survives detection_events compression/retention.

    polyline: RDP-simplified [[ts_ms, cx, cy], ...] of bbox centers;
    keyframes: [[ts_ms, x, y, w, h], ...] sampled at most every 2 s.
    """

    polyline: list[tuple[int, float, float]]
    keyframes: list[tuple[int, float, float, float, float]]


class TrackMetaOut(BaseModel):
    standing_ratio: float
    movement: float
    raw_track_ids: list[int]
    overlay: Optional[TrackOverlayOut] = None


class TrackOut(BaseModel):
    track_no: int
    role: Literal["teacher", "student", "unknown"]
    role_confidence: Optional[float]
    first_ms: int
    last_ms: int
    meta: TrackMetaOut


class EventOut(BaseModel):
    kind: Literal["enter", "exit", "board_enter", "board_leave"]
    video_ts_ms: int
    track_no: Optional[int]


class EntryExitOut(BaseModel):
    kind: Literal["enter", "exit"]
    ts_ms: int


class OccupancyOut(BaseModel):
    ts_ms: int
    students: int
    teacher: bool


class HeatmapOut(BaseModel):
    """Spatial dwell histogram over a grid_w x grid_h grid of the frame.

    teacher / students are row-major flattened per-cell sample counts (grid_h
    rows, grid_w cols); at a fixed sample rate a cell's count is proportional
    to time spent there. Empty lists when no teacher / no detections.
    """

    grid_w: int
    grid_h: int
    teacher: list[int]
    students: list[int]


class QualityTiers(BaseModel):
    overall: Literal["high", "medium", "low"]
    occupancy: Literal["high", "medium", "low"]
    identity: Literal["high", "medium", "low"]
    coverage: Literal["high", "medium", "low"]
    teacher: Literal["high", "medium", "low"]


class DataQualityOut(BaseModel):
    """Additive per-run trust report (app/quality.py). Annotates, never alters,
    the derived numbers: how well the camera covered the lesson, how much the
    tracker fragmented, and a re-identification-independent concurrent crowd
    count that cross-checks the identity-based occupancy."""

    detections: int
    frames: int
    identities: int
    raw_tracks: int
    fragmentation: float
    coverage: float
    occupied_buckets: int
    span_buckets: int
    concurrent_peak: int
    concurrent_typical: int
    confidence: QualityTiers
    notes: list[str]


class AnalyticsOut(BaseModel):
    teacher_present_ms: int
    teacher_board_ms: Optional[int]
    entries: int
    exits: int
    presence_intervals: list[list[int]]
    board_intervals: list[list[int]]
    entry_exit: list[EntryExitOut]
    occupancy: list[OccupancyOut]
    avg_students: float
    max_students: int
    heatmap: HeatmapOut
    # Optional so rows/tests predating the quality report still validate.
    data_quality: Optional[DataQualityOut] = None


class AnalysisResult(BaseModel):
    video: VideoInfoOut
    tracks: list[TrackOut]
    events: list[EventOut]
    analytics: AnalyticsOut


class DetectBoardResponse(BaseModel):
    """POST /detect-board response (feature contract).

    polygon: normalized 0-1 points, 4..12 of them, or null when nothing
    scored >= 0.25. confidence is the geometric board-likeness score of the
    best candidate (even when it fell below the polygon threshold).
    """

    polygon: Optional[
        Annotated[list[PolygonPoint], Field(min_length=4, max_length=12)]
    ] = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    method: str
    frame_ts_ms: int
