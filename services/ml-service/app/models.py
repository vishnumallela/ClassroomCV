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
# Detector classes
# --------------------------------------------------------------------------- #

# The five classes the RF-DETR checkpoint was fine-tuned on, in its own label
# order. They live here rather than in app/detector.py so the light consumers
# (db, tests) can name a class without importing torch. detector.py owns the
# load-time check that a checkpoint really declares them in this order.
CLASS_DOOR = 0
CLASS_SCREEN = 1
CLASS_TEACHER = 2
CLASS_POINTING = 3
CLASS_WRITING = 4
CLASS_NAMES: dict[int, str] = {
    CLASS_DOOR: "door",
    CLASS_SCREEN: "screen",
    CLASS_TEACHER: "teacher",
    CLASS_POINTING: "pointing",
    CLASS_WRITING: "writing",
}


# --------------------------------------------------------------------------- #
# In-memory dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """One RF-DETR detection on one sampled frame.

    `cls` is the detector's class id (app/detector.py CLASS_*), which is what
    made every other field on this dataclass unnecessary: posture, occlusion
    and body proportions existed only to work out which detected person was the
    adult, and the model now says so directly.

    `track_no` is set only on the teacher's accepted detections (see
    app/teacher.py); everything else carries None. A teacher-class detection
    with track_no None is not discarded — since migration 0014 it is persisted
    as an unattributed box, which is what a later attribution rule reads.
    """

    video_ts_ms: int
    cls: int
    bbox: dict  # {x, y, w, h} normalized 0-1, top-left based
    conf: float
    track_no: Optional[int] = None
    # Appearance descriptor (app/appearance.py), computed at detection time for
    # teacher-class boxes and persisted in detection_events.meta so that
    # attribution can be re-derived from stored rows. None when the box was
    # too small to read, or the row predates descriptors.
    app: Optional[list[float]] = None


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
    # The scheduled period as offsets into the video, when the lesson's
    # timetable and recording anchor are known. Attribution's primary rule
    # (docs/teacher-attribution-plan.md, Phase 3) is about who was in the room
    # during the PERIOD, not during the recording; without these it falls back
    # to the whole recording and caps its confidence at medium.
    period_start_ms: Optional[int] = None
    period_end_ms: Optional[int] = None


class RederiveRequest(BaseModel):
    video_id: str
    zones: list[ZoneIn] = Field(default_factory=list)
    period_start_ms: Optional[int] = None
    period_end_ms: Optional[int] = None


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
    # No 'merging' stage any more: there are no identity fragments to merge.
    stage: Literal["detecting", "deriving"]
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
    """How far she ranged, how much of the lesson she was found in, and the
    playback overlay. `coverage` and `mean_conf` are the two inputs behind
    role_confidence, exposed so the dashboard can explain the tier it shows."""

    movement: float
    detections: int
    coverage: float
    mean_conf: float
    overlay: Optional[TrackOverlayOut] = None


class TrackOut(BaseModel):
    # "teacher" is the attributed segment (track_no 1, exactly one). "adult" is
    # any other substantial teacher-class segment the tracker followed — a
    # second teacher during a handover, a student presenting at the front for
    # long enough to be a body rather than a flicker — numbered 2.. and handed
    # to attribution. Students as a class are still never detected, never
    # persisted and never drawn; an "adult" here is a body the detector called
    # a teacher and attribution has not (yet) called hers.
    track_no: int
    role: Literal["teacher", "adult"]
    role_confidence: Optional[float]
    first_ms: int
    last_ms: int
    meta: TrackMetaOut


class EventOut(BaseModel):
    # Every kind derive() can emit. This Literal is a hard response gate: a kind
    # produced upstream but missing here fails the WHOLE AnalysisResult at the
    # end of a paid GPU run — the pointing/writing launch lost a 17-minute
    # detection pass to exactly that. Extend it in the same commit as any new
    # events.append(kind=...).
    kind: Literal[
        "enter",
        "exit",
        "board_enter",
        "board_leave",
        "pointing_start",
        "pointing_end",
        "writing_start",
        "writing_end",
    ]
    video_ts_ms: int
    track_no: Optional[int]


class EntryExitOut(BaseModel):
    kind: Literal["enter", "exit"]
    ts_ms: int


class HeatmapOut(BaseModel):
    """Teacher dwell histogram over a grid_w x grid_h grid of the frame.

    Row-major flattened per-cell sample counts (grid_h rows, grid_w cols); at
    a fixed sample rate a cell's count is proportional to time spent there.
    Teacher-only since the 2026-08 KPI slimming. Empty list when no teacher.
    """

    grid_w: int
    grid_h: int
    teacher: list[int]


class QualityTiers(BaseModel):
    overall: Literal["high", "medium", "low"]
    coverage: Literal["high", "medium", "low"]
    continuity: Literal["high", "medium", "low"]
    teacher: Literal["high", "medium", "low"]
    # Whether following ONE person was the right thing to do at all. Defaulted
    # rather than required so a payload built before this signal existed still
    # validates; app/quality.py always sets it explicitly.
    attribution: Literal["high", "medium", "low"] = "high"


class AttributionCandidateOut(BaseModel):
    track_no: int
    first_ms: int
    last_ms: int
    present_ms: int
    in_period_ms: int
    handed_over: bool
    segments: int


class AttributionOut(BaseModel):
    """Which tracked adult the lesson assesses, how sure, and why (Phase 3).

    `confidence` is the tier the dashboard withholds on: anything but "high"
    keeps the punctuality numbers as Not Observed. `reason` is written to be
    read by the person looking at the card, not by code.
    """

    confidence: Literal["high", "medium", "low"]
    reason: str
    chosen_track_no: Optional[int]
    period_known: bool
    splits: int
    candidates: list[AttributionCandidateOut]


class DataQualityOut(BaseModel):
    """Additive per-run trust report (app/quality.py). Annotates, never alters,
    the derived numbers: how much of the lesson the teacher was actually found
    in, how broken her timeline is, how sure the detector was, and whether there
    was only one adult to follow — the trust inputs behind the teacher KPIs.

    EVERY FIELD app/quality.py EMITS MUST BE DECLARED HERE. Pydantic's default
    is extra="ignore", so a key added to the assess() dict without a field on
    this model is dropped silently by AnalysisResult.model_validate — it never
    reaches the API, the jsonb column or the dashboard, and nothing errors to
    say so. `data_quality` being schemaless jsonb in Postgres makes that trap
    easier to fall into, not harder: the database would have accepted it.
    """

    detections: int
    frames: int
    sampled_frames: int
    coverage: float
    mean_confidence: float
    breaks: int
    longest_gap_ms: int
    # More than one adult in the room. The fields above describe how well one
    # person was followed; these say whether "one person" was the right
    # question — see the attribution section of app/quality.py. Defaulted so an
    # older stored report still validates as "measured, one adult".
    multiple_adults_detected: bool = False
    max_simultaneous_adults: int = 1
    co_presence_ms: int = 0
    # Optional so rows and tests predating Phase 3 still validate. Absent means
    # "nothing decided" — the same rule as every other optional field here.
    attribution: Optional[AttributionOut] = None
    confidence: QualityTiers
    notes: list[str]


class AnalyticsOut(BaseModel):
    """The four teacher KPIs (entry/exit, board time, heatmap, pointing and
    writing) plus their supporting intervals for the timeline. Per-student
    occupancy analytics were removed in the 2026-08 KPI slimming.

    The three Optional ms fields carry a three-state answer: a number is a
    measurement, and None means the input for it was absent — no board zone
    configured, or (for the actions) a /rederive replaying teacher-only stored
    rows. 0 would claim it was measured and found to be zero.
    """

    teacher_present_ms: int
    teacher_board_ms: Optional[int]
    teacher_pointing_ms: Optional[int] = None
    teacher_writing_ms: Optional[int] = None
    entries: int
    exits: int
    presence_intervals: list[list[int]]
    board_intervals: list[list[int]]
    pointing_intervals: list[list[int]] = []
    writing_intervals: list[list[int]] = []
    entry_exit: list[EntryExitOut]
    heatmap: HeatmapOut
    # Optional so rows/tests predating the quality report still validate.
    data_quality: Optional[DataQualityOut] = None


class ProposedZoneOut(BaseModel):
    """A zone the analysis placed for itself, for the caller to persist.

    Emitted only for a kind that had no zone configured, and only above
    zones.AUTO_ACCEPT. The analysis has already USED it, so a caller that drops
    these on the floor keeps the KPIs but loses the polygon and will re-propose
    it on the next run.
    """

    kind: Literal["board", "door"]
    polygon: list[PolygonPoint]
    confidence: float
    method: str
    frame_ts_ms: int


class AnalysisResult(BaseModel):
    video: VideoInfoOut
    tracks: list[TrackOut]
    events: list[EventOut]
    analytics: AnalyticsOut
    proposed_zones: list[ProposedZoneOut] = []


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
