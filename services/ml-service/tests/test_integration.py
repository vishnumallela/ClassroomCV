"""End-to-end pipeline test over a synthetic detection stream (no model, no DB).

Exercises run_pipeline the way production runs it — detect, gate the static
classes, follow the teacher, derive the KPIs, write only her rows — with
detect_video monkeypatched so the shape and semantics are checked without a
GPU.
"""

import math

import pytest

from app import db, detector, jobs
from app.models import (
    CLASS_DOOR,
    CLASS_SCREEN,
    CLASS_TEACHER,
    AnalysisResult,
    Detection,
    VideoMeta,
)

VIDEO_ID = "11111111-2222-3333-4444-555555555555"
DURATION_MS = 60_000
BOARD = [[0.05, 0.05], [0.35, 0.05], [0.35, 0.3], [0.05, 0.3]]


@pytest.fixture(autouse=True)
def _passthrough_resolve(monkeypatch):
    """Bypass URL/local resolution: these tests mock detect_video with a fake
    /fake/classroom.mp4 path that no longer needs to exist on disk."""
    monkeypatch.setattr(detector, "resolve_video_source", lambda vp: (vp, False))


def _teacher_det(ts: int, conf: float = 0.9) -> Detection:
    """A teacher pacing sinusoidally across the left half of the room."""
    cx = 0.2 + 0.1 * math.sin(2 * math.pi * ts / 20_000.0)
    return Detection(
        video_ts_ms=ts,
        cls=CLASS_TEACHER,
        bbox={"x": round(cx - 0.06, 5), "y": 0.15, "w": 0.12, "h": 0.5},
        conf=conf,
    )


def _static_det(ts: int, cls: int, x: float) -> Detection:
    return Detection(
        video_ts_ms=ts,
        cls=cls,
        bbox={"x": x, "y": 0.1, "w": 0.2, "h": 0.18},
        conf=0.8,
    )


def _synthetic() -> tuple[VideoMeta, list[Detection]]:
    dets: list[Detection] = []
    for ts in range(0, DURATION_MS + 1, 200):
        dets.append(_teacher_det(ts))
        # The board, detected every frame inside its zone.
        dets.append(_static_det(ts, CLASS_SCREEN, 0.1))
        # A door detected far outside any configured door zone: the gate must
        # drop it rather than let it move the zone.
        dets.append(_static_det(ts, CLASS_DOOR, 0.75))
    meta = VideoMeta(duration_ms=DURATION_MS, fps=30.0, width=1280, height=720)
    return meta, dets


def test_full_pipeline_shape_and_semantics(monkeypatch):
    meta, dets = _synthetic()

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        assert video_path == "/fake/classroom.mp4"
        if progress_cb:
            progress_cb(0.5)
            progress_cb(1.0)
        return meta, dets

    written: dict = {}

    async def fake_replace(video_id, detections, **kwargs):
        written["video_id"] = video_id
        written["rows"] = [d for d in detections if d.track_no is not None]
        return len(written["rows"])

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    monkeypatch.setattr(db, "replace_detections", fake_replace)

    stages: list[tuple[str, float]] = []
    result = jobs.run_pipeline(
        VIDEO_ID,
        "/fake/classroom.mp4",
        5.0,
        [{"kind": "board", "polygon": BOARD}],
        progress_cb=lambda s, f: stages.append((s, f)),
        write_db=True,
    )

    parsed = AnalysisResult.model_validate(result)
    assert set(result.keys()) == {"video", "tracks", "events", "analytics", "proposed_zones"}

    assert parsed.video.duration_ms == DURATION_MS
    assert parsed.video.width == 1280 and parsed.video.height == 720

    # Exactly one track, and it is the teacher: the board and door are detected
    # but never become tracks, and there is no student class at all.
    assert len(parsed.tracks) == 1
    teacher = parsed.tracks[0]
    assert teacher.role == "teacher"
    assert teacher.track_no == 1
    assert teacher.role_confidence is not None
    assert (teacher.first_ms, teacher.last_ms) == (0, DURATION_MS)
    assert teacher.meta.detections == len(range(0, DURATION_MS + 1, 200))
    assert teacher.meta.coverage == 1.0
    assert teacher.meta.movement > 0.1  # she paces

    # Present throughout: one interval, enter at 0, no final exit.
    a = parsed.analytics
    assert a.presence_intervals == [[0, DURATION_MS]]
    assert a.teacher_present_ms == DURATION_MS
    assert a.entries == 1 and a.exits == 0
    assert [e.model_dump() for e in a.entry_exit] == [{"kind": "enter", "ts_ms": 0}]

    # Heatmap: every teacher detection lands one dwell sample.
    assert sum(a.heatmap.teacher) > 0
    assert len(a.heatmap.teacher) == a.heatmap.grid_w * a.heatmap.grid_h

    # Overlay: the walking teacher keeps interior polyline points; keyframes 2s apart.
    ov = teacher.meta.overlay
    assert ov is not None
    assert ov.polyline[0][0] == 0 and ov.polyline[-1][0] == DURATION_MS
    assert len(ov.polyline) > 2
    key_ts = [k[0] for k in ov.keyframes]
    assert key_ts[0] == 0 and all(b - a >= 2000 for a, b in zip(key_ts, key_ts[1:]))

    # Events reference the teacher identity and are time-sorted.
    assert any(e.kind == "enter" for e in parsed.events)
    assert all(e.track_no == 1 for e in parsed.events)
    ts_list = [e.video_ts_ms for e in parsed.events]
    assert ts_list == sorted(ts_list)

    # ONLY the teacher's rows are persisted — no board, door or student boxes
    # exist in the database to leak into an overlay.
    assert written["video_id"] == VIDEO_ID
    assert {r.track_no for r in written["rows"]} == {1}
    assert all(r.cls == CLASS_TEACHER for r in written["rows"])

    # Progress covers both stages; detection is scaled into 0..0.9.
    assert {s for s, _ in stages} == {"detecting", "deriving"}
    assert max(f for s, f in stages if s == "detecting") <= 0.9

    # The quality report describes HER, not a room of identities.
    dq = a.data_quality
    assert dq is not None
    assert dq.coverage == 1.0 and dq.breaks == 0
    assert dq.confidence.overall == "high"


def test_teleporting_teacher_box_is_rejected(monkeypatch):
    """A second 'teacher' appearing across the room mid-lesson is not her.

    Continuity beats confidence: the intruder is more confident, and must still
    lose, or a momentary false positive would steal the track and keep it.
    """
    dets: list[Detection] = []
    for ts in range(0, 20_001, 200):
        dets.append(_teacher_det(ts, conf=0.7))
        if 10_000 <= ts <= 12_000:
            intruder = _teacher_det(ts, conf=0.99)
            intruder.bbox = {"x": 0.9, "y": 0.6, "w": 0.08, "h": 0.2}
            dets.append(intruder)
    meta = VideoMeta(duration_ms=20_000, fps=30.0, width=1280, height=720)

    monkeypatch.setattr(
        detector, "detect_video", lambda *a, **k: (meta, dets)
    )
    result = jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
    parsed = AnalysisResult.model_validate(result)

    assert len(parsed.tracks) == 1
    # Her own trajectory never leaves the left half of the room, so the track
    # must not have jumped to x=0.9.
    ov = parsed.tracks[0].meta.overlay
    assert ov is not None
    assert max(p[1] for p in ov.polyline) < 0.5


def test_pipeline_short_empty_video_yields_valid_empty_result(monkeypatch):
    """A short (<= EMPTY_RESULT_GUARD_MS) clip with no detectable people is a
    legitimate empty result: run_pipeline returns a valid empty AnalysisResult
    (the zero-detection failure guard only fires on longer videos)."""

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        return VideoMeta(duration_ms=4_000, fps=30.0, width=640, height=480), []

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    result = jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
    parsed = AnalysisResult.model_validate(result)
    assert parsed.tracks == [] and parsed.events == []
    assert parsed.analytics.teacher_present_ms == 0
    assert sum(parsed.analytics.heatmap.teacher) == 0  # nobody detected


def test_pipeline_empty_over_5s_video_is_failure(monkeypatch):
    """Sanity guard: zero detections on a > EMPTY_RESULT_GUARD_MS video is a
    codec/model breakage, not a legitimately empty class — run_pipeline must
    raise instead of ingesting a 'done' result that silently zeroes every
    dashboard metric (the root of 'done with detections but zero tracks')."""

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        return VideoMeta(duration_ms=10_000, fps=30.0, width=640, height=480), []

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    with pytest.raises(RuntimeError, match="zero detections"):
        jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
