"""Integration-style test: full pipeline on synthetic detections.

Monkeypatches the detector to emit synthetic detections for a fake 60s video
and a stub DB writer, then asserts the full AnalysisResult shape (validated
against the pydantic model). No GPU, no DB, no video file needed.
"""

import math

import numpy as np

from app import db, detector, jobs
from app.models import AnalysisResult, Detection, VideoMeta

VIDEO_ID = "11111111-2222-3333-4444-555555555555"
DURATION_MS = 60_000


def _hist(bin_idx: int) -> np.ndarray:
    h = np.zeros(960, dtype=np.float32)
    h[bin_idx] = 1.0
    return h


def _teacher_det(ts: int, raw_id: int) -> Detection:
    cx = 0.2 + 0.1 * math.sin(2 * math.pi * ts / 20_000.0)
    return Detection(
        video_ts_ms=ts,
        raw_track_id=raw_id,
        bbox={"x": round(cx - 0.06, 5), "y": 0.15, "w": 0.12, "h": 0.5},
        conf=0.9,
        standing=True,
        back_to_camera=(ts % 400 == 0),
    )


def _student_det(ts: int) -> Detection:
    return Detection(
        video_ts_ms=ts,
        raw_track_id=2,
        bbox={"x": 0.7, "y": 0.55, "w": 0.1, "h": 0.2},
        conf=0.8,
        standing=False,
        back_to_camera=False,
    )


def _synthetic() -> tuple[VideoMeta, list[Detection], dict]:
    dets: list[Detection] = []
    # teacher fragmented into raw tracks 1 (0..20s) and 7 (22..60s), same torso hist
    for ts in range(0, 20_001, 200):
        dets.append(_teacher_det(ts, 1))
    for ts in range(22_000, DURATION_MS + 1, 200):
        dets.append(_teacher_det(ts, 7))
    # seated student raw track 2, different hist
    for ts in range(5_000, 58_001, 500):
        dets.append(_student_det(ts))
    hists = {1: [_hist(1)] * 3, 7: [_hist(1)] * 3, 2: [_hist(500)] * 3}
    meta = VideoMeta(duration_ms=DURATION_MS, fps=30.0, width=1280, height=720)
    return meta, dets, hists


def test_full_pipeline_shape_and_semantics(monkeypatch):
    meta, dets, hists = _synthetic()

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        assert video_path == "/fake/classroom.mp4"
        if progress_cb:
            progress_cb(0.5)
            progress_cb(1.0)
        return meta, dets, hists

    written: dict = {}

    async def fake_replace(video_id, detections, **kwargs):
        written["video_id"] = video_id
        written["rows"] = list(detections)
        return len(detections)

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    monkeypatch.setattr(db, "replace_detections", fake_replace)

    stages: list[tuple[str, float]] = []
    board = [[0.05, 0.05], [0.35, 0.05], [0.35, 0.3], [0.05, 0.3]]
    result = jobs.run_pipeline(
        VIDEO_ID,
        "/fake/classroom.mp4",
        5.0,
        [{"kind": "board", "polygon": board}],
        progress_cb=lambda s, f: stages.append((s, f)),
        write_db=True,
    )

    # exact SPEC shape
    parsed = AnalysisResult.model_validate(result)
    assert set(result.keys()) == {"video", "tracks", "events", "analytics"}

    # video block
    assert parsed.video.duration_ms == DURATION_MS
    assert parsed.video.width == 1280 and parsed.video.height == 720

    # merge: raw tracks 1 + 7 collapse into one identity, student stays separate
    assert len(parsed.tracks) == 2
    teachers = [t for t in parsed.tracks if t.role == "teacher"]
    students = [t for t in parsed.tracks if t.role == "student"]
    assert len(teachers) == 1 and len(students) == 1
    assert teachers[0].track_no == 1  # first appearance -> track_no 1
    assert teachers[0].meta.raw_track_ids == [1, 7]
    assert teachers[0].role_confidence is not None
    assert teachers[0].meta.standing_ratio > 0.9
    assert (teachers[0].first_ms, teachers[0].last_ms) == (0, DURATION_MS)

    # teacher present throughout (2s fragment gap < 5s threshold): one interval,
    # enter at 0, no final exit (still present at video end)
    a = parsed.analytics
    assert a.presence_intervals == [[0, DURATION_MS]]
    assert a.teacher_present_ms == DURATION_MS
    assert a.entries == 1 and a.exits == 0
    assert [e.model_dump() for e in a.entry_exit] == [{"kind": "enter", "ts_ms": 0}]

    # occupancy: 12 buckets of 5000ms, one student mid-video
    assert len(a.occupancy) == 12
    assert a.max_students == 1
    assert 0.0 < a.avg_students <= 1.0
    assert a.occupancy[1].students == 1 and a.occupancy[1].teacher is True

    # permanent overlay tier: the walking teacher keeps interior polyline
    # points, the static student collapses to endpoints; keyframes >= 2s apart
    t_ov = teachers[0].meta.overlay
    assert t_ov is not None
    assert t_ov.polyline[0][0] == 0 and t_ov.polyline[-1][0] == DURATION_MS
    assert len(t_ov.polyline) > 2
    key_ts = [k[0] for k in t_ov.keyframes]
    assert key_ts[0] == 0 and all(b - a >= 2000 for a, b in zip(key_ts, key_ts[1:]))
    s_ov = students[0].meta.overlay
    assert s_ov is not None and len(s_ov.polyline) == 2

    # events reference the teacher identity and are time-sorted
    assert any(e.kind == "enter" for e in parsed.events)
    assert all(e.track_no == 1 for e in parsed.events)
    ts_list = [e.video_ts_ms for e in parsed.events]
    assert ts_list == sorted(ts_list)

    # DB writer got post-merge identities
    assert written["video_id"] == VIDEO_ID
    assert len(written["rows"]) == len(dets)
    assert {r.track_no for r in written["rows"]} == {1, 2}

    # progress staging covered all three stages, detection scaled into 0..0.8
    seen_stages = {s for s, _ in stages}
    assert seen_stages == {"detecting", "merging", "deriving"}
    assert max(f for s, f in stages if s == "detecting") <= 0.8


def test_pipeline_short_empty_video_yields_valid_empty_result(monkeypatch):
    """A short (<= EMPTY_RESULT_GUARD_MS) clip with no detectable people is a
    legitimate empty result: run_pipeline returns a valid empty AnalysisResult
    (the zero-detection failure guard only fires on longer videos)."""

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        return VideoMeta(duration_ms=4_000, fps=30.0, width=640, height=480), [], {}

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    result = jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
    parsed = AnalysisResult.model_validate(result)
    assert parsed.tracks == [] and parsed.events == []
    assert parsed.analytics.teacher_present_ms == 0
    assert parsed.analytics.max_students == 0
    assert len(parsed.analytics.occupancy) == 1  # 4s -> one 5000ms bucket


def test_pipeline_empty_over_5s_video_is_failure(monkeypatch):
    """Sanity guard: zero detections on a > EMPTY_RESULT_GUARD_MS video is a
    codec/model breakage, not a legitimately empty class — run_pipeline must
    raise instead of ingesting a 'done' result that silently zeroes every
    dashboard metric (the root of 'done with detections but zero tracks')."""
    import pytest

    def fake_detect(video_path, sample_fps=5.0, progress_cb=None):
        return VideoMeta(duration_ms=10_000, fps=30.0, width=640, height=480), [], {}

    monkeypatch.setattr(detector, "detect_video", fake_detect)
    with pytest.raises(RuntimeError, match="zero detections"):
        jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
