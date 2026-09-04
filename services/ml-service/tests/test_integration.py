"""End-to-end pipeline test over a synthetic detection stream (no model, no DB).

Exercises run_pipeline the way production runs it — detect, gate the static
classes, follow the teacher, derive the KPIs, write the teacher-class rows —
with detect_video monkeypatched so the shape and semantics are checked without
a GPU.

test_full_pipeline_shape_and_semantics is the SINGLE-TEACHER BASELINE. The
multi-adult work must not move it: a lesson with one adult in the room has to
produce the numbers it always produced, and the assertions below are what says
so at the unit level. (The other half of that guarantee — a real 37-minute
lesson at 94.5% coverage — needs a GPU run, because detection_events was wiped
for the older videos and there is no stored fixture to replay.)
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
    assert [e.model_dump() for e in a.entry_exit] == [{"kind": "enter", "ts_ms": 0, "method": "start"}]

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

    # One adult, measured and said so. "high" attribution and a False flag are
    # different claims from a missing field, which is what a lesson analysed
    # before this check carries — the dashboard must be able to tell them apart.
    assert dq.multiple_adults_detected is False
    assert dq.max_simultaneous_adults == 1
    assert dq.co_presence_ms == 0
    assert dq.confidence.attribution == "high"


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


# --------------------------------------------------------------------------- #
# Two adults in the room
# --------------------------------------------------------------------------- #


def _handover() -> tuple[VideoMeta, list[Detection]]:
    """The recording that started this: period 3 filmed while period 2 is still
    packing up.

    Both adults are detected at 0.7-0.86 and stand close enough that either box
    is reachable from the previous frame, so _pick_candidate settles each
    instant on confidence alone and the "teacher track" swaps between them. The
    outgoing teacher is present from the first frame and leaves a third of the
    way in; the incoming one arrives before she goes.
    """
    dets: list[Detection] = []
    for i, ts in enumerate(range(0, DURATION_MS + 1, 200)):
        dets.append(_static_det(ts, CLASS_SCREEN, 0.1))
        # Incoming teacher: arrives at 15s and stays.
        if ts >= 15_000:
            dets.append(_teacher_det(ts, conf=0.86 if i % 2 else 0.80))
        # Outgoing teacher: present from the start, leaves at 50s. Offset a
        # little in x — close, the way two people at the front of a room are.
        if ts <= 50_000:
            out = _teacher_det(ts, conf=0.80 if i % 2 else 0.86)
            out.bbox = {**out.bbox, "x": round(out.bbox["x"] + 0.08, 5)}
            dets.append(out)
    meta = VideoMeta(duration_ms=DURATION_MS, fps=30.0, width=1280, height=720)
    return meta, dets


def test_handover_is_flagged_and_survives_the_response_model(monkeypatch):
    """Two adults for 35 of 60 seconds: the run must SAY so, all the way out.

    The "all the way out" is the point of asserting on the parsed model rather
    than the raw dict. Pydantic defaults to extra="ignore", so a field added to
    quality.assess() without a matching one on DataQualityOut is dropped
    silently by AnalysisResult.model_validate — it never reaches the API, the
    jsonb column or the dashboard, and nothing raises to say so. The database
    column is schemaless jsonb, so it would have accepted the field happily;
    this assertion is the only thing standing between that and a flag that
    exists everywhere except where it is read.
    """
    meta, dets = _handover()
    monkeypatch.setattr(detector, "detect_video", lambda *a, **k: (meta, dets))
    result = jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
    parsed = AnalysisResult.model_validate(result)

    dq = parsed.analytics.data_quality
    assert dq is not None
    assert dq.multiple_adults_detected is True
    assert dq.max_simultaneous_adults == 2
    assert dq.co_presence_ms >= 30_000
    assert dq.confidence.attribution == "medium"  # Phase 3 decided, on a 10 s tail: still withheld
    assert dq.confidence.overall == "medium"
    assert any("adults" in n.lower() for n in dq.notes)


def test_the_handover_is_two_bodies_not_one_blend(monkeypatch):
    """Phase 2: the two adults come out as two tracks, and neither is a blend.

    Before the segment tracker this fixture produced ONE track with coverage
    1.0, zero breaks and high confidence — there was always *a* box — while its
    arrival time belonged to the teacher who was leaving. That perfect-looking
    result was the bug. Now the primary is one real person with that person's
    real gaps, so its coverage is honestly below 1.0, and the second adult is
    its own numbered track for attribution to weigh.
    """
    meta, dets = _handover()
    monkeypatch.setattr(detector, "detect_video", lambda *a, **k: (meta, dets))
    result = jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False)
    parsed = AnalysisResult.model_validate(result)

    assert [t.role for t in parsed.tracks] == ["teacher", "adult"]
    teacher, adult = parsed.tracks
    assert (teacher.track_no, adult.track_no) == (1, 2)
    # Each track is one body. The fixture offsets the outgoing teacher by
    # exactly +0.08 in x at every shared instant, so if the lanes never mix,
    # the per-instant difference between the two tracks is a constant.
    by_ts: dict = {}
    for d in dets:
        if d.cls == CLASS_TEACHER and d.track_no in (1, 2):
            by_ts.setdefault(d.video_ts_ms, {})[d.track_no] = d.bbox["x"]
    diffs = {round(x[1] - x[2], 3) for x in by_ts.values() if len(x) == 2}
    assert len(diffs) == 1, f"lanes mixed: {sorted(diffs)[:6]}"
    # The blend's false perfection is gone: coverage is the primary's own.
    assert teacher.meta.coverage < 1.0
    # The refusal still stands until attribution decides between them.
    dq = parsed.analytics.data_quality
    assert dq is not None and dq.multiple_adults_detected is True


def test_every_teacher_box_reaches_the_writer_with_its_segment(monkeypatch):
    """Phase 1 + 2 end to end: both adults' boxes reach db.replace_detections,
    each carrying its own segment number.

    Asserted on what run_pipeline HANDS the writer rather than on the writer's
    own filter (test_db.py covers that), because the original defect was
    upstream of both: the single chain stamped only the winners onto a list the
    writer then filtered by track_no, and the second adult was gone before any
    storage decision was made.
    """
    meta, dets = _handover()
    monkeypatch.setattr(detector, "detect_video", lambda *a, **k: (meta, dets))

    handed: dict = {}

    async def fake_replace(video_id, detections, **kwargs):
        handed["all"] = list(detections)
        return len(detections)

    monkeypatch.setattr(db, "replace_detections", fake_replace)
    jobs.run_pipeline(VIDEO_ID, "/fake.mp4", 5.0, [], write_db=True)

    teacher_boxes = [d for d in handed["all"] if d.cls == CLASS_TEACHER]
    by_no: dict = {}
    for d in teacher_boxes:
        by_no.setdefault(d.track_no, []).append(d)

    assert set(by_no) == {1, 2}, "both adults must reach the writer, each numbered"
    assert len(by_no[1]) + len(by_no[2]) == len(teacher_boxes)
    # And the numbering means something: the two are different bodies.
    x1 = sum(d.bbox["x"] for d in by_no[1]) / len(by_no[1])
    x2 = sum(d.bbox["x"] for d in by_no[2]) / len(by_no[2])
    assert abs(x1 - x2) > 0.05


def test_attribution_picks_the_adult_who_stayed_and_reports_why(monkeypatch):
    """Phase 3 through run_pipeline: the handover fixture, with the period.

    The outgoing adult (0-50 s) has MORE presence; the incoming one (15-60 s)
    is the one who remains. Attribution must pick the second, say so in words,
    and — because only ten seconds of her alone follow the handover — call it
    medium, which keeps the punctuality numbers withheld downstream. No
    descriptors in this fixture: the two lanes never cross, so continuity
    alone separates them, and the test pins that the rest of the machinery
    (people -> track 1 -> data_quality.attribution -> the Pydantic gate) holds.
    """
    meta, dets = _handover()
    monkeypatch.setattr(detector, "detect_video", lambda *a, **k: (meta, dets))
    result = jobs.run_pipeline(
        VIDEO_ID, "/fake.mp4", 5.0, [], write_db=False, period_ms=(0, 60_000)
    )
    parsed = AnalysisResult.model_validate(result)

    teacher = parsed.tracks[0]
    assert teacher.role == "teacher" and teacher.track_no == 1
    assert 14_000 <= teacher.first_ms <= 16_000, "the adult who ARRIVED is the teacher"
    assert teacher.last_ms >= 59_000
    assert [t.role for t in parsed.tracks[1:]] == ["adult"]
    assert parsed.tracks[1].first_ms == 0

    dq = parsed.analytics.data_quality
    assert dq is not None and dq.attribution is not None
    assert dq.attribution.confidence == "medium"
    assert dq.attribution.chosen_track_no == 1
    assert dq.attribution.period_known is True
    assert "left while this one remained" in dq.attribution.reason
    assert dq.confidence.attribution == "medium"
    assert dq.multiple_adults_detected is True
    # The teacher's own timeline, not a blend: presence starts when SHE arrived.
    assert 14_000 <= parsed.analytics.presence_intervals[0][0] <= 16_000
    # And the outgoing adult's boxes reach the writer numbered 2, not dropped.
    assert {d.track_no for d in dets if d.cls == CLASS_TEACHER} >= {1, 2}
