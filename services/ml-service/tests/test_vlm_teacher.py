"""Unit tests for the vision-LLM teacher-ID fallback.

Offline: the network call (_ask_point) and video decode (cv2.VideoCapture /
detector.resolve_video_source) are monkeypatched. Only the pure parsing helpers
and the vote/gate logic in identify_teacher are under test.
"""

from app import vlm_teacher as V
from app.models import Detection


def _det(ts, track_no, x, y, w=0.1, h=0.3):
    return Detection(ts, track_no, {"x": x, "y": y, "w": w, "h": h}, 0.9, True, False, track_no=track_no)


# --------------------------------------------------------------------------- #
# Pure parsing: code-fence stripping + point extraction
# --------------------------------------------------------------------------- #


def test_parse_point_plain_json():
    assert V._parse_point('{"x": 0.4, "y": 0.6}') == (0.4, 0.6)


def test_parse_point_strips_markdown_code_fence():
    # Gemini sometimes wraps the answer in ```json ... ``` — must still parse.
    text = '```json\n{"x": 0.57, "y": 0.61}\n```'
    assert V._parse_point(text) == (0.57, 0.61)


def test_parse_point_null_xy_returns_none():
    assert V._parse_point('{"x": null, "y": null}') is None


def test_parse_point_no_json_returns_none():
    assert V._parse_point("I cannot see a teacher in this image.") is None


def test_parse_answer_distinguishes_absent_from_failure():
    # explicit null -> "absent" (the model SAW no teacher: a refute signal),
    # a point -> "point", garbage -> "none" (no signal). This split is what lets
    # claim-verify count "no teacher in this frame" against a chained student.
    assert V._parse_answer('{"x": null, "y": null}') == ("absent", None)
    assert V._parse_answer('{"x": 0.4, "y": 0.6}') == ("point", (0.4, 0.6))
    assert V._parse_answer("no teacher here, sorry") == ("none", None)


def test_parse_point_picks_last_json_object_if_several():
    # A reasoning-style response may echo the schema before the real answer.
    text = 'Example: {"x": 0.1, "y": 0.1}\nAnswer: {"x": 0.8, "y": 0.2}'
    assert V._parse_point(text) == (0.8, 0.2)


# --------------------------------------------------------------------------- #
# identify_teacher: fail-closed gates that don't need network/video
# --------------------------------------------------------------------------- #


def test_disabled_flag_returns_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("VLM_TEACHER_FALLBACK", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = {1: [_det(0, 1, 0.5, 0.5)]}
        assert V.identify_teacher("video.mp4", dets, 60_000) is None
    finally:
        get_settings.cache_clear()


def test_missing_key_returns_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = {1: [_det(0, 1, 0.5, 0.5)]}
        assert V.identify_teacher("video.mp4", dets, 60_000) is None
    finally:
        get_settings.cache_clear()


def test_no_detections_returns_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert V.identify_teacher("video.mp4", {}, 60_000) is None
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# identify_teacher: vote logic, with the network + video layers monkeypatched
# --------------------------------------------------------------------------- #


class _FakeCapture:
    """Stands in for cv2.VideoCapture: read() always succeeds with a 1x1 frame."""

    def set(self, *_args, **_kwargs):
        pass

    def read(self):
        import numpy as np

        return True, np.zeros((4, 4, 3), dtype="uint8")

    def release(self):
        pass


def _wire_offline(monkeypatch, points):
    """points: list of (x, y) or None, one per VLM call.

    Patches both the single-shot helper (identify_teacher) and the per-frame
    helper the concurrent batch calls (locate_teacher). Frames are asked in
    parallel, so which frame receives which point is not fixed — these fixtures
    are written so the tally does not depend on the pairing.
    """
    monkeypatch.setattr(V.detector, "resolve_video_source", lambda p: (p, False))
    monkeypatch.setattr(V.cv2, "VideoCapture", lambda _path: _FakeCapture())
    it = iter(points)
    monkeypatch.setattr(V, "_ask_point", lambda *a, **kw: next(it, None))

    def _answer(*_a, **_kw):
        pt = next(it, None)
        return ("point", pt) if pt is not None else ("fail", None)

    monkeypatch.setattr(V, "_ask_answer", _answer)


def _wire_offline_answers(monkeypatch, answers):
    """answers: list of (status, pt) for _ask_answer, in order.
    status in {"point","absent","fail","none"}; pt is (x,y) or None."""
    monkeypatch.setattr(V.detector, "resolve_video_source", lambda p: (p, False))
    monkeypatch.setattr(V.cv2, "VideoCapture", lambda _path: _FakeCapture())
    it = iter(answers)
    monkeypatch.setattr(V, "_ask_answer", lambda *a, **kw: next(it, ("fail", None)))


def _six_frame_dets(track_positions: dict[int, tuple[float, float]]):
    """One identity per track, present across the whole 100s window so every
    sampled timestamp has a centre for every track."""
    dets_by_track = {}
    for track_no, (x, y) in track_positions.items():
        dets_by_track[track_no] = [_det(ts, track_no, x, y) for ts in range(0, 100_000, 5_000)]
    return dets_by_track


def test_majority_vote_picks_the_winning_track(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("VLM_FRAMES", "6")
    monkeypatch.setenv("VLM_MIN_VOTES", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _six_frame_dets({1: (0.2, 0.2), 2: (0.8, 0.8)})
        # 4 votes near track 2, 2 votes near track 1 -> track 2 wins.
        _wire_offline(monkeypatch, [(0.79, 0.81), (0.81, 0.79), (0.2, 0.2), (0.8, 0.8), (0.78, 0.78), (0.19, 0.21)])
        result = V.identify_teacher("video.mp4", dets, 100_000)
        assert result is not None
        track_no, confidence, votes = result
        assert track_no == 2
        assert votes[2] == 4
        assert confidence == round(4 / 6, 3)
    finally:
        get_settings.cache_clear()


def test_inconclusive_vote_returns_none(monkeypatch):
    """Every frame answers a DIFFERENT track: no winner clears vlm_min_votes."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("VLM_FRAMES", "3")
    monkeypatch.setenv("VLM_MIN_VOTES", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _six_frame_dets({1: (0.1, 0.1), 2: (0.5, 0.5), 3: (0.9, 0.9)})
        _wire_offline(monkeypatch, [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])
        assert V.identify_teacher("video.mp4", dets, 100_000) is None
    finally:
        get_settings.cache_clear()


def test_all_frames_fail_returns_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _six_frame_dets({1: (0.5, 0.5)})
        _wire_offline(monkeypatch, [None, None, None, None, None, None])
        assert V.identify_teacher("video.mp4", dets, 100_000) is None
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Point-in-bbox mapping + locate_teacher (VLM verify/override)
# --------------------------------------------------------------------------- #


def test_pool_fragments_groups_her_disjoint_fragments():
    # She is tracked as three ids that never share a frame -> one person, so the
    # votes are summed instead of competing (measured: 3/12 becomes 6/12).
    from collections import Counter

    votes = Counter({12: 3, 573: 2, 652: 1})
    frames = {12: set(range(0, 100)), 573: set(range(200, 300)), 652: set(range(400, 500))}
    assert sorted(V.pool_fragments(votes, frames)) == [12, 573, 652]


def test_pool_fragments_refuses_co_present_fragments():
    # Two boxes in the same frames are two people, however the votes fell.
    from collections import Counter

    votes = Counter({12: 4, 99: 3})
    frames = {12: set(range(0, 100)), 99: set(range(50, 150))}
    assert V.pool_fragments(votes, frames) == [12]


def test_pool_fragments_tolerates_a_handoff_overlap():
    # A dying id and its successor can share a frame or two; that is tracker
    # noise at a handoff, not evidence of two people.
    from collections import Counter

    votes = Counter({12: 3, 13: 2})
    frames = {12: set(range(0, 101)), 13: set(range(100, 200))}
    assert sorted(V.pool_fragments(votes, frames)) == [12, 13]


def test_budget_stops_asking_once_spent(monkeypatch):
    # A long, badly fragmented lesson must not run past the transport limit.
    # Once the budget is gone the remaining frames report "fail", which every
    # caller reads as no signal, so the derive degrades instead of timing out.
    calls = {"n": 0}

    def _count(*_a, **_kw):
        calls["n"] += 1
        return ("point", (0.5, 0.5))

    monkeypatch.setattr(V, "_ask_answer", _count)
    V.begin_derive(5)
    first = V._ask_batch("k", "m", [(i, "b64") for i in range(4)])
    second = V._ask_batch("k", "m", [(i, "b64") for i in range(10, 14)])
    V.begin_derive(0)  # leave the module unbudgeted for other tests
    assert calls["n"] == 5, "only the budgeted calls are made"
    assert len(first) == 4 and len(second) == 4, "every frame still gets an entry"
    unasked = [v for v in second.values() if v == ("fail", None)]
    assert len(unasked) == 3, "the unasked frames read as no signal"


def test_budget_of_zero_is_unlimited(monkeypatch):
    monkeypatch.setattr(V, "_ask_answer", lambda *a, **kw: ("absent", None))
    V.begin_derive(0)
    out = V._ask_batch("k", "m", [(i, "b64") for i in range(9)])
    assert len(out) == 9
    assert all(v == ("absent", None) for v in out.values())


def test_point_to_track_picks_containing_box():
    boxes = {1: (0.0, 0.0, 0.2, 0.2), 2: (0.4, 0.4, 0.3, 0.5)}
    assert V._point_to_track(0.5, 0.6, boxes) == 2  # only box 2 contains (0.5,0.6)


def test_point_to_track_none_when_point_in_empty_space():
    boxes = {1: (0.0, 0.0, 0.2, 0.2)}
    assert V._point_to_track(0.9, 0.9, boxes) is None  # ambiguous -> no vote


def test_point_to_track_prefers_adult_over_foreground_kid_on_overlap():
    # A tall standing box (teacher) and a shorter foreground box (kid) both contain
    # the torso point; the point sits more centrally in her box, so she wins.
    teacher = (0.40, 0.20, 0.15, 0.60)
    kid = (0.35, 0.45, 0.25, 0.30)
    assert V._point_to_track(0.475, 0.55, {1: teacher, 2: kid}) == 1


def test_locate_teacher_votes_by_point_in_bbox(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("VLM_MIN_VOTES", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        def span(track_no, x):
            return [
                Detection(ts, track_no, {"x": x, "y": 0.3, "w": 0.2, "h": 0.4}, 0.9, True, False, track_no=track_no)
                for ts in range(0, 100_000, 5_000)
            ]

        dets = {1: span(1, 0.1), 2: span(2, 0.7)}  # box1 left, box2 right (disjoint)
        # 3 points land in box 2, 1 in box 1 -> track 2 wins by point-in-bbox.
        _wire_offline(monkeypatch, [(0.8, 0.5), (0.8, 0.5), (0.8, 0.5), (0.15, 0.5)])
        res = V.locate_teacher("video.mp4", dets, 100_000, n_frames=4)
        assert res.track == 2
        assert res.answered == 4
        assert res.votes[2] == 3 and res.votes[1] == 1
    finally:
        get_settings.cache_clear()


def test_locate_teacher_no_vote_when_points_land_in_empty_space(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = {
            1: [Detection(ts, 1, {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.4}, 0.9, True, False, track_no=1) for ts in range(0, 100_000, 5_000)]
        }
        # every point is far from the only box -> no vote -> no teacher, but the VLM
        # DID answer all 4 frames: track None yet answered==4 (the reject signal).
        _wire_offline(monkeypatch, [(0.9, 0.9), (0.95, 0.9), (0.9, 0.95), (0.99, 0.99)])
        res = V.locate_teacher("video.mp4", dets, 100_000, n_frames=4)
        assert res.track is None
        assert res.answered == 4
        assert res.votes.get(1, 0) == 0  # box 1 never got a point -> caller REJECTs it
    finally:
        get_settings.cache_clear()


def test_locate_teacher_answered_zero_when_every_call_fails(monkeypatch):
    """All _ask_point return None (e.g. HTTP errors): answered==0 so the caller
    KEEPS the geometric pick instead of demoting a possibly-correct teacher."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = {
            1: [Detection(ts, 1, {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.4}, 0.9, True, False, track_no=1) for ts in range(0, 100_000, 5_000)]
        }
        _wire_offline(monkeypatch, [None, None, None, None])
        res = V.locate_teacher("video.mp4", dets, 100_000, n_frames=4)
        assert res.track is None
        assert res.answered == 0  # nothing answered -> not a rejection, a failure
        assert res.votes == {}
    finally:
        get_settings.cache_clear()


def test_locate_teacher_answered_when_no_winner_clears_threshold(monkeypatch):
    """Votes scatter below vlm_min_votes: track is None but answered counts every
    frame and votes are reported, so the caller can see the geometric pick got 0."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("VLM_MIN_VOTES", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        def span(track_no, x):
            return [
                Detection(ts, track_no, {"x": x, "y": 0.3, "w": 0.15, "h": 0.4}, 0.9, True, False, track_no=track_no)
                for ts in range(0, 100_000, 5_000)
            ]

        # geometric pick would be track 1; VLM points once at box 2, once at box 3,
        # twice into empty space -> no winner, track 1 gets 0 votes, answered==4.
        dets = {1: span(1, 0.1), 2: span(2, 0.45), 3: span(3, 0.8)}
        _wire_offline(monkeypatch, [(0.5, 0.5), (0.85, 0.5), (0.99, 0.01), (0.99, 0.99)])
        res = V.locate_teacher("video.mp4", dets, 100_000, n_frames=4)
        assert res.track is None
        assert res.answered == 4
        assert res.votes.get(1, 0) == 0  # the geometric pick -> REJECT territory
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# vlm_supports_span: VLM-veto of a teacher-chain claim (point-in-fragment-box)
# --------------------------------------------------------------------------- #


def _span_dets(x=0.1, y=0.3, w=0.2, h=0.4):
    """One fragment's detections across a 100s span at ~5fps (200ms), so the
    evenly-spaced sample targets always land within _TS_TOLERANCE_MS of a det."""
    return [
        Detection(ts, 1, {"x": x, "y": y, "w": w, "h": h}, 0.9, True, False, track_no=1)
        for ts in range(0, 100_000, 200)
    ]


def test_vlm_supports_span_true_when_points_land_inside(monkeypatch):
    """The VLM points INTO the fragment's box every frame -> it IS the teacher."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets(0.1, 0.3, 0.2, 0.4)  # box x in [0.1,0.3], y in [0.3,0.7]
        _wire_offline_answers(monkeypatch, [("point", (0.15, 0.5))] * 6)  # all inside
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is True
    finally:
        get_settings.cache_clear()


def test_vlm_supports_span_false_when_teacher_absent_across_span(monkeypatch):
    """The raw-79 chimera shape: the VLM reports "no teacher" for most of the span
    (she hasn't entered yet), and the one 'inside' hit is the handoff frame where
    the student's box has drifted onto her. Support fraction 1/6 -> VETO."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets(0.1, 0.3, 0.2, 0.4)
        answers = [("absent", None)] * 5 + [("point", (0.15, 0.5))]  # 5 absent, 1 inside
        _wire_offline_answers(monkeypatch, answers)
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is False
    finally:
        get_settings.cache_clear()


def test_vlm_supports_span_false_when_points_land_outside(monkeypatch):
    """The teacher is consistently somewhere else -> every point refutes -> VETO."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets(0.1, 0.3, 0.2, 0.4)
        _wire_offline_answers(monkeypatch, [("point", (0.9, 0.9))] * 6)  # all OUTSIDE
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is False
    finally:
        get_settings.cache_clear()


def test_vlm_supports_span_none_when_vlm_fails(monkeypatch):
    """Every call fails (no usable signal) -> None, so the caller KEEPS the claim
    (a network failure must never drop a genuine crouch-steal reclaim)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets()
        _wire_offline_answers(monkeypatch, [("fail", None)] * 6)
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is None
    finally:
        get_settings.cache_clear()


def test_vlm_supports_span_none_on_too_few_answers(monkeypatch):
    """Only 2 usable answers (1 in, 1 out): below the min-answered floor -> None
    (keep the claim; don't veto on thin evidence)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets(0.1, 0.3, 0.2, 0.4)
        answers = [("point", (0.15, 0.5)), ("point", (0.9, 0.9))] + [("fail", None)] * 4
        _wire_offline_answers(monkeypatch, answers)
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is None
    finally:
        get_settings.cache_clear()


def test_vlm_supports_span_true_when_mostly_inside_with_a_few_misses(monkeypatch):
    """A genuine reclaim isn't sunk by a couple of drift frames: 4 inside, 2 out
    -> support 0.67 >= floor -> keep."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dets = _span_dets(0.1, 0.3, 0.2, 0.4)
        answers = [("point", (0.15, 0.5))] * 4 + [("point", (0.95, 0.95)), ("absent", None)]
        _wire_offline_answers(monkeypatch, answers)
        assert V.vlm_supports_span("v.mp4", dets, n_frames=6) is True
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# backfill_teacher_entry: reclaim the teacher's pre-chain entry/seated fragments
# --------------------------------------------------------------------------- #


def _backfill_tracks(teacher_first=100_000):
    """Teacher track 39 from teacher_first on (first det centre ~(0.70,0.50)); her
    seated fragment (track 11, centre ~(0.72,0.55)) exists in the entry window just
    before it; track 7 is a real student far to the left (centre ~(0.15,0.55))."""
    return {
        39: [_det(ts, 39, 0.65, 0.35, 0.10, 0.30) for ts in range(teacher_first, 120_000, 200)],
        11: [_det(ts, 11, 0.67, 0.40, 0.10, 0.30) for ts in range(88_000, teacher_first, 200)],
        7: [_det(ts, 7, 0.10, 0.40, 0.10, 0.30) for ts in range(0, 120_000, 200)],
    }


def test_backfill_tracks_continuous_trajectory(monkeypatch):
    """The VLM points at her seated fragment (track 11) along a smooth path, then
    reports no teacher (she hadn't entered) -> reclaim track 11's entry dets."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dbt = _backfill_tracks()
        # points land on track 11's centre (0.72,0.55), continuous with the seed.
        _wire_offline_answers(monkeypatch, [("point", (0.72, 0.55))] * 4 + [("absent", None)] * 2)
        out = V.backfill_teacher_entry("v.mp4", dbt, 39, step_ms=2000)
        assert out, "should reclaim the seated entry fragment"
        assert {d.track_no for d in out} == {11}
        assert all(88_000 <= d.video_ts_ms < 100_000 for d in out)  # window-bounded
    finally:
        get_settings.cache_clear()


def test_backfill_requires_min_anchors(monkeypatch):
    """A single continuous anchor is below _BACKFILL_MIN_ANCHORS -> reclaim nothing
    (one point is not a trajectory)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dbt = _backfill_tracks()
        _wire_offline_answers(monkeypatch, [("point", (0.72, 0.55))] + [("absent", None)] * 2)
        assert V.backfill_teacher_entry("v.mp4", dbt, 39, step_ms=2000) == []
    finally:
        get_settings.cache_clear()


def test_backfill_rejects_discontinuous_stray(monkeypatch):
    """A point that jumps far to a student (track 7) is > _BACKFILL_MAX_MOVE from the
    trace and is skipped; only the continuously-tracked fragment (11) is reclaimed."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dbt = _backfill_tracks()
        _wire_offline_answers(
            monkeypatch,
            [("point", (0.72, 0.55)),   # -> track 11, continuous with seed
             ("point", (0.72, 0.55)),   # -> track 11
             ("point", (0.15, 0.55)),   # -> jumps to student 7: > MAX_MOVE -> SKIP
             ("point", (0.72, 0.55)),   # -> back on track 11
             ("absent", None), ("absent", None)],
        )
        out = V.backfill_teacher_entry("v.mp4", dbt, 39, step_ms=2000)
        assert {d.track_no for d in out} == {11}  # the far student was never anchored
    finally:
        get_settings.cache_clear()


def test_backfill_stops_immediately_when_teacher_absent(monkeypatch):
    """If the VLM reports no teacher right away (she isn't in the room before the
    chain start), reclaim nothing."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dbt = _backfill_tracks()
        _wire_offline_answers(monkeypatch, [("absent", None)] * 2)
        assert V.backfill_teacher_entry("v.mp4", dbt, 39, step_ms=2000) == []
    finally:
        get_settings.cache_clear()


def test_backfill_empty_without_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert V.backfill_teacher_entry("v.mp4", _backfill_tracks(), 39) == []
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# anchor_teacher_in_window: long holes filled per sub-window
# --------------------------------------------------------------------------- #


def _hole_tracks():
    """Inside one long hole she is student track 2 early, student track 3 late;
    track 9 is an unrelated pupil elsewhere in the room the whole time."""
    dbt = {
        1: [_det(ts, 1, 0.5, 0.6) for ts in range(0, 20_000, 400)],  # teacher, before
        2: [_det(ts, 2, 0.30, 0.50, 0.10, 0.30) for ts in range(20_000, 60_000, 400)],
        3: [_det(ts, 3, 0.70, 0.50, 0.10, 0.30) for ts in range(60_000, 100_000, 400)],
        9: [_det(ts, 9, 0.95, 0.90, 0.06, 0.10) for ts in range(0, 100_000, 400)],
    }
    return dbt


def test_long_hole_follows_her_across_several_ids(monkeypatch):
    # The case the candidate-based fill cannot do: one hole, two different
    # student ids. Points land on track 2 early and track 3 late.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        _wire_offline_answers(
            monkeypatch,
            [("point", (0.35, 0.65))] * 2 + [("point", (0.75, 0.65))] * 3,
        )
        got = V.anchor_teacher_in_window(
            "v.mp4", _hole_tracks(), teacher_no=1,
            start_ms=20_000, end_ms=100_000, step_ms=20_000,
        )
        claimed = {d.track_no for d in got}
        assert claimed == {2, 3}, f"should follow her across both ids, got {claimed}"
    finally:
        get_settings.cache_clear()


def test_long_hole_claims_nothing_while_she_is_out_of_the_room(monkeypatch):
    # She genuinely leaves for part of a long lesson; those stretches must stay
    # unlabelled rather than be filled with whoever is nearest.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        _wire_offline_answers(monkeypatch, [("absent", None)] * 5)
        got = V.anchor_teacher_in_window(
            "v.mp4", _hole_tracks(), teacher_no=1,
            start_ms=20_000, end_ms=100_000, step_ms=20_000,
        )
        assert got == [], "an absent teacher must claim nothing"
    finally:
        get_settings.cache_clear()


def test_long_hole_never_reclaims_an_already_claimed_frame(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        dbt = _hole_tracks()
        taken = {d.video_ts_ms for d in dbt[2]}
        _wire_offline_answers(monkeypatch, [("point", (0.35, 0.65))] * 5)
        got = V.anchor_teacher_in_window(
            "v.mp4", dbt, teacher_no=1, start_ms=20_000, end_ms=60_000,
            claimed_ts=taken, step_ms=20_000,
        )
        assert got == [], "frames already claimed must not be taken twice"
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# What one answer speaks for (_anchor_segment)
# --------------------------------------------------------------------------- #
#
# An answer identifies a body at an instant. Claiming a fixed slice of CLOCK
# around it puts the label boundary wherever ten seconds happens to land --
# mid-stride through a body, which is why the teacher's label flipped to
# "Student N" and back with nothing physical happening at the boundary.
# Claiming the whole raw fragment is the opposite error: the tracker's box
# migrates off a seated child onto her as she walks past, so a fragment can
# hold two people.


class _FlatModel:
    """Teacher-sized means h ~= 0.40 here, at any depth."""

    def ratio(self, d):
        return d.bbox["h"] / 0.40


def _frag(specs):
    """[(ts, h)] -> one raw fragment's ts-sorted detections."""
    return [_det(ts, 7, 0.5, 0.5, h=h) for ts, h in specs]


def test_segment_claims_the_whole_continuous_body():
    frag = _frag([(t, 0.40) for t in range(0, 20_001, 400)])
    got = V._anchor_segment(frag, len(frag) // 2, _FlatModel(), 0, 20_000)
    assert len(got) == len(frag)


def test_segment_stops_at_a_tracker_box_migration():
    # The real shape: the box sits on a seated child, balloons over three
    # frames, and lands on the teacher. Seeded on HER side it must not walk
    # back down the ramp onto him -- this is the ~5700-detection error.
    child = [(t, 0.25) for t in range(0, 10_001, 400)]
    ramp = [(10_400, 0.30), (10_800, 0.35)]
    teacher = [(t, 0.42) for t in range(11_200, 20_001, 400)]
    frag = _frag(child + ramp + teacher)
    seed = len(child) + len(ramp) + 5
    got = V._anchor_segment(frag, seed, _FlatModel(), 0, 20_000)
    assert got, "should claim her side"
    assert min(d.video_ts_ms for d in got) >= 10_400
    assert all(d.bbox["h"] >= 0.30 for d in got), "walked down onto the child"


def test_segment_refuses_a_seed_that_is_not_teacher_sized():
    # One real anchor pointed into a corner child's box; a clock window around
    # it would have labelled him teacher for twenty seconds.
    frag = _frag([(t, 0.15) for t in range(0, 20_001, 400)])
    assert V._anchor_segment(frag, 10, _FlatModel(), 0, 20_000) == []


def test_segment_bridges_a_brief_occlusion():
    # A sample or two of a collapsed box is someone passing in front of her,
    # not the end of her trajectory: bridge it, and keep going.
    specs = [(t, 0.40) for t in range(0, 20_001, 400)]
    specs[25] = (specs[25][0], 0.15)
    frag = _frag(specs)
    got = V._anchor_segment(frag, 10, _FlatModel(), 0, 20_000)
    assert len(got) == len(frag), "a one-frame dip must not cut the run"


def test_segment_stops_at_a_sustained_collapse():
    specs = [(t, 0.40) for t in range(0, 10_001, 400)]
    specs += [(t, 0.15) for t in range(10_400, 20_001, 400)]
    frag = _frag(specs)
    got = V._anchor_segment(frag, 5, _FlatModel(), 0, 20_000)
    assert max(d.video_ts_ms for d in got) < 12_000


def test_segment_stops_at_a_break_in_tracking():
    # Either side of a long silence is not one continuous trajectory.
    frag = _frag([(t, 0.40) for t in range(0, 5_001, 400)]
                 + [(t, 0.40) for t in range(20_000, 25_001, 400)])
    got = V._anchor_segment(frag, 3, _FlatModel(), 0, 25_000)
    assert max(d.video_ts_ms for d in got) <= 5_000


def test_segment_stays_inside_the_hole():
    frag = _frag([(t, 0.40) for t in range(0, 40_001, 400)])
    got = V._anchor_segment(frag, 50, _FlatModel(), 10_000, 30_000)
    assert min(d.video_ts_ms for d in got) >= 10_000
    assert max(d.video_ts_ms for d in got) <= 30_000
