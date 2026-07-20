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
    """points: list of (x, y) or None, one per call to _ask_point, in order."""
    monkeypatch.setattr(V.detector, "resolve_video_source", lambda p: (p, False))
    monkeypatch.setattr(V.cv2, "VideoCapture", lambda _path: _FakeCapture())
    it = iter(points)
    monkeypatch.setattr(V, "_ask_point", lambda *a, **kw: next(it, None))


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
