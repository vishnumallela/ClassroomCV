"""Unit tests for teacher board-activity classification (app.activity)."""

from __future__ import annotations

from app.activity import derive_board_activity
from app.models import Detection

# Board occupying the upper-middle of the frame.
BOARD = [[0.40, 0.10], [0.70, 0.10], [0.70, 0.40], [0.40, 0.40]]
# Teacher standing at the board (bbox overlaps the board region).
BBOX = {"x": 0.45, "y": 0.30, "w": 0.10, "h": 0.35}
STEP_MS = 200  # 5 fps


def _det(ts_ms, *, arms, fc=0.8, ts_scale=0.2, bbox=BBOX):
    return Detection(
        video_ts_ms=ts_ms,
        raw_track_id=1,
        bbox=dict(bbox),
        conf=0.9,
        standing=True,
        back_to_camera=False,
        track_no=1,
        activity=None if arms is None else {"ts": ts_scale, "fc": fc, "arms": arms},
    )


def _run(dets):
    return derive_board_activity(dets, BOARD)


def test_no_board_returns_nulls():
    out = derive_board_activity([_det(0, arms=[[0.55, 0.2, 0.4]])], None)
    assert out["pointing_ms"] is None
    assert out["writing_ms"] is None
    assert out["segments"] == []


def test_sustained_pointing():
    # Wrist inside the board, clearly raised, held steady for 5 s.
    dets = [_det(i * STEP_MS, arms=[[0.55, 0.20, 0.45]]) for i in range(26)]
    out = _run(dets)
    assert out["pointing_ms"] > 3_000, out
    assert out["writing_ms"] == 0
    assert any(s["kind"] == "pointing" for s in out["segments"])


def test_at_board_no_arms_is_near():
    dets = [_det(i * STEP_MS, arms=None) for i in range(26)]
    out = _run(dets)
    assert out["near_ms"] > 3_000, out
    assert out["pointing_ms"] == 0
    assert all(s["kind"] == "near" for s in out["segments"])


def test_hand_down_is_not_interaction():
    # Wrist near the board but hanging well below the shoulder -> not engaged.
    dets = [_det(i * STEP_MS, arms=[[0.55, 0.55, -0.6]]) for i in range(26)]
    out = _run(dets)
    assert out["pointing_ms"] == 0
    assert out["writing_ms"] == 0


def test_away_from_board_is_empty():
    far = {"x": 0.02, "y": 0.55, "w": 0.08, "h": 0.30}
    dets = [_det(i * STEP_MS, arms=[[0.06, 0.5, 0.4]], bbox=far) for i in range(26)]
    out = _run(dets)
    assert out["segments"] == []
    assert out["pointing_ms"] == 0


def test_writing_micro_motion_on_board():
    # Wrist ON the board, facing it, with small local jitter (writing signature).
    dets = []
    for i in range(26):
        jitter = 0.006 if i % 2 == 0 else -0.006
        dets.append(_det(i * STEP_MS, arms=[[0.55 + jitter, 0.22 + jitter, 0.35]], fc=0.85))
    out = _run(dets)
    assert out["writing_ms"] > 0, out
    assert any(s["kind"] == "writing" for s in out["segments"])


def test_gap_breaks_segments():
    # Two pointing bursts separated by a >5 s absence must not bridge.
    first = [_det(i * STEP_MS, arms=[[0.55, 0.20, 0.45]]) for i in range(20)]
    start2 = 20 * STEP_MS + 8_000
    second = [_det(start2 + i * STEP_MS, arms=[[0.55, 0.20, 0.45]]) for i in range(20)]
    out = _run(first + second)
    pointing_segs = [s for s in out["segments"] if s["kind"] == "pointing"]
    assert len(pointing_segs) >= 2, out
