"""Teacher board-activity classification: pointing / writing / near.

Ported from the class-teacher-detctor reference (board.py), adapted to
ClassroomCV's normalized (0-1) coordinate convention and its derive-time
pipeline.

At detection time we already stored board-INDEPENDENT pose features on
Detection.activity ({torso_scale, facing, per-arm wrist + wrist_up_ratio}).
Here we combine them with the board polygon to classify each teacher frame as
NONE / NEAR / POINTING / WRITING, run a writing micro-motion detector, and pass
the raw per-frame labels through a debounce state machine that emits stable
segments {kind, start_ms, end_ms} plus per-kind millisecond totals.

Keeping the board-relative decision HERE (not in the detector) is what lets
/rederive re-classify against an edited board polygon without re-running YOLO.

Per-frame decision (mirrors reference board.py:classify_frame):
  1. not at the board .......................... NONE
  2. at board, no usable arm ................... NEAR
  3. at board, hand not near board OR not raised NONE
  4. hand on board + micro-motion + facing ..... WRITING
  5. otherwise (hand raised toward board) ...... POINTING
POINTING + WRITING are the meaningful interactions; NEAR is tracked separately.
"""

from __future__ import annotations

import math
from typing import Optional

from app.geometry import bboxes_intersect, expand_bbox, point_in_polygon, polygon_bbox
from app.models import Detection

# Frame labels (lowercase to match the API/DB board_interactions kinds).
NONE, NEAR, POINTING, WRITING = "none", "near", "pointing", "writing"

# --- at-board proximity (normalized units) --------------------------------- #
# The teacher is "at the board" when her bbox intersects the board bbox expanded
# by this margin. Boards mount above head height, so the person box typically
# sits just below / overlapping the board box.
AT_BOARD_EXPAND = 0.06

# --- hand engagement ------------------------------------------------------- #
# Wrist counts as near the board when inside the board polygon OR within the
# board bbox expanded by this margin.
WRIST_NEAR_EXPAND = 0.06
# Wrist must be raised at least this much: wrist_up_ratio is (shoulder_y -
# wrist_y) / torso_scale, so 0 == shoulder height, positive == raised. Ported
# from the reference hands_up_min (-0.30): pointing measured ~+0.3 torso-lengths,
# a hand hanging at the side ~-0.44. -0.30 keeps a slightly-below-shoulder point
# while rejecting a resting arm.
HANDS_UP_MIN = -0.30
# Writing additionally requires the teacher to be facing the board (occluded
# face). facing_score ~0 faces the students, ~0.85 back-turned.
FACING_GATE = 0.5
# POINTING requires the arm to be EXTENDED toward the board: the shoulder->wrist
# reach must be at least this many torso-lengths. A bent arm resting near the
# board is not a point. NOTE: unlike the reference's literal 1.2, this is
# calibrated to OUR torso-scale + pose model — reach here tops out ~0.86 because
# a board-pointing arm is rarely fully straight. 0.40 was swept against the
# reference demo (pointing 51.8s -> 39.8s vs the reference's 40.8s, matching its
# 8-segment structure). Only applied to detections carrying reach (4-element
# arms); older analyses stay backward-compatible.
POINTING_REACH_RATIO = 0.40

# --- writing micro-motion (torso-length units) ----------------------------- #
# The engaged wrist must move (mean per-frame step >= activity) but stay local
# (spread <= max) — writing, not a broad sweep. Calibrated by the reference at
# ~8 fps; our 5 fps sampling makes each step larger, so these are a starting
# point to tune against real footage.
WRITE_ACTIVITY_MIN = 0.012
WRITE_SPREAD_MAX = 0.55
WRITE_WINDOW_MS = 1_000
WRITE_MIN_SAMPLES = 3

# --- temporal debounce (ms) ------------------------------------------------ #
STATE_HOLD_MS = 600  # a new label must persist this long before it commits
MERGE_GAP_MS = 1_000  # merge same-kind segments closer together than this
MIN_SEGMENT_MS = 500  # drop segments shorter than this (pose jitter)
# A sampling gap this large means the teacher was absent (no detections): break
# the run so a segment never bridges an absence. Matches events.PRESENCE_GAP_MS.
ACTIVITY_GAP_MS = 5_000


def _point_box_dist(x: float, y: float, box: tuple) -> float:
    """Euclidean distance from a point to an axis-aligned box (0 if inside)."""
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _in_box(x: float, y: float, box: tuple) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


class _WristMotion:
    """Rolling buffer of the engaged wrist's torso-normalized position.

    is_writing() is the reference writing signature: enough movement (mean step)
    but confined to a small region (spread) over the recent window.
    """

    def __init__(self, window_ms: int = WRITE_WINDOW_MS) -> None:
        self.window_ms = window_ms
        self._buf: list[tuple[int, float, float]] = []  # (ts_ms, x/ts, y/ts)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, ts_ms: int, wx: float, wy: float, scale: float) -> None:
        s = scale if scale > 1e-6 else 1e-6
        self._buf.append((ts_ms, wx / s, wy / s))
        cutoff = ts_ms - self.window_ms
        if self._buf[0][0] < cutoff:
            self._buf = [p for p in self._buf if p[0] >= cutoff]

    def is_writing(self) -> bool:
        if len(self._buf) < WRITE_MIN_SAMPLES:
            return False
        xs = [p[1] for p in self._buf]
        ys = [p[2] for p in self._buf]
        steps = [
            math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            for i in range(1, len(self._buf))
        ]
        activity = sum(steps) / len(steps) if steps else 0.0
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return activity >= WRITE_ACTIVITY_MIN and spread <= WRITE_SPREAD_MAX


class _Debounce:
    """Min-hold state machine turning raw per-frame labels into stable periods.

    A new label must persist STATE_HOLD_MS before it replaces the committed one;
    the closed period runs from its start to the moment the successor first
    appeared (mirrors reference InteractionStateMachine.update).
    """

    def __init__(self, hold_ms: int = STATE_HOLD_MS) -> None:
        self.hold_ms = hold_ms
        self._committed: Optional[str] = None
        self._committed_start: Optional[int] = None
        self._cand: Optional[str] = None
        self._cand_since: Optional[int] = None
        self._last_ts: Optional[int] = None
        self.periods: list[list] = []  # [start_ms, end_ms, kind]

    def update(self, ts_ms: int, raw: str) -> None:
        if self._committed is None:
            self._committed = raw
            self._committed_start = ts_ms
        elif raw == self._committed:
            self._cand = None
            self._cand_since = None
        elif raw != self._cand:
            self._cand = raw
            self._cand_since = ts_ms
        elif self._cand_since is not None and ts_ms - self._cand_since >= self.hold_ms:
            self.periods.append([self._committed_start, self._cand_since, self._committed])
            self._committed = raw
            self._committed_start = self._cand_since
            self._cand = None
            self._cand_since = None
        self._last_ts = ts_ms

    def finalize(self) -> list[list]:
        if (
            self._committed is not None
            and self._committed_start is not None
            and self._last_ts is not None
        ):
            self.periods.append([self._committed_start, self._last_ts, self._committed])
        return self.periods


def _classify_frame(
    det: Detection,
    board_polygon: list[list[float]],
    board_box: tuple,
    at_box: tuple,
    near_box: tuple,
    motion: _WristMotion,
) -> str:
    """Raw per-frame label from stored pose features + board geometry."""
    b = det.bbox
    det_box = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
    if not bboxes_intersect(det_box, at_box):
        motion.reset()
        return NONE

    act = det.activity
    arms = act.get("arms") if act else None
    if not arms:
        motion.reset()
        return NEAR

    scale = float(act.get("ts") or 0.0)
    # The engaged arm is the one whose wrist is closest to the board.
    arm = min(arms, key=lambda a: _point_box_dist(float(a[0]), float(a[1]), board_box))
    wx, wy, up = float(arm[0]), float(arm[1]), float(arm[2])
    reach = float(arm[3]) if len(arm) > 3 else None  # None for legacy detections
    motion.update(det.video_ts_ms, wx, wy, scale)

    on_board = point_in_polygon(wx, wy, board_polygon)
    hand_near = on_board or _in_box(wx, wy, near_box)
    hand_raised = up >= HANDS_UP_MIN
    if not (hand_near and hand_raised):
        return NONE

    if on_board and motion.is_writing() and float(act.get("fc", 0.0)) >= FACING_GATE:
        return WRITING
    # A raised hand near the board is only POINTING when the arm is extended;
    # a bent arm (low reach) resting at the board is not a point (matches the
    # reference, which returns NONE for a hand-down-at-board frame).
    if reach is not None and reach < POINTING_REACH_RATIO:
        return NONE
    return POINTING


def _clean(periods: list[list]) -> list[list]:
    """Drop NONE + too-short periods, then merge same-kind across small gaps."""
    kept = [p for p in periods if p[2] != NONE and (p[1] - p[0]) >= MIN_SEGMENT_MS]
    kept.sort(key=lambda p: p[0])
    merged: list[list] = []
    for s, e, k in kept:
        if merged and merged[-1][2] == k and s - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1][1] = e
        else:
            merged.append([s, e, k])
    return merged


def derive_board_activity(
    teacher_dets: list[Detection],
    board_polygon: Optional[list[list[float]]],
) -> dict:
    """Classify the teacher's board activity into pointing / writing / near.

    Returns {"pointing_ms", "writing_ms", "near_ms", "segments"} where segments
    is [{kind, start_ms, end_ms}, ...]. The three *_ms values are None when no
    board zone exists (the activity is undefined without a board).
    """
    if not board_polygon:
        return {
            "pointing_ms": None,
            "writing_ms": None,
            "near_ms": None,
            "segments": [],
        }

    dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)
    board_box = polygon_bbox(board_polygon)
    at_box = expand_bbox(board_box, AT_BOARD_EXPAND)
    near_box = expand_bbox(board_box, WRIST_NEAR_EXPAND)

    all_periods: list[list] = []
    i, n = 0, len(dets)
    while i < n:
        # Contiguous presence run: consecutive detections < ACTIVITY_GAP_MS apart.
        j = i + 1
        while (
            j < n and dets[j].video_ts_ms - dets[j - 1].video_ts_ms < ACTIVITY_GAP_MS
        ):
            j += 1
        motion = _WristMotion()
        sm = _Debounce()
        for d in dets[i:j]:
            sm.update(
                d.video_ts_ms,
                _classify_frame(d, board_polygon, board_box, at_box, near_box, motion),
            )
        all_periods.extend(sm.finalize())
        i = j

    segments = _clean(all_periods)
    by_kind = {POINTING: 0, WRITING: 0, NEAR: 0}
    for s, e, k in segments:
        by_kind[k] = by_kind.get(k, 0) + (e - s)
    return {
        "pointing_ms": by_kind[POINTING],
        "writing_ms": by_kind[WRITING],
        "near_ms": by_kind[NEAR],
        "segments": [{"kind": k, "start_ms": s, "end_ms": e} for s, e, k in segments],
    }
