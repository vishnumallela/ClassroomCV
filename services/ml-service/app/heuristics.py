"""KPI heuristics — the tunable rule layer for Luminary's teacher metrics.

Everything the product reports reduces to a handful of teacher-centric KPIs,
and every rule that turns raw detections into one of them lives HERE, in one
file, because this is the layer that gets tuned per school, extended with
add-ons, and re-calibrated when cameras move:

1. TEACHER ENTRY / EXIT — presence_intervals + door_entry_exit (+ the two
   gap-bridging rules that keep camera blind spots from fabricating
   crossings).
2. TEACHER BOARD TIME — board_condition + intervals_from_samples
   (hysteresis so one occluded sample cannot open/close a session).
3. TEACHER HEATMAP — teacher_heatmap (dwell histogram of her bbox centers;
   at a fixed sample rate a cell's count is proportional to time spent
   there).
4. TEACHER ACTIONS — action_samples + the same hysteresis, over the
   detector's own `pointing` and `writing` classes, attributed to her.

Rules of the file:
- Pure functions over Detection lists; no I/O, no model calls, no globals.
  That keeps every rule unit-testable in milliseconds and safe to re-run on
  stored detections (/rederive).
- Every constant is a named knob with the measurement that set it. Change a
  knob here and only here.
- Anything that is NOT one of these KPIs does not belong in this file —
  and after the 2026-08 slimming, is not computed at all.

Who is "the teacher" is decided upstream (roles.assign_roles, verified by
teacher_id.verify_teacher); these heuristics trust the label they are given.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Optional

from app.geometry import bboxes_intersect, expand_bbox, polygon_bbox
from app.models import Detection

# --------------------------------------------------------------------------- #
# Knobs (each one names the failure it was set against)
# --------------------------------------------------------------------------- #

# A gap in the teacher's detections at least this long splits presence into
# two intervals (candidate exit/enter). Shorter gaps are sampling jitter.
PRESENCE_GAP_MS = 5_000
# An interval running into the last END_MARGIN_MS of the video produces no
# exit: the lesson simply ended with her in the room.
END_MARGIN_MS = 5_000
# A presence gap no longer than this whose bounding detections are BOTH away
# from every door is a camera blind spot (the teacher stepped to a
# near-camera corner desk, out of view), not a door exit: presence is bridged
# so a walk past the door followed by an off-camera moment is not mislabelled
# a crossing.
OFFSCREEN_BRIDGE_MS = 12_000

# --- entry / exit: through the door, or by absence beyond a buffer ----------
# The door decides DIRECTION when there is one: an exit is the teacher moving
# TOWARD the door (or standing in the doorway) as she vanishes, an entry is her
# moving AWAY from it after she appears — not merely vanishing near it, which
# on real footage is as often the door frame occluding her as it is a crossing.
# Without a door, or when no crossing was seen, an absence longer than
# OUT_OF_FRAME_BUFFER_MS is an exit and the return after it an entry; anything
# shorter is a blind spot or an occlusion and is bridged into presence.
OUT_OF_FRAME_BUFFER_MS = 20_000
# The foot point must close on (or open from) the door by this much of the
# frame across the trajectory window for the movement to count as a crossing.
DOOR_APPROACH_MIN = 0.03

# Door test window anchoring the check at the presence edge (the
# vanish/appear samples). 4s reached back far enough to catch the teacher
# merely WALKING PAST the door on her way out of frame (pixel-verified false
# exit at 250.8s); 2s still covers door-frame occlusion dropping the track a
# beat early.
DOOR_WINDOW_MS = 2_000
DOOR_EXPAND = 0.15

# Board zone expansion: the teacher stands AT the board, not IN it, so the
# polygon is grown by 12% of the frame on every side before the bbox test.
BOARD_EXPAND = 0.12
# Hysteresis: this long continuously at the board opens a session…
BOARD_ON_MS = 2_000
# …and this long continuously away closes it. Asymmetric on purpose: turning
# to address the class mid-writing must not close the session.
BOARD_OFF_MS = 3_000
# Opening tolerance: a candidate run survives brief false flickers (a single
# occluded sample) instead of resetting; without this, opening needs ~10
# CONSECUTIVE true samples at 5 fps and short board stints never register.
BOARD_FLICKER_SAMPLES = 2
BOARD_FLICKER_BUDGET_MS = 600

# --- teacher actions (pointing / writing) ---
# An action box counts as HERS when its center falls inside her bbox grown by
# this fraction of the frame. Center-in-box rather than IoU on purpose: it is
# correct whether the annotator drew the action on her whole body (center ~=
# her center) or on the hand/forearm alone (center still inside her box), and
# this pipeline has no per-frame ground truth for the action classes yet to
# choose between those on evidence.
ACTION_EXPAND = 0.05
# Hysteresis for actions, shorter than the board's because a gesture is an
# event and standing at the board is a station: 600 ms is 3 samples at 5 fps,
# enough to reject a single-frame false positive without losing a real point.
#
# HONESTY NOTE: unlike every other knob in this file, these three are NOT set
# from a measurement — the action classes have no annotated ground truth in
# eval/gt yet. They are deliberate placeholders. Calibrate them the way
# BOARD_ON_MS was: annotate action spans on a real lesson, then sweep.
ACTION_ON_MS = 600
ACTION_OFF_MS = 1_000
ACTION_FLICKER_SAMPLES = 1

# Heatmap grid: 32x18 matches the 16:9 CCTV aspect; fine enough to separate
# aisles, coarse enough that an hour of 5 fps samples still reads as a path.
HEATMAP_GRID_W = 32
HEATMAP_GRID_H = 18


# --------------------------------------------------------------------------- #
# KPI 1 — teacher entry / exit
# --------------------------------------------------------------------------- #


def presence_intervals(
    ts_sorted: list[int], gap_ms: int = PRESENCE_GAP_MS
) -> list[list[int]]:
    """Union of detection timestamps into intervals, split at gaps >= gap_ms."""
    if not ts_sorted:
        return []
    intervals: list[list[int]] = [[ts_sorted[0], ts_sorted[0]]]
    for ts in ts_sorted[1:]:
        if ts - intervals[-1][1] >= gap_ms:
            intervals.append([ts, ts])
        else:
            intervals[-1][1] = ts
    return intervals


def entry_exit_from_intervals(
    intervals: list[list[int]],
    duration_ms: int,
    end_margin_ms: int = END_MARGIN_MS,
) -> list[dict]:
    """[{kind:'enter'|'exit', ts_ms}] from every interval edge (no-door path)."""
    events: list[dict] = []
    for i, (start, end) in enumerate(intervals):
        events.append({"kind": "enter", "ts_ms": start})
        is_last = i == len(intervals) - 1
        if not is_last or end < duration_ms - end_margin_ms:
            events.append({"kind": "exit", "ts_ms": end})
    return events


def near_zone(det: Detection, polygon: list[list[float]], expand: float = 0.12) -> bool:
    """True when the detection bbox CENTER lies inside the expanded polygon bbox.

    Center containment, not bbox intersection: a person walking out of the
    camera's view grows a frame-corner box wide enough to clip any generously
    expanded zone, which turned a bottom-left camera-edge disappearance into a
    door crossing on verified footage. Someone genuinely at the zone has their
    center inside the expanded box.
    """
    x0, y0, x1, y1 = expand_bbox(polygon_bbox(polygon), expand)
    b = det.bbox
    cx = b["x"] + b["w"] / 2.0
    cy = b["y"] + b["h"] / 2.0
    return x0 <= cx <= x1 and y0 <= cy <= y1


def door_entry_exit(
    teacher_dets: list[Detection],
    intervals: list[list[int]],
    door_polygons: list[list[list[float]]],
    duration_ms: int,
    end_margin_ms: int = END_MARGIN_MS,
    window_ms: int = DOOR_WINDOW_MS,
) -> list[dict]:
    """Enter/exit events counted only for presence edges at the door zone.

    Tracking usually loses a person a beat before they physically reach the
    door (door-frame occlusion, partial exit from view), so the door test is
    WINDOWED: an interval edge is a crossing when ANY sample within window_ms
    of that edge is near the door. Single-sample tests miss most true
    crossings on real footage.

    Video-edge semantics match the presence-based rule: an interval that
    starts the video counts as an enter (the teacher was already inside), and
    the final interval running into the last end_margin_ms of the video does
    not produce an exit. Interior presence gaps away from the door (mid-room
    occlusions) produce no events, so counts reflect real door crossings.
    """
    dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)
    events: list[dict] = []

    def any_near(lo: int, hi: int) -> bool:
        return any(
            lo <= d.video_ts_ms <= hi
            and any(near_zone(d, p, DOOR_EXPAND) for p in door_polygons)
            for d in dets
        )

    for i, (start, end) in enumerate(intervals):
        at_video_start = i == 0 and start <= end_margin_ms
        if at_video_start or any_near(start, start + window_ms):
            events.append({"kind": "enter", "ts_ms": start})
        is_last = i == len(intervals) - 1
        if not is_last or end < duration_ms - end_margin_ms:
            if any_near(end - window_ms, end):
                events.append({"kind": "exit", "ts_ms": end})
    return events


def bridge_offscreen_gaps(
    intervals: list[list[int]],
    teacher_dets: list[Detection],
    door_polygons: list[list[list[float]]],
    bridge_ms: int = OFFSCREEN_BRIDGE_MS,
    expand: float = DOOR_EXPAND,
) -> list[list[int]]:
    """Merge presence intervals split by a short away-from-door gap.

    The teacher constantly walks to pupils' desks, and a near-camera corner
    desk sits in the lens blind spot: she vanishes for a few seconds and
    reappears at the same corner without ever touching the door. Left split,
    that gap becomes a spurious exit/enter pair (worse, a walk PAST the door on
    the way there lands inside the door window). A gap is bridged only when it
    is short AND the detections bounding it are both clear of every door; a gap
    whose vanish or reappearance is door-adjacent is a real crossing and kept.
    """
    if not door_polygons or len(intervals) < 2 or not teacher_dets:
        return intervals
    dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)
    ts = [d.video_ts_ms for d in dets]

    def at_door(i: int) -> bool:
        return 0 <= i < len(dets) and any(
            near_zone(dets[i], p, expand) for p in door_polygons
        )

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev_end = merged[-1][1]
        gap = start - prev_end
        vanished_at_door = at_door(bisect_right(ts, prev_end) - 1)
        appeared_at_door = at_door(bisect_left(ts, start))
        if 0 < gap <= bridge_ms and not vanished_at_door and not appeared_at_door:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _foot(det: Detection) -> tuple[float, float]:
    b = det.bbox
    return b["x"] + b["w"] / 2.0, b["y"] + b["h"]


def _door_distance(det: Detection, polygon: list[list[float]]) -> float:
    x0, y0, x1, y1 = polygon_bbox(polygon)
    fx, fy = _foot(det)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5


def in_doorway(det: Detection, polygon: list[list[float]]) -> bool:
    """Centre inside the door polygon's own box: she is IN the doorway."""
    return near_zone(det, polygon, expand=0.0)


def door_direction(
    window: list[Detection], polygon: list[list[float]], min_change: float = DOOR_APPROACH_MIN
) -> Optional[str]:
    """"toward" / "away" / None from how the foot point's distance to the door
    changed across the window (first sample to last)."""
    if len(window) < 2:
        return None
    delta = _door_distance(window[-1], polygon) - _door_distance(window[0], polygon)
    if delta <= -min_change:
        return "toward"
    if delta >= min_change:
        return "away"
    return None


def classify_presence(
    intervals: list[list[int]],
    teacher_dets: list[Detection],
    door_polygons: list[list[list[float]]],
    duration_ms: int,
    buffer_ms: int = OUT_OF_FRAME_BUFFER_MS,
    window_ms: int = DOOR_WINDOW_MS,
    end_margin_ms: int = END_MARGIN_MS,
) -> tuple[list[list[int]], list[dict]]:
    """Presence with occlusions bridged, and the entry/exit events, in one pass.

    Each gap between two presence intervals is one of three things:
      "door"   — a crossing was SEEN: she moved toward the door (or stood in
                 the doorway) as she vanished, or moved away from it as she
                 reappeared. An exit and an entry, whatever the gap's length.
      "buffer" — no crossing seen, but she was gone for at least buffer_ms:
                 an exit and an entry, inferred from the absence.
      bridged  — neither: a blind spot or an occlusion, and presence continues.
    The first sighting is always an entry ("start", or "door" when she came in
    through the door after the video began). The last interval ends with an
    exit only if a crossing was seen, or if the video still had at least
    buffer_ms to run — vanishing seconds before the end is not leaving.
    Every event carries its `method`, so a count can say how many exits were
    seen at the door and how many were inferred.
    """
    if not intervals:
        return [], []
    dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)

    def window(lo: int, hi: int) -> list[Detection]:
        return [d for d in dets if lo <= d.video_ts_ms <= hi]

    def near_any(w: list[Detection]) -> bool:
        return any(near_zone(d, p, DOOR_EXPAND) for d in w for p in door_polygons)

    def exit_seen(end: int) -> bool:
        w = window(end - window_ms, end)
        if not w or not door_polygons or not near_any(w):
            return False
        return any(in_doorway(w[-1], p) or door_direction(w, p) == "toward" for p in door_polygons)

    def enter_seen(start: int) -> bool:
        w = window(start, start + window_ms)
        if not w or not door_polygons or not near_any(w):
            return False
        return any(in_doorway(w[0], p) or door_direction(w, p) == "away" for p in door_polygons)

    out: list[list[int]] = []
    events: list[dict] = []
    cur = list(intervals[0])
    first = cur[0]
    events.append(
        {
            "kind": "enter",
            "ts_ms": first,
            "method": "door" if first > end_margin_ms and enter_seen(first) else "start",
        }
    )
    for start, end in intervals[1:]:
        gap = start - cur[1]
        if exit_seen(cur[1]) or enter_seen(start):
            method = "door"
        elif gap >= buffer_ms:
            method = "buffer"
        else:
            cur[1] = max(cur[1], end)
            continue
        events.append({"kind": "exit", "ts_ms": cur[1], "method": method})
        events.append({"kind": "enter", "ts_ms": start, "method": method})
        out.append(cur)
        cur = [start, end]
    out.append(cur)
    remaining = duration_ms - cur[1]
    if remaining >= end_margin_ms:
        if exit_seen(cur[1]):
            events.append({"kind": "exit", "ts_ms": cur[1], "method": "door"})
        elif remaining >= buffer_ms:
            events.append({"kind": "exit", "ts_ms": cur[1], "method": "buffer"})
    return out, events


def bridge_short_gaps(
    intervals: list[list[int]], max_gap_ms: int = OFFSCREEN_BRIDGE_MS
) -> list[list[int]]:
    """Merge presence intervals separated by a gap no larger than max_gap_ms.

    Used ONLY when there is no door zone, so we cannot tell a crossing from a
    blind spot by position. A gap of a few seconds is far more likely a
    tracking dropout than the teacher leaving and returning that fast, so
    bridging stops a brief gap from fabricating an enter/exit pair (the no-door
    path counts every interval edge as a crossing) and from undercounting
    presence. A longer gap stays split as a genuine absence.
    """
    if len(intervals) < 2:
        return intervals
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


# --------------------------------------------------------------------------- #
# KPI 2 — teacher board time
# --------------------------------------------------------------------------- #


def board_condition(det: Detection, board_polygon: list[list[float]]) -> bool:
    """Teacher-at-board test: bbox clips the expanded board AND center-x is in
    the board's x-range.

    back_to_camera used to bypass the x-range gate, but any back-turned box
    within the 12% expansion counted as at-board: pixel-verified board_enter
    fired at 225.0s with the teacher standing among desks 0.28 of the frame
    LEFT of the board. Standing at the board puts the bbox center inside the
    board's x-range anyway (the zone spans the board's full width), so the
    bypass only ever admitted false positives.
    """
    poly_box = polygon_bbox(board_polygon)
    expanded = expand_bbox(poly_box, BOARD_EXPAND)
    b = det.bbox
    det_box = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
    if not bboxes_intersect(det_box, expanded):
        return False
    cx = b["x"] + b["w"] / 2.0
    return poly_box[0] <= cx <= poly_box[2]


def intervals_from_samples(
    samples: list[tuple[int, bool]],
    on_ms: int = BOARD_ON_MS,
    off_ms: int = BOARD_OFF_MS,
    gap_ms: int = PRESENCE_GAP_MS,
    flicker_samples: int = BOARD_FLICKER_SAMPLES,
    flicker_budget_ms: int = BOARD_FLICKER_BUDGET_MS,
) -> list[list[int]]:
    """Hysteresis state machine over (ts, condition) samples (time-sorted).

    Shared by board time and the action KPIs — the machine is generic and only
    the four knobs differ, so the name does not claim a board. Defaults are the
    board's, which is what every pre-existing caller wants.

    Opens an interval once the condition has been continuously true for
    >= on_ms (interval starts at the first true sample of the run); closes
    once it has been false for >= off_ms (interval ends at the last true
    sample). A still-open interval at the end of samples is closed at the
    last true timestamp.

    Samples exist only while the teacher is detected, so a sampling gap
    >= gap_ms is a hard break (the teacher was absent): an open interval is
    closed at the last true sample before the gap and any candidate run is
    reset. Board intervals therefore never bridge periods when the teacher
    was not present, keeping them consistent with presence_intervals.
    """
    intervals: list[list[int]] = []
    on = False
    run_start: Optional[int] = None  # first true ts of current candidate run
    last_true: Optional[int] = None
    start: Optional[int] = None
    prev_ts: Optional[int] = None
    false_streak = 0
    false_run_ms = 0

    for ts, cond in samples:
        if prev_ts is not None and ts - prev_ts >= gap_ms:
            if on and start is not None and last_true is not None:
                intervals.append([start, last_true])
            on = False
            run_start = None
            start = None
            last_true = None
            false_streak = 0
            false_run_ms = 0
        if not on:
            if cond:
                false_streak = 0
                if run_start is None:
                    run_start = ts
                    false_run_ms = 0
                if ts - run_start >= on_ms and false_run_ms <= flicker_budget_ms:
                    on = True
                    start = run_start
                    last_true = ts
            elif run_start is not None:
                false_streak += 1
                if prev_ts is not None:
                    false_run_ms += ts - prev_ts
                if false_streak >= flicker_samples or false_run_ms > flicker_budget_ms:
                    run_start = None
                    false_streak = 0
                    false_run_ms = 0
        else:
            if cond:
                last_true = ts
            elif last_true is not None and ts - last_true >= off_ms:
                intervals.append([start, last_true])
                on = False
                run_start = None
                start = None
                last_true = None
        prev_ts = ts

    if on and start is not None and last_true is not None:
        intervals.append([start, last_true])
    return intervals


# --------------------------------------------------------------------------- #
# KPI 3 — teacher heatmap
# --------------------------------------------------------------------------- #


def teacher_heatmap(
    teacher_dets: list[Detection],
    grid_w: int = HEATMAP_GRID_W,
    grid_h: int = HEATMAP_GRID_H,
) -> dict:
    """Row-major dwell histogram of the teacher's bbox centers.

    Each detection drops one sample into the grid cell its bbox center falls
    in; sampled at a fixed rate, a cell's count is proportional to time spent
    there, so the grid traces her teaching path — board, aisles, the desks she
    visits. Teacher-only by design: per-student analytics were removed in the
    2026-08 KPI slimming, and not accumulating a students grid keeps the
    stored analytics row small.
    """
    grid = [0] * (grid_w * grid_h)
    for d in teacher_dets:
        cx = d.bbox["x"] + d.bbox["w"] / 2.0
        cy = d.bbox["y"] + d.bbox["h"] / 2.0
        gx = min(grid_w - 1, max(0, int(cx * grid_w)))
        gy = min(grid_h - 1, max(0, int(cy * grid_h)))
        grid[gy * grid_w + gx] += 1
    return {"grid_w": grid_w, "grid_h": grid_h, "teacher": grid}


# --------------------------------------------------------------------------- #
# KPI 4 — teacher actions (pointing / writing)
# --------------------------------------------------------------------------- #


def action_samples(
    teacher_dets: list[Detection],
    action_dets: list[Detection],
    expand: float = ACTION_EXPAND,
) -> list[tuple[int, bool]]:
    """(ts, is_acting) per teacher sample, for one action class.

    The detector emits `pointing`/`writing` as their own classes rather than as
    an attribute of the teacher box, so they have to be attributed back to her.
    Two rules do it:

    ATTRIBUTION  An action box is hers when its center lies inside her box for
                 that same frame, grown by `expand`. An action box with no
                 teacher box in the frame is discarded rather than counted —
                 it is either a false positive or somebody else's arm, and
                 neither is a teacher KPI.

    SAMPLING     One sample per TEACHER detection, not per action detection.
                 That makes the sample series identical in shape to the board's,
                 so the same hysteresis applies and action time is a subset of
                 presence time by construction — `writing_ms` can never exceed
                 `present_ms`, which a per-action-box series would allow.
    """
    if not teacher_dets:
        return []
    by_ts: dict[int, list[Detection]] = {}
    for d in action_dets:
        by_ts.setdefault(d.video_ts_ms, []).append(d)

    samples: list[tuple[int, bool]] = []
    for t in teacher_dets:
        b = t.bbox
        x0 = b["x"] - expand
        y0 = b["y"] - expand
        x1 = b["x"] + b["w"] + expand
        y1 = b["y"] + b["h"] + expand
        hit = False
        for a in by_ts.get(t.video_ts_ms, ()):
            cx = a.bbox["x"] + a.bbox["w"] / 2.0
            cy = a.bbox["y"] + a.bbox["h"] / 2.0
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                hit = True
                break
        samples.append((t.video_ts_ms, hit))
    return samples


def action_intervals(
    teacher_dets: list[Detection],
    action_dets: list[Detection],
) -> list[list[int]]:
    """Hysteresis-smoothed intervals during which the teacher was doing one action."""
    return intervals_from_samples(
        action_samples(teacher_dets, action_dets),
        on_ms=ACTION_ON_MS,
        off_ms=ACTION_OFF_MS,
        flicker_samples=ACTION_FLICKER_SAMPLES,
    )
