"""Event + analytics derivation from merged, role-labelled detections.

- Teacher presence: union of the teacher identity's detection timestamps;
  gaps >= 5000 ms split intervals (exit at gap start / enter at gap end).
  Video-start presence => enter at first_ms. Final exit only when the last
  presence ends before duration - 5000 ms.
- Board intervals (only when a board zone exists): per-sample condition =
  teacher bbox intersects polygon bbox expanded by 12% of the frame AND
  bbox-center within the polygon x-range. Hysteresis: 2 s sustained ON to
  open, 3 s sustained OFF to close; a sampling gap >= 5 s (teacher absent,
  no samples) hard-closes the interval so board time never bridges an
  absence.
- Occupancy: 5000 ms buckets; students = distinct non-teacher identities
  detected in the bucket; teacher = bool.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Optional

from app.geometry import bboxes_intersect, expand_bbox, polygon_bbox
from app.models import Detection

PRESENCE_GAP_MS = 5_000
END_MARGIN_MS = 5_000
BUCKET_MS = 5_000
# A presence gap no longer than this whose bounding detections are BOTH away
# from every door is a camera blind spot (the teacher stepped to a near-camera
# corner desk, out of view), not a door exit: presence is bridged so a walk
# past the door followed by an off-camera moment is not mislabelled a crossing.
OFFSCREEN_BRIDGE_MS = 12_000
BOARD_EXPAND = 0.12  # 12% of frame (normalized units) on every side
BOARD_ON_MS = 2_000
BOARD_OFF_MS = 3_000
# Opening tolerance: a candidate run survives brief false flickers (a single
# occluded sample) instead of resetting; without this, opening needs ~10
# CONSECUTIVE true samples at 5 fps and short board stints never register.
BOARD_FLICKER_SAMPLES = 2
BOARD_FLICKER_BUDGET_MS = 600


# --------------------------------------------------------------------------- #
# Presence intervals + enter/exit events
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
    """[{kind:'enter'|'exit', ts_ms}] per SPEC edge rules."""
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


# Window anchoring the door test at the presence edge (the vanish/appear
# samples). 4s reached back far enough to catch the teacher merely WALKING
# PAST the door on her way out of frame (pixel-verified false exit at 250.8s);
# 2s still covers door-frame occlusion dropping the track a beat early.
DOOR_WINDOW_MS = 2_000
DOOR_EXPAND = 0.15


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


# --------------------------------------------------------------------------- #
# Board intervals (hysteresis)
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


def board_intervals_from_samples(
    samples: list[tuple[int, bool]],
    on_ms: int = BOARD_ON_MS,
    off_ms: int = BOARD_OFF_MS,
    gap_ms: int = PRESENCE_GAP_MS,
) -> list[list[int]]:
    """Hysteresis state machine over (ts, condition) samples (time-sorted).

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
                if ts - run_start >= on_ms and false_run_ms <= BOARD_FLICKER_BUDGET_MS:
                    on = True
                    start = run_start
                    last_true = ts
            elif run_start is not None:
                false_streak += 1
                if prev_ts is not None:
                    false_run_ms += ts - prev_ts
                if false_streak >= BOARD_FLICKER_SAMPLES or false_run_ms > BOARD_FLICKER_BUDGET_MS:
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
# Occupancy
# --------------------------------------------------------------------------- #


def occupancy_buckets(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    duration_ms: int,
    bucket_ms: int = BUCKET_MS,
) -> list[dict]:
    """[{ts_ms, students, teacher}] per bucket over the whole video.

    'students' counts distinct non-teacher identities (students AND unknowns,
    so occupancy stays useful in the degraded no-teacher case).
    """
    max_ts = max(
        (d.video_ts_ms for dets in dets_by_track.values() for d in dets),
        default=0,
    )
    # Detections at ts == duration clamp into the last bucket; only extend the
    # range when detections genuinely exceed the reported duration.
    span = duration_ms if max_ts <= duration_ms else max_ts + 1
    n_buckets = max(1, math.ceil(span / bucket_ms)) if span > 0 else 0
    if n_buckets == 0:
        return []

    students: list[set[int]] = [set() for _ in range(n_buckets)]
    teacher: list[bool] = [False] * n_buckets
    for track_no, dets in dets_by_track.items():
        role = roles_map.get(track_no, ("unknown", None))[0]
        for d in dets:
            idx = min(n_buckets - 1, d.video_ts_ms // bucket_ms)
            if role == "teacher":
                teacher[idx] = True
            else:
                students[idx].add(track_no)

    return [
        {"ts_ms": i * bucket_ms, "students": len(students[i]), "teacher": teacher[i]}
        for i in range(n_buckets)
    ]


# --------------------------------------------------------------------------- #
# Spatial heatmap
# --------------------------------------------------------------------------- #

HEATMAP_GRID_W = 32
HEATMAP_GRID_H = 18  # 16:9, matches the CCTV aspect


def spatial_heatmap(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    grid_w: int = HEATMAP_GRID_W,
    grid_h: int = HEATMAP_GRID_H,
) -> dict:
    """Row-major dwell histograms of bbox centers, split teacher vs students.

    Each detection drops one sample into the grid cell its bbox center falls
    in; sampled at a fixed rate, a cell's count is proportional to time spent
    there, so the teacher grid traces her teaching path (board, aisles, the
    desks she visits) and the students grid shows seating density. Non-teacher
    identities (students AND unknowns) go in the students grid, matching
    occupancy_buckets.
    """
    teacher = [0] * (grid_w * grid_h)
    students = [0] * (grid_w * grid_h)
    for track_no, dets in dets_by_track.items():
        role = roles_map.get(track_no, ("unknown", None))[0]
        grid = teacher if role == "teacher" else students
        for d in dets:
            cx = d.bbox["x"] + d.bbox["w"] / 2.0
            cy = d.bbox["y"] + d.bbox["h"] / 2.0
            gx = min(grid_w - 1, max(0, int(cx * grid_w)))
            gy = min(grid_h - 1, max(0, int(cy * grid_h)))
            grid[gy * grid_w + gx] += 1
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "teacher": teacher,
        "students": students,
    }


# --------------------------------------------------------------------------- #
# Full derivation
# --------------------------------------------------------------------------- #


def derive(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    duration_ms: int,
    zones: list[dict],
) -> tuple[list[dict], dict]:
    """Return (events, analytics) dicts matching the SPEC AnalysisResult shapes.

    Never raises on the no-teacher case: teacher analytics become zeros/null.
    """
    board_polygon = next(
        (z["polygon"] for z in zones if z.get("kind") == "board"), None
    )
    door_polygons = [z["polygon"] for z in zones if z.get("kind") == "door"]
    teacher_no = next(
        (t for t, (role, _) in roles_map.items() if role == "teacher"), None
    )

    events: list[dict] = []
    presence: list[list[int]] = []
    entry_exit: list[dict] = []
    teacher_dets: list[Detection] = []

    if teacher_no is not None and dets_by_track.get(teacher_no):
        teacher_dets = sorted(
            dets_by_track[teacher_no], key=lambda d: d.video_ts_ms
        )
        presence = presence_intervals([d.video_ts_ms for d in teacher_dets])
        if door_polygons:
            presence = bridge_offscreen_gaps(presence, teacher_dets, door_polygons)
            entry_exit = door_entry_exit(
                teacher_dets, presence, door_polygons, duration_ms
            )
        else:
            entry_exit = entry_exit_from_intervals(presence, duration_ms)
        events.extend(
            {"kind": e["kind"], "video_ts_ms": e["ts_ms"], "track_no": teacher_no}
            for e in entry_exit
        )

    teacher_present_ms = sum(end - start for start, end in presence)

    board_iv: list[list[int]] = []
    teacher_board_ms: Optional[int] = None
    if board_polygon is not None:
        teacher_board_ms = 0
        if teacher_dets:
            samples = [
                (d.video_ts_ms, board_condition(d, board_polygon))
                for d in teacher_dets
            ]
            board_iv = board_intervals_from_samples(samples)
            teacher_board_ms = sum(end - start for start, end in board_iv)
            for start, end in board_iv:
                events.append(
                    {"kind": "board_enter", "video_ts_ms": start, "track_no": teacher_no}
                )
                events.append(
                    {"kind": "board_leave", "video_ts_ms": end, "track_no": teacher_no}
                )

    occupancy = occupancy_buckets(dets_by_track, roles_map, duration_ms)
    counts = [b["students"] for b in occupancy]
    avg_students = round(sum(counts) / len(counts), 2) if counts else 0.0
    max_students = max(counts) if counts else 0
    heatmap = spatial_heatmap(dets_by_track, roles_map)

    events.sort(key=lambda e: (e["video_ts_ms"], e["kind"]))

    analytics = {
        "teacher_present_ms": teacher_present_ms,
        "teacher_board_ms": teacher_board_ms,
        "entries": sum(1 for e in entry_exit if e["kind"] == "enter"),
        "exits": sum(1 for e in entry_exit if e["kind"] == "exit"),
        "presence_intervals": presence,
        "board_intervals": board_iv,
        "entry_exit": entry_exit,
        "occupancy": occupancy,
        "avg_students": avg_students,
        "max_students": max_students,
        "heatmap": heatmap,
    }
    return events, analytics
