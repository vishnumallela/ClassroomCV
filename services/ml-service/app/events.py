"""Event + analytics derivation — a thin orchestrator over app.heuristics.

Luminary reports four teacher KPIs (entry/exit, board time, heatmap, and the
pointing/writing actions); every rule that computes them lives in
app/heuristics.py — the one file meant to be tuned and extended. This module
just wires the role-labelled detections through those rules and shapes the
AnalysisResult payload. Per-student analytics (occupancy buckets, avg/max students, a
students heatmap grid) were removed in the 2026-08 KPI slimming, which also
keeps the stored analytics row light.

Never raises on the no-teacher case: teacher analytics become zeros/null.
"""

from __future__ import annotations

from typing import Optional

from app.heuristics import (
    action_intervals,
    board_condition,
    bridge_offscreen_gaps,
    bridge_short_gaps,
    classify_presence,
    door_entry_exit,
    entry_exit_from_intervals,
    intervals_from_samples,
    presence_intervals,
    teacher_heatmap,
)
from app.config import get_settings
from app.models import CLASS_POINTING, CLASS_WRITING, Detection


def derive(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    duration_ms: int,
    zones: list[dict],
    all_detections: Optional[list[Detection]] = None,
) -> tuple[list[dict], dict]:
    """Return (events, analytics) dicts matching the SPEC AnalysisResult shapes.

    `all_detections` carries the classes that are NOT the teacher's own box —
    currently `pointing` and `writing`. It is optional so /rederive, which
    replays stored rows and has only her boxes, still works: the action KPIs
    then come back None (unknown) rather than 0 (measured as absent). Those two
    are different claims and the dashboard must not confuse them.
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

    # --- KPI 1: teacher entry / exit -------------------------------------- #
    if teacher_no is not None and dets_by_track.get(teacher_no):
        teacher_dets = sorted(
            dets_by_track[teacher_no], key=lambda d: d.video_ts_ms
        )
        # One pass decides both: gaps she crossed the door through (direction
        # from her movement relative to it) or stayed away beyond the buffer
        # are exits and entries; the rest are occlusions, bridged into presence.
        presence, entry_exit = classify_presence(
            presence_intervals([d.video_ts_ms for d in teacher_dets]),
            teacher_dets,
            door_polygons,
            duration_ms,
        )
        events.extend(
            {"kind": e["kind"], "video_ts_ms": e["ts_ms"], "track_no": teacher_no}
            for e in entry_exit
        )

    # Presence underpins entry/exit and the timeline; it is support data for
    # KPI 1, not its own headline metric.
    teacher_present_ms = sum(end - start for start, end in presence)

    # --- KPI 2: teacher board time ----------------------------------------- #
    board_iv: list[list[int]] = []
    teacher_board_ms: Optional[int] = None
    if board_polygon is not None:
        teacher_board_ms = 0
        if teacher_dets:
            samples = [
                (d.video_ts_ms, board_condition(d, board_polygon))
                for d in teacher_dets
            ]
            board_iv = intervals_from_samples(samples)
            teacher_board_ms = sum(end - start for start, end in board_iv)
            for start, end in board_iv:
                events.append(
                    {"kind": "board_enter", "video_ts_ms": start, "track_no": teacher_no}
                )
                events.append(
                    {"kind": "board_leave", "video_ts_ms": end, "track_no": teacher_no}
                )

    # --- KPI 3: teacher heatmap -------------------------------------------- #
    heatmap = teacher_heatmap(teacher_dets)

    # --- KPI 4: teacher actions (pointing / writing) ------------------------ #
    # None, not 0, when the action classes were never offered: /rederive replays
    # stored detections, which are teacher-only, so it cannot recompute these.
    # Reporting 0 there would turn "we did not look" into "she never wrote".
    pointing_iv: list[list[int]] = []
    writing_iv: list[list[int]] = []
    teacher_pointing_ms: Optional[int] = None
    teacher_writing_ms: Optional[int] = None
    if all_detections is not None:
        teacher_pointing_ms = 0
        teacher_writing_ms = 0
        if teacher_dets:
            # action_conf, not DETECT_FLOOR: the detector emits everything down
            # to 0.15 so one pass can serve every consumer, and a KPI measured
            # in seconds must not be built from boxes the model barely believes.
            action_conf = get_settings().action_conf
            for cls, kind in ((CLASS_POINTING, "pointing"), (CLASS_WRITING, "writing")):
                dets = [
                    d for d in all_detections if d.cls == cls and d.conf >= action_conf
                ]
                iv = action_intervals(teacher_dets, dets)
                total = sum(end - start for start, end in iv)
                if cls == CLASS_POINTING:
                    pointing_iv, teacher_pointing_ms = iv, total
                else:
                    writing_iv, teacher_writing_ms = iv, total
                for start, end in iv:
                    events.append(
                        {"kind": f"{kind}_start", "video_ts_ms": start, "track_no": teacher_no}
                    )
                    events.append(
                        {"kind": f"{kind}_end", "video_ts_ms": end, "track_no": teacher_no}
                    )

    events.sort(key=lambda e: (e["video_ts_ms"], e["kind"]))

    analytics = {
        "teacher_present_ms": teacher_present_ms,
        "teacher_board_ms": teacher_board_ms,
        "entries": sum(1 for e in entry_exit if e["kind"] == "enter"),
        "exits": sum(1 for e in entry_exit if e["kind"] == "exit"),
        "presence_intervals": presence,
        "board_intervals": board_iv,
        "teacher_pointing_ms": teacher_pointing_ms,
        "teacher_writing_ms": teacher_writing_ms,
        "pointing_intervals": pointing_iv,
        "writing_intervals": writing_iv,
        "entry_exit": entry_exit,
        "heatmap": heatmap,
        # Attached by the caller (jobs.derive_result), which holds the teacher
        # track's own coverage/confidence signals.
        "data_quality": None,
    }
    return events, analytics
