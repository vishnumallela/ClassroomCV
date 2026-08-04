"""Event + analytics derivation — a thin orchestrator over app.heuristics.

Luminary reports exactly three teacher KPIs (entry/exit, board time,
heatmap); every rule that computes them lives in app/heuristics.py — the
one file meant to be tuned and extended. This module just wires the
role-labelled detections through those rules and shapes the AnalysisResult
payload. Per-student analytics (occupancy buckets, avg/max students, a
students heatmap grid) were removed in the 2026-08 KPI slimming, which also
keeps the stored analytics row light.

Never raises on the no-teacher case: teacher analytics become zeros/null.
"""

from __future__ import annotations

from typing import Optional

from app import quality
from app.heuristics import (
    board_condition,
    board_intervals_from_samples,
    bridge_offscreen_gaps,
    bridge_short_gaps,
    door_entry_exit,
    entry_exit_from_intervals,
    presence_intervals,
    teacher_heatmap,
)
from app.models import Detection


def derive(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    duration_ms: int,
    zones: list[dict],
) -> tuple[list[dict], dict]:
    """Return (events, analytics) dicts matching the SPEC AnalysisResult shapes."""
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
        presence = presence_intervals([d.video_ts_ms for d in teacher_dets])
        if door_polygons:
            presence = bridge_offscreen_gaps(presence, teacher_dets, door_polygons)
            entry_exit = door_entry_exit(
                teacher_dets, presence, door_polygons, duration_ms
            )
        else:
            # No door to distinguish a crossing from a blind spot: bridge brief
            # gaps so a tracking dropout is not reported as leaving and
            # re-entering the room.
            presence = bridge_short_gaps(presence)
            entry_exit = entry_exit_from_intervals(presence, duration_ms)
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
            board_iv = board_intervals_from_samples(samples)
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

    # Additive, never mutates a derived number: an honest confidence report so
    # the dashboard can say how much each figure can be trusted.
    teacher_conf = (
        roles_map.get(teacher_no, ("", None))[1] if teacher_no is not None else None
    )
    data_quality = quality.assess(
        dets_by_track,
        roles_map,
        duration_ms,
        teacher_confidence=teacher_conf,
    )

    events.sort(key=lambda e: (e["video_ts_ms"], e["kind"]))

    analytics = {
        "teacher_present_ms": teacher_present_ms,
        "teacher_board_ms": teacher_board_ms,
        "entries": sum(1 for e in entry_exit if e["kind"] == "enter"),
        "exits": sum(1 for e in entry_exit if e["kind"] == "exit"),
        "presence_intervals": presence,
        "board_intervals": board_iv,
        "entry_exit": entry_exit,
        "heatmap": heatmap,
        "data_quality": data_quality,
    }
    return events, analytics
