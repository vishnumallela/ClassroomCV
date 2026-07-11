"""Unit tests for presence/enter-exit derivation, board hysteresis, occupancy."""

from app.events import (
    board_condition,
    bridge_offscreen_gaps,
    door_entry_exit,
    board_intervals_from_samples,
    derive,
    entry_exit_from_intervals,
    occupancy_buckets,
    presence_intervals,
    spatial_heatmap,
)
from app.models import Detection


def _det(ts, track, x=0.5, y=0.5, w=0.1, h=0.3, standing=False, btc=False, raw=None):
    return Detection(
        video_ts_ms=ts,
        raw_track_id=raw if raw is not None else track,
        bbox={"x": x, "y": y, "w": w, "h": h},
        conf=0.9,
        standing=standing,
        back_to_camera=btc,
        track_no=track,
    )


# --------------------------------------------------------------------------- #
# Presence intervals + enter/exit
# --------------------------------------------------------------------------- #


def test_presence_gap_splits_and_events():
    ts = list(range(0, 10_001, 500)) + list(range(20_000, 30_001, 500))
    intervals = presence_intervals(ts)
    assert intervals == [[0, 10_000], [20_000, 30_000]]
    events = entry_exit_from_intervals(intervals, duration_ms=60_000)
    assert events == [
        {"kind": "enter", "ts_ms": 0},
        {"kind": "exit", "ts_ms": 10_000},
        {"kind": "enter", "ts_ms": 20_000},
        {"kind": "exit", "ts_ms": 30_000},
    ]


def test_presence_gap_just_under_threshold_does_not_split():
    assert presence_intervals([0, 4_800, 9_600]) == [[0, 9_600]]


def test_presence_gap_exactly_threshold_splits():
    assert presence_intervals([0, 5_000]) == [[0, 0], [5_000, 5_000]]


def test_no_final_exit_when_present_at_video_end():
    events = entry_exit_from_intervals([[0, 58_000]], duration_ms=60_000)
    assert events == [{"kind": "enter", "ts_ms": 0}]


def test_final_exit_edge_rule_boundary():
    # end >= duration - 5000 -> suppressed
    assert entry_exit_from_intervals([[0, 55_000]], 60_000) == [
        {"kind": "enter", "ts_ms": 0}
    ]
    # end < duration - 5000 -> exit emitted
    assert entry_exit_from_intervals([[0, 54_999]], 60_000) == [
        {"kind": "enter", "ts_ms": 0},
        {"kind": "exit", "ts_ms": 54_999},
    ]


# --------------------------------------------------------------------------- #
# Board hysteresis
# --------------------------------------------------------------------------- #


def _samples(spans, step=200, end=40_000):
    """spans = list of (start, end) where the condition is True."""
    out = []
    for ts in range(0, end + 1, step):
        cond = any(s <= ts <= e for s, e in spans)
        out.append((ts, cond))
    return out


def test_board_run_shorter_than_on_threshold_is_ignored():
    assert board_intervals_from_samples(_samples([(0, 1_800)])) == []


def test_board_on_off_timing():
    iv = board_intervals_from_samples(_samples([(0, 10_000)]))
    # opens after 2s sustained (start backdated to first true), closes 3s after last true
    assert iv == [[0, 10_000]]


def test_board_brief_dropout_does_not_close():
    iv = board_intervals_from_samples(_samples([(0, 10_000), (12_800, 20_000)]))
    # false from 10200..12600 lasts 2600ms < 3000ms off threshold -> one interval
    assert iv == [[0, 20_000]]


def test_board_dropout_at_off_threshold_closes():
    iv = board_intervals_from_samples(_samples([(0, 10_000), (14_000, 20_000)]))
    # false 10200..13800 -> at 13000 elapsed hits 3000ms -> close, then reopen
    assert iv == [[0, 10_000], [14_000, 20_000]]


def test_board_interval_open_at_end_of_samples_is_closed():
    iv = board_intervals_from_samples(_samples([(0, 40_000)]))
    assert iv == [[0, 40_000]]


def test_board_interval_does_not_bridge_detection_gap():
    # Teacher at the board 0..12s, absent (no samples at all) until 5min,
    # back at the board 300..310s. The gap must hard-close the first
    # interval instead of bridging ~5 minutes of absence.
    samples = [(ts, True) for ts in range(0, 12_001, 200)]
    samples += [(ts, True) for ts in range(300_000, 310_001, 200)]
    assert board_intervals_from_samples(samples) == [
        [0, 12_000],
        [300_000, 310_000],
    ]
    # consistency with presence: board time can never exceed presence time
    presence = presence_intervals([ts for ts, _ in samples])
    presence_ms = sum(e - s for s, e in presence)
    board_ms = sum(
        e - s for s, e in board_intervals_from_samples(samples)
    )
    assert board_ms <= presence_ms


def test_isolated_true_samples_across_gap_do_not_open_interval():
    # Two lone true samples 5 minutes apart must not count as a
    # "continuous" 2s on-run (the opening branch must reset at the gap).
    assert board_intervals_from_samples([(0, True), (300_000, True)]) == []


def test_board_gap_below_threshold_still_bridges():
    # A sampling gap just under the 5s presence threshold is not a break;
    # hysteresis (3s off tolerance is irrelevant here: no false samples)
    # keeps one interval.
    samples = [(ts, True) for ts in range(0, 10_001, 200)]
    samples += [(ts, True) for ts in range(14_800, 20_001, 200)]
    assert board_intervals_from_samples(samples) == [[0, 20_000]]


def test_board_condition_geometry():
    poly = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]]
    near = _det(0, 1, x=0.15, y=0.2, w=0.1, h=0.3)  # center x=0.2 inside x-range
    far = _det(0, 1, x=0.8, y=0.6, w=0.1, h=0.3)
    side = _det(0, 1, x=0.45, y=0.1, w=0.2, h=0.3)  # intersects expanded box only
    side_btc = _det(0, 1, x=0.45, y=0.1, w=0.2, h=0.3, btc=True)
    assert board_condition(near, poly) is True
    assert board_condition(far, poly) is False
    assert board_condition(side, poly) is False  # center outside x-range
    # back_to_camera no longer bypasses the x-range gate: on audited footage
    # the bypass fired board_enter with the teacher mid-room, far off-board
    assert board_condition(side_btc, poly) is False


# --------------------------------------------------------------------------- #
# Occupancy bucketing
# --------------------------------------------------------------------------- #


def test_occupancy_buckets_counts_and_teacher_flag():
    dets_by_track = {
        1: [_det(t, 1) for t in (0, 1_000, 6_000)],  # teacher
        2: [_det(t, 2) for t in range(0, 20_000, 1_000)],  # student, whole video
        3: [_det(t, 3) for t in (5_000, 6_000)],  # student, bucket 1 only
    }
    roles = {1: ("teacher", 0.9), 2: ("student", 0.6), 3: ("student", 0.6)}
    buckets = occupancy_buckets(dets_by_track, roles, duration_ms=20_000)
    assert [b["ts_ms"] for b in buckets] == [0, 5_000, 10_000, 15_000]
    assert [b["students"] for b in buckets] == [1, 2, 1, 1]
    assert [b["teacher"] for b in buckets] == [True, True, False, False]


def test_occupancy_covers_full_duration_with_empty_buckets():
    buckets = occupancy_buckets({}, {}, duration_ms=15_000)
    assert [b["ts_ms"] for b in buckets] == [0, 5_000, 10_000]
    assert all(b["students"] == 0 and b["teacher"] is False for b in buckets)


# --------------------------------------------------------------------------- #
# Full derivation incl. degraded no-teacher case
# --------------------------------------------------------------------------- #

_ANALYTICS_KEYS = {
    "teacher_present_ms",
    "teacher_board_ms",
    "entries",
    "exits",
    "presence_intervals",
    "board_intervals",
    "entry_exit",
    "occupancy",
    "avg_students",
    "max_students",
    "heatmap",
    "teacher_pointing_ms",
    "teacher_writing_ms",
    "teacher_board_near_ms",
    "board_interactions",
    "data_quality",
}


def test_derive_full_teacher_at_board():
    poly = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]]
    teacher = [
        _det(t, 1, x=0.15, y=0.15, w=0.1, h=0.4, standing=True, btc=True)
        for t in range(0, 30_001, 200)
    ]
    student = [_det(t, 2, x=0.7, y=0.6) for t in range(0, 60_000, 500)]
    roles = {1: ("teacher", 0.9), 2: ("student", 0.7)}
    events, analytics = derive(
        {1: teacher, 2: student},
        roles,
        duration_ms=60_000,
        zones=[{"kind": "board", "polygon": poly}],
    )
    assert set(analytics.keys()) == _ANALYTICS_KEYS
    assert analytics["presence_intervals"] == [[0, 30_000]]
    assert analytics["teacher_present_ms"] == 30_000
    assert analytics["entries"] == 1 and analytics["exits"] == 1
    assert analytics["board_intervals"] == [[0, 30_000]]
    assert analytics["teacher_board_ms"] == 30_000
    assert analytics["max_students"] == 1
    kinds = [(e["kind"], e["video_ts_ms"]) for e in events]
    assert ("enter", 0) in kinds and ("exit", 30_000) in kinds
    assert ("board_enter", 0) in kinds and ("board_leave", 30_000) in kinds
    assert all(e["track_no"] == 1 for e in events)
    # events sorted by timestamp
    assert [e["video_ts_ms"] for e in events] == sorted(
        e["video_ts_ms"] for e in events
    )


def test_derive_no_teacher_degrades_gracefully():
    dets = {
        1: [_det(t, 1) for t in range(0, 30_000, 500)],
        2: [_det(t, 2) for t in range(0, 30_000, 500)],
    }
    roles = {1: ("unknown", None), 2: ("unknown", None)}
    events, analytics = derive(
        dets,
        roles,
        duration_ms=30_000,
        zones=[{"kind": "board", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3]]}],
    )
    assert events == []
    assert analytics["teacher_present_ms"] == 0
    assert analytics["teacher_board_ms"] == 0  # board zone drawn but no teacher
    assert analytics["entries"] == 0 and analytics["exits"] == 0
    assert analytics["presence_intervals"] == []
    assert analytics["board_intervals"] == []
    assert analytics["entry_exit"] == []
    # occupancy still useful: unknowns are counted as students
    assert analytics["max_students"] == 2


def test_derive_no_board_zone_means_null_board_ms():
    dets = {1: [_det(t, 1, standing=True) for t in range(0, 30_000, 500)]}
    roles = {1: ("teacher", 0.9)}
    _, analytics = derive(dets, roles, duration_ms=30_000, zones=[])
    assert analytics["teacher_board_ms"] is None
    assert analytics["board_intervals"] == []


# --------------------------------------------------------------------------- #
# Door-based entry/exit (windowed)
# --------------------------------------------------------------------------- #

DOOR = [[0.0, 0.2], [0.1, 0.2], [0.1, 0.8], [0.0, 0.8]]


def test_door_crossing_counted_when_edge_sample_is_within_window():
    # Teacher walks toward the door, tracking drops her 2s before the gap edge.
    dets = [_det(ts, 1, x=0.5) for ts in range(0, 10_000, 1_000)]
    dets += [_det(10_000, 1, x=0.06), _det(11_000, 1, x=0.06)]  # near door
    dets += [_det(12_000, 1, x=0.3), _det(13_000, 1, x=0.35)]   # walked away, lost
    intervals = [[0, 13_000], [40_000, 60_000]]
    dets += [_det(40_000, 1, x=0.07)]  # reappears at the door
    dets += [_det(ts, 1, x=0.5) for ts in range(41_000, 58_000, 1_000)]
    dets += [_det(59_000, 1, x=0.07), _det(60_000, 1, x=0.06)]  # leaves via door
    events = door_entry_exit(dets, intervals, [DOOR], duration_ms=120_000)
    kinds = [e["kind"] for e in events]
    # Crossings-only: present-at-start is not an entry, so the first event is the
    # exit at 13s (door-adjacent via the window), then re-enter at 40s, exit at 60s.
    assert kinds == ["exit", "enter", "exit"]
    assert events[0]["ts_ms"] == 13_000
    assert events[1]["ts_ms"] == 40_000


def test_mid_room_occlusion_gap_produces_no_door_events():
    dets = [_det(ts, 1, x=0.5) for ts in range(0, 20_000, 1_000)]
    dets += [_det(ts, 1, x=0.55) for ts in range(30_000, 50_000, 1_000)]
    intervals = [[0, 19_000], [30_000, 49_000]]
    events = door_entry_exit(dets, intervals, [DOOR], duration_ms=120_000)
    kinds = [e["kind"] for e in events]
    # No crossings at all: present-at-start is not an entry, and the mid-room gap
    # is an occlusion, not a door crossing.
    assert kinds == []


def test_bridge_offscreen_gap_away_from_door():
    # A short presence gap whose bounding samples are both away from the door
    # is a camera blind spot (corner desk): presence is bridged continuous.
    dets = [_det(ts, 1, x=0.5) for ts in range(0, 10_001, 500)]
    dets += [_det(ts, 1, x=0.5) for ts in range(18_000, 30_001, 500)]
    intervals = [[0, 10_000], [18_000, 30_000]]
    assert bridge_offscreen_gaps(intervals, dets, [DOOR]) == [[0, 30_000]]


def test_no_bridge_when_vanish_is_at_the_door():
    # Vanishing at the door is a real crossing and must not be bridged away.
    dets = [_det(ts, 1, x=0.5) for ts in range(0, 9_001, 500)]
    dets += [_det(10_000, 1, x=0.05)]  # last sample at the door
    dets += [_det(ts, 1, x=0.5) for ts in range(18_000, 30_001, 500)]
    intervals = [[0, 10_000], [18_000, 30_000]]
    assert bridge_offscreen_gaps(intervals, dets, [DOOR]) == [[0, 10_000], [18_000, 30_000]]


def test_no_bridge_when_gap_too_long():
    dets = [_det(ts, 1, x=0.5) for ts in range(0, 10_001, 500)]
    dets += [_det(ts, 1, x=0.5) for ts in range(30_000, 40_001, 500)]
    intervals = [[0, 10_000], [30_000, 40_000]]  # 20s gap > bridge window
    assert bridge_offscreen_gaps(intervals, dets, [DOOR]) == [[0, 10_000], [30_000, 40_000]]


# --------------------------------------------------------------------------- #
# Spatial heatmap
# --------------------------------------------------------------------------- #


def test_spatial_heatmap_splits_teacher_and_students_by_cell():
    # bbox x/y is the top-left corner, so a w=0.1,h=0.3 box at (0.5,0.5) centers
    # at (0.55, 0.65) -> col 5, row 6; the corner student clamps to (9, 9).
    dbt = {
        1: [_det(ts, 1, x=0.5, y=0.5) for ts in range(0, 5_001, 500)],
        2: [_det(ts, 2, x=0.95, y=0.95) for ts in range(0, 5_001, 500)],
    }
    roles = {1: ("teacher", 0.9), 2: ("student", 0.5)}
    hm = spatial_heatmap(dbt, roles, grid_w=10, grid_h=10)
    assert hm["grid_w"] == 10 and hm["grid_h"] == 10
    assert len(hm["teacher"]) == 100 and len(hm["students"]) == 100
    assert hm["teacher"][6 * 10 + 5] == 11 and sum(hm["teacher"]) == 11
    assert hm["students"][9 * 10 + 9] == 11 and sum(hm["students"]) == 11
    # channels are disjoint: no teacher mass in the student cell.
    assert hm["teacher"][9 * 10 + 9] == 0


def test_spatial_heatmap_unknown_counts_as_students():
    dbt = {3: [_det(0, 3, x=0.1, y=0.1)]}
    hm = spatial_heatmap(dbt, {3: ("unknown", None)}, grid_w=4, grid_h=4)
    assert sum(hm["students"]) == 1 and sum(hm["teacher"]) == 0


# --------------------------------------------------------------------------- #
# No-door short-gap bridge
# --------------------------------------------------------------------------- #


def test_bridge_short_gaps_merges_brief_and_keeps_long():
    from app.events import bridge_short_gaps

    # 6s gap (brief) bridges; 20s gap (real absence) stays split.
    ivs = [[0, 10_000], [16_000, 30_000], [50_000, 60_000]]
    assert bridge_short_gaps(ivs) == [[0, 30_000], [50_000, 60_000]]


def test_derive_no_door_bridges_blind_spot_no_phantom_crossing():
    # Teacher present 0-10s, 6s blind-spot gap, present 16-58s. With no door,
    # the brief gap must NOT become an exit+enter and presence must include it.
    dets = [_det(t, 1, standing=True) for t in range(0, 10_001, 500)]
    dets += [_det(t, 1, standing=True) for t in range(16_000, 58_001, 500)]
    _events, analytics = derive(
        {1: dets}, {1: ("teacher", 0.9)}, duration_ms=60_000, zones=[]
    )
    assert analytics["presence_intervals"] == [[0, 58_000]]
    assert analytics["entries"] == 1 and analytics["exits"] == 0
