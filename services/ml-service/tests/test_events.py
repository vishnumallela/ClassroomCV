"""Unit tests for presence/enter-exit derivation, board hysteresis, occupancy."""

from app.events import (
    board_condition,
    board_intervals_from_samples,
    derive,
    entry_exit_from_intervals,
    occupancy_buckets,
    presence_intervals,
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
    assert board_condition(side, poly) is False  # center outside x-range, facing camera
    assert board_condition(side_btc, poly) is True  # back to camera counts


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
