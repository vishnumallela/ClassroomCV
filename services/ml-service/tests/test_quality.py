"""Unit tests for the additive data-quality assessment (app/quality.py)."""

from app import quality
from app.models import Detection


def _det(ts, track, raw=None, x=0.5, y=0.5, w=0.05, h=0.1):
    return Detection(
        video_ts_ms=ts,
        raw_track_id=raw if raw is not None else track,
        bbox={"x": x, "y": y, "w": w, "h": h},
        conf=0.9,
        standing=False,
        back_to_camera=False,
        track_no=track,
    )


def _clean_room(n_students=10, frames=60, step=1000):
    """n students, each one clean identity (raw_id == track_no), plus a teacher."""
    dets_by_track: dict[int, list[Detection]] = {}
    for s in range(1, n_students + 1):
        dets_by_track[s] = [_det(t * step, s, x=0.1 + 0.05 * s) for t in range(frames)]
    dets_by_track[999] = [_det(t * step, 999, x=0.9) for t in range(frames)]
    roles = {s: ("student", 0.7) for s in range(1, n_students + 1)}
    roles[999] = ("teacher", 0.8)
    return dets_by_track, roles


def test_clean_room_is_high_confidence():
    dets_by_track, roles = _clean_room(n_students=10, frames=60)
    q = quality.assess(dets_by_track, roles, duration_ms=60_000, teacher_confidence=0.8)
    assert q["fragmentation"] == 1.0
    assert q["coverage"] == 1.0
    assert q["confidence"]["identity"] == "high"
    assert q["confidence"]["coverage"] == "high"
    assert q["confidence"]["teacher"] == "high"
    assert q["confidence"]["overall"] == "high"
    assert q["notes"] == []


def test_fragmentation_downgrades_identity_confidence():
    # Two final identities, but the merge absorbed many raw tracker ids into
    # each (10 raw ids across 2 identities -> fragmentation 5.0 -> low).
    dets_by_track: dict[int, list[Detection]] = {1: [], 2: []}
    raw = 0
    for identity in (1, 2):
        for _chunk in range(5):  # 5 raw fragments per identity
            for t in range(4):
                dets_by_track[identity].append(_det(raw * 100 + t * 1000, identity, raw=raw))
            raw += 1
    roles = {1: ("student", 0.6), 2: ("student", 0.6)}
    q = quality.assess(dets_by_track, roles, duration_ms=20_000, teacher_confidence=None)
    assert q["raw_tracks"] == 10
    assert q["identities"] == 2
    assert q["fragmentation"] == 5.0
    assert q["confidence"]["identity"] == "low"
    assert any("fragment" in n.lower() for n in q["notes"])


def test_low_coverage_flagged():
    # One student present only in the first and last of a long span (big gap).
    dets_by_track = {
        1: [_det(0, 1), _det(60_000, 1)],
    }
    roles = {1: ("student", None)}
    q = quality.assess(dets_by_track, roles, duration_ms=60_000, teacher_confidence=None)
    assert q["coverage"] < quality.COVERAGE_LOW
    assert q["confidence"]["coverage"] == "low"
    assert any("coverage" in n.lower() or "floor" in n.lower() for n in q["notes"])


def test_no_teacher_is_low_teacher_tier():
    dets_by_track, roles = _clean_room(n_students=6, frames=30)
    roles = {k: ("student", None) for k in dets_by_track}  # nobody is the teacher
    q = quality.assess(dets_by_track, roles, duration_ms=30_000, teacher_confidence=None)
    assert q["confidence"]["teacher"] == "low"
    assert q["confidence"]["overall"] == "low"  # weakest link


def test_narrow_teacher_margin_is_low_tier_with_note():
    dets_by_track, roles = _clean_room(n_students=4, frames=30)
    q = quality.assess(dets_by_track, roles, duration_ms=30_000, teacher_confidence=0.51)
    assert q["confidence"]["teacher"] == "low"
    assert any("margin" in n.lower() or "tentative" in n.lower() for n in q["notes"])


def test_empty_room_does_not_crash():
    q = quality.assess({}, {}, duration_ms=0, teacher_confidence=None)
    assert q["detections"] == 0
    assert q["confidence"]["overall"] == "low"
