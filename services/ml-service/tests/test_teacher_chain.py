"""Unit tests for teacher timeline stitching (app.teacher_chain).

The stitcher reclaims the teacher's trajectory from tracker id-steals and
evicts student detections wrongly merged into her identity. Scenarios mirror
the real-footage failures the module was built for.
"""

from app import teacher_chain as tc
from app.models import Detection


def _d(ts, raw, track, x, y=0.45, h=0.45, standing=True):
    return Detection(
        video_ts_ms=ts,
        raw_track_id=raw,
        bbox={"x": x - 0.05, "y": y - h / 2, "w": 0.1, "h": h},
        conf=0.9,
        standing=standing,
        back_to_camera=False,
        track_no=track,
    )


def test_reclaims_continuation_held_by_student_track():
    # Teacher identity (track 1) is a mobile tall fragment raw 10 across 0..30s;
    # her continuation was stolen by student track 2 as raw 20 (30.5..50s),
    # also tall and mobile. The chain claims both into the teacher.
    dbt = {1: [], 2: []}
    for ts in range(0, 30_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.2 + 0.5 * ts / 30_000))
    for ts in range(30_500, 50_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.7 - 0.4 * (ts - 30_500) / 19_500))
    claims, _ = tc.stitch_teacher(1, dbt)
    claimed = {c.fragment.raw_id for c in claims}
    assert 10 in claimed and 20 in claimed


def test_rejects_seated_static_student_at_the_handoff():
    # Teacher raw 10 (track 1) ends near x=0.8; a seated student raw 20
    # (track 2) sits motionless right next to that spot. Proximity alone would
    # grab it, but a sustained no-motion span is never the teacher.
    dbt = {1: [], 2: []}
    for ts in range(0, 20_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.5 + 0.3 * ts / 20_000))  # walks to 0.8
    for ts in range(20_500, 45_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.82))  # static, tall, adjacent
    claims, _ = tc.stitch_teacher(1, dbt)
    assert 20 not in {c.fragment.raw_id for c in claims}


def test_keeps_own_earlier_fragment_when_seed_is_the_later_one():
    # Teacher identity holds two of her own fragments: raw 10 (0..15s) and the
    # longer raw 11 (20..50s). Seeding from the longer fragment must not evict
    # the earlier one — the backward walk reclaims it.
    dbt = {1: []}
    for ts in range(0, 15_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.3 + 0.3 * ts / 15_000))
    for ts in range(20_000, 50_001, 500):
        dbt[1].append(_d(ts, 11, 1, 0.6 - 0.3 * (ts - 20_000) / 30_000))
    claims, evictions = tc.stitch_teacher(1, dbt)
    claimed = {c.fragment.raw_id for c in claims}
    assert 10 in claimed and 11 in claimed
    assert all(f.raw_id != 10 for f, _lo, _hi in evictions)


def _walk_then_sit(dbt, *, end_x=0.8):
    """Teacher raw 10 (track 1) walks 0..20s, then her box collapses over the
    final second (she sits: h 0.45 -> 0.20) and the raw id dies."""
    for ts in range(0, 19_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.5 + (end_x - 0.5) * ts / 19_000))
    for i, h in enumerate((0.38, 0.30, 0.20)):
        dbt[1].append(_d(19_500 + i * 500, 10, 1, end_x, h=h, standing=False))


def test_sit_claims_seated_fragment_born_at_collapse():
    # After the sit-down collapse, a FRESH raw id (20, student track 2) is born
    # 0.5s later at her spot, seated-sized and static. CONTINUE/START/RECOVER
    # all reject it (sub-height, static); the SIT branch claims it.
    dbt = {1: [], 2: []}
    _walk_then_sit(dbt)
    for ts in range(21_000, 36_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.8, h=0.22, standing=False))
    claims, _ = tc.stitch_teacher(1, dbt)
    assert 20 in {c.fragment.raw_id for c in claims}


def test_sit_ignores_preexisting_seated_fragment():
    # The desk-visit steal shape: she collapses NEXT TO a pupil whose fragment
    # has existed since ts=0. Birth-at-the-collapse is required, so the
    # long-lived seated fragment is not claimed.
    dbt = {1: [], 2: []}
    _walk_then_sit(dbt)
    for ts in range(0, 36_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.82, h=0.22, standing=False))
    claims, _ = tc.stitch_teacher(1, dbt)
    assert 20 not in {c.fragment.raw_id for c in claims}


def test_sit_requires_the_chain_to_die_in_a_collapse():
    # Her chain ends at FULL height (she walked out of tracking, no sit-down);
    # a seated fragment born moments later nearby must not be claimed.
    dbt = {1: [], 2: []}
    for ts in range(0, 20_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.5 + 0.3 * ts / 20_000))
    for ts in range(21_000, 36_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.8, h=0.22, standing=False))
    claims, _ = tc.stitch_teacher(1, dbt)
    assert 20 not in {c.fragment.raw_id for c in claims}


def test_sit_rejects_fragment_born_too_late():
    # Collapse happens, but the seated fragment appears 5s later — outside the
    # SIT birth window; someone else took the seat.
    dbt = {1: [], 2: []}
    _walk_then_sit(dbt)
    for ts in range(25_500, 40_001, 500):
        dbt[2].append(_d(ts, 20, 2, 0.8, h=0.22, standing=False))
    claims, _ = tc.stitch_teacher(1, dbt)
    assert 20 not in {c.fragment.raw_id for c in claims}


def test_evicts_a_wrongly_merged_seated_student_fragment():
    # A seated corner student raw 30 was merged into the teacher identity.
    # The chain claims the real teacher fragment and evicts the student.
    dbt = {1: []}
    for ts in range(0, 40_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.2 + 0.6 * ts / 40_000))  # mobile teacher
    for ts in range(5_000, 35_001, 500):
        dbt[1].append(_d(ts, 30, 1, 0.9, h=0.2, standing=False))  # static short kid
    _claims, evictions = tc.stitch_teacher(1, dbt)
    assert any(f.raw_id == 30 for f, _lo, _hi in evictions)
