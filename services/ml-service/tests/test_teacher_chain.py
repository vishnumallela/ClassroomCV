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


def test_find_switch_index_flags_a_fragment_that_relocates():
    # The tracker id sits on the teacher at cx~0.36, then slides onto a pupil at
    # cx~0.18 and stays there (measured shape of a real intra-fragment switch).
    dets = [_d(ts, 10, 1, 0.36) for ts in range(0, 20_000, 200)]
    dets += [_d(ts, 10, 1, 0.18) for ts in range(20_000, 36_000, 200)]
    cut = tc.find_switch_index(dets)
    assert cut is not None
    # the split lands at the relocation, not somewhere arbitrary
    assert abs(dets[cut].video_ts_ms - 20_000) <= 2_000


def test_find_switch_index_ignores_a_fragment_that_stays_put():
    # A teacher working at the board drifts a little but never relocates.
    dets = [
        _d(ts, 10, 1, 0.36 + (0.02 if (ts // 1000) % 2 else -0.02))
        for ts in range(0, 36_000, 200)
    ]
    assert tc.find_switch_index(dets) is None


def test_find_switch_index_needs_enough_detections_on_both_sides():
    # A long run plus a couple of stray frames elsewhere is noise, not a switch.
    dets = [_d(ts, 10, 1, 0.36) for ts in range(0, 20_000, 200)]
    dets += [_d(ts, 10, 1, 0.05) for ts in range(20_000, 20_800, 200)]
    assert tc.find_switch_index(dets) is None


def test_hole_fill_reclaims_her_own_fragment_the_walk_stepped_over():
    # Her identity holds three fragments; the greedy walk chains raw 10 -> raw 12
    # and steps over raw 11, which sits in the hole between them. Unclaimed
    # fragments of her identity are evicted to a student track, so without the
    # hole fill she becomes "Student N" for those seconds.
    dbt = {1: []}
    for ts in range(0, 20_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.2 + 0.3 * ts / 20_000))
    for ts in range(21_000, 27_000, 500):  # the stepped-over middle
        dbt[1].append(_d(ts, 11, 1, 0.52 + 0.1 * (ts - 21_000) / 6_000))
    for ts in range(28_000, 50_001, 500):
        dbt[1].append(_d(ts, 12, 1, 0.62 + 0.2 * (ts - 28_000) / 22_000))
    claims, evictions = tc.stitch_teacher(1, dbt)
    assert 11 in {c.fragment.raw_id for c in claims}
    assert all(f.raw_id != 11 for f, _lo, _hi in evictions)


def test_hole_fill_contributes_only_unclaimed_frames():
    # A hole-filling fragment that overlaps frames the walk already claimed may
    # contribute ONLY the free ones -- she cannot be two boxes in one frame.
    # (The walk's own CONTINUE handoff shares its boundary frame between two
    # fragments; that is pre-existing and separate from the hole fill.)
    dbt = {1: []}
    for ts in range(0, 20_001, 500):
        dbt[1].append(_d(ts, 10, 1, 0.2 + 0.3 * ts / 20_000))
    for ts in range(28_000, 50_001, 500):
        dbt[1].append(_d(ts, 12, 1, 0.62 + 0.2 * (ts - 28_000) / 22_000))
    # sits in the hole but also overlaps the tail fragment's first seconds
    for ts in range(21_000, 33_001, 500):
        dbt[1].append(_d(ts, 11, 1, 0.52 + 0.1 * (ts - 21_000) / 12_000))
    claims, _ = tc.stitch_teacher(1, dbt)
    fill = next((c for c in claims if c.fragment.raw_id == 11), None)
    assert fill is not None, "the hole fragment should be reclaimed"
    others = {d.video_ts_ms for c in claims if c.fragment.raw_id != 11 for d in c.dets}
    assert not ({d.video_ts_ms for d in fill.dets} & others)


def test_gap_fill_only_offers_detections_inside_the_window():
    # A pupil track spans the whole video; only its slice inside the freed
    # window may be offered, so acting on it cannot disturb the timeline.
    dbt = {1: [], 2: []}
    dbt[2] = [_d(ts, 20, 2, 0.35) for ts in range(0, 60_000, 200)]
    cands = tc.gap_fill_candidates(dbt, teacher_no=1, start_ms=20_000, end_ms=30_000, anchor=(0.35, 0.45))
    assert cands, "the pupil track overlaps the window"
    assert all(20_000 <= d.video_ts_ms <= 30_000 for d in cands[0])


def test_gap_fill_offers_the_mobile_candidate_for_judging():
    # A static pupil sitting exactly where the teacher was is a plausible decoy,
    # so ranking alone cannot settle it — what matters is that the mobile figure
    # (the walking adult) is OFFERED, so the VLM gets to judge her.
    dbt = {1: [], 2: [], 3: []}
    dbt[2] = [_d(ts, 20, 2, 0.30) for ts in range(20_000, 30_000, 200)]  # static at anchor
    dbt[3] = [
        _d(ts, 30, 3, 0.32 + 0.30 * (ts - 20_000) / 10_000) for ts in range(20_000, 30_000, 200)
    ]  # walks 0.32 -> 0.62
    cands = tc.gap_fill_candidates(dbt, teacher_no=1, start_ms=20_000, end_ms=30_000, anchor=(0.30, 0.45))
    assert {c[0].raw_track_id for c in cands} == {20, 30}


def test_gap_fill_ranks_a_far_candidate_below_a_near_one():
    # Distance to her last known position is the primary ranking signal.
    dbt = {1: [], 2: [], 3: []}
    dbt[2] = [_d(ts, 20, 2, 0.34) for ts in range(20_000, 30_000, 200)]  # near anchor
    dbt[3] = [_d(ts, 30, 3, 0.90) for ts in range(20_000, 30_000, 200)]  # across the room
    cands = tc.gap_fill_candidates(dbt, teacher_no=1, start_ms=20_000, end_ms=30_000, anchor=(0.35, 0.45))
    assert cands[0][0].raw_track_id == 20


def test_gap_fill_excludes_blocked_detections():
    # The just-trimmed detections must never be offered back.
    dbt = {1: [], 2: []}
    dbt[2] = [_d(ts, 20, 2, 0.35) for ts in range(20_000, 30_000, 200)]
    blocked = {(d.raw_track_id, d.video_ts_ms) for d in dbt[2]}
    assert tc.gap_fill_candidates(dbt, 1, 20_000, 30_000, (0.35, 0.45), blocked) == []


def test_gap_fill_ignores_a_flicker():
    dbt = {1: [], 2: []}
    dbt[2] = [_d(ts, 20, 2, 0.35) for ts in range(20_000, 20_600, 200)]  # 3 dets
    assert tc.gap_fill_candidates(dbt, 1, 20_000, 30_000, (0.35, 0.45)) == []


def test_claim_to_idx_trims_the_tail():
    frag = tc.Fragment(raw_id=10, host_track_no=1, dets=[_d(ts, 10, 1, 0.3) for ts in range(0, 5_000, 500)])
    assert len(tc.Claim(frag, 0).dets) == 10  # unbounded claim is unchanged
    assert len(tc.Claim(frag, 0, 4).dets) == 4
    assert len(tc.Claim(frag, 6, None).dets) == 4


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


# --------------------------------------------------------------------------- #
# Reclaiming an eviction
# --------------------------------------------------------------------------- #
#
# Every teacher-identity range the chain does not claim is evicted to a fresh
# student track, and no later stage can undo it -- the repair stages skip
# detections already carrying her number, which these still do when they run.
# So a wrongly-evicted range is where her "out of the room" time comes from.


def _vec(*seed):
    """A unit vector; nearby seeds give a high cosine, distant ones a low one."""
    import numpy as np

    v = np.zeros(8, dtype=np.float64)
    for i, s in enumerate(seed):
        v[i] = s
    return v / float(np.linalg.norm(v))


def _teacher_claims(raw_ids):
    """Whole-fragment claims, the only kind the appearance reference is built from."""
    out = []
    for raw in raw_ids:
        dets = [_d(ts, raw, 1, 0.2 + 0.5 * ts / 20_000) for ts in range(0, 20_001, 500)]
        out.append(tc.Claim(tc.Fragment(raw, 1, dets), 0, len(dets)))
    return out


def _model():
    return tc.fit_height_model(
        [_d(ts, 10, 1, 0.3) for ts in range(0, 20_001, 500)]
    )


def test_reclaims_her_own_body_from_eviction():
    # She walks through the hole, teacher-sized, looking like herself, at
    # timestamps nobody else holds. Refusing this eviction is what keeps her
    # from being renamed "Student N" while plainly on camera.
    claims = _teacher_claims([10, 11, 12])
    embeds = {10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06), 20: _vec(1, 0.05)}
    hers = [_d(ts, 20, 1, 0.2 + 0.6 * (ts - 30_000) / 20_000) for ts in range(30_000, 50_001, 500)]
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, hers), 0, len(hers))], claims, embeds, _model(), set()
    )
    assert {ts for _raw, ts in keep} == {d.video_ts_ms for d in hers}


def test_refuses_a_stranger_who_walks_the_same_hole():
    # Same geometry, different person. Appearance is the only thing separating
    # them, and it must be enough -- this is the chimera the user sees at once.
    claims = _teacher_claims([10, 11, 12])
    embeds = {10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06), 20: _vec(0.2, 1)}
    other = [_d(ts, 20, 1, 0.2 + 0.6 * (ts - 30_000) / 20_000) for ts in range(30_000, 50_001, 500)]
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, other), 0, len(other))], claims, embeds, _model(), set()
    )
    assert keep == set()


def test_only_the_best_match_wins_contested_time():
    # Two adult-sized fragments cross the same window. At most one is her, so
    # the weaker match may not ride in behind the stronger one.
    claims = _teacher_claims([10, 11, 12])
    embeds = {
        10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06),
        20: _vec(1, 0.05),   # her
        21: _vec(1, 0.30),   # close enough to pass the floor alone
    }
    span = range(30_000, 50_001, 500)
    hers = [_d(ts, 20, 1, 0.2 + 0.6 * (ts - 30_000) / 20_000) for ts in span]
    twin = [_d(ts, 21, 1, 0.8 - 0.5 * (ts - 30_000) / 20_000) for ts in span]
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, hers), 0, len(hers)), (tc.Fragment(21, 1, twin), 0, len(twin))],
        claims, embeds, _model(), set(),
    )
    assert {raw for raw, _ts in keep} == {20}


def test_refuses_a_tail_the_stored_embed_cannot_speak_for():
    # A fragment's CLIP vector is sampled from its opening seconds, so on a tail
    # eviction it describes the part that was CLAIMED, not the part in question.
    # With no valid appearance evidence the reclaim fails closed.
    claims = _teacher_claims([10, 11, 12])
    embeds = {10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06), 20: _vec(1, 0.05)}
    frag = [_d(ts, 20, 1, 0.2 + 0.6 * ts / 60_000) for ts in range(0, 60_001, 500)]
    tail = len(frag) - 20
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, frag), tail, len(frag))], claims, embeds, _model(), set()
    )
    assert keep == set()


def test_subtracts_time_she_already_holds_instead_of_refusing_the_range():
    # An overlap costs the overlap, not the whole range: a 20s guess from the
    # anchor must not void a longer appearance-verified reclaim.
    claims = _teacher_claims([10, 11, 12])
    embeds = {10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06), 20: _vec(1, 0.05)}
    hers = [_d(ts, 20, 1, 0.2 + 0.6 * (ts - 30_000) / 20_000) for ts in range(30_000, 50_001, 500)]
    held = {d.video_ts_ms for d in hers[:8]}
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, hers), 0, len(hers))], claims, embeds, _model(), held
    )
    assert {ts for _raw, ts in keep} == {d.video_ts_ms for d in hers[8:]}


def test_reclaims_nothing_without_enough_reference_fragments():
    # Too few whole-fragment claims to learn what she looks like -> no reference,
    # so the gate stays shut rather than guessing.
    claims = _teacher_claims([10])
    embeds = {10: _vec(1, 0.05), 20: _vec(1, 0.05)}
    hers = [_d(ts, 20, 1, 0.2 + 0.6 * (ts - 30_000) / 20_000) for ts in range(30_000, 50_001, 500)]
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, hers), 0, len(hers))], claims, embeds, _model(), set()
    )
    assert keep == set()


def test_refuses_a_seated_student_sized_range():
    # A short static box is a child at a desk however well it scores.
    claims = _teacher_claims([10, 11, 12])
    embeds = {10: _vec(1, 0.05), 11: _vec(1, 0.04), 12: _vec(1, 0.06), 20: _vec(1, 0.05)}
    kid = [_d(ts, 20, 1, 0.9, h=0.18, standing=False) for ts in range(30_000, 50_001, 500)]
    keep = tc.reclaimable_evictions(
        [(tc.Fragment(20, 1, kid), 0, len(kid))], claims, embeds, _model(), set()
    )
    assert keep == set()
