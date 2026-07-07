"""Unit tests for identity merge scoring and threshold behavior."""

import math

import numpy as np

from app.merge import (
    MERGE_THRESHOLD,
    RawTrack,
    build_raw_tracks,
    hist_correlation,
    merge_tracks,
    pair_score,
)
from app.models import Detection


def _hist(bin_idx: int, size: int = 64) -> np.ndarray:
    h = np.zeros(size, dtype=np.float32)
    h[bin_idx] = 1.0
    return h


def _embed(dim: int, size: int = 512) -> np.ndarray:
    e = np.zeros(size, dtype=np.float64)
    e[dim] = 1.0
    return e


def _rt(
    raw_id,
    first,
    last,
    hist=None,
    area=0.02,
    n=10,
    embed=None,
    height=0.0,
    centers=None,
) -> RawTrack:
    first_center, last_center = centers if centers else (None, None)
    return RawTrack(
        raw_id=raw_id,
        first_ms=first,
        last_ms=last,
        hist=hist,
        mean_area=area,
        n_dets=n,
        first_center=first_center,
        last_center=last_center,
        embed=embed,
        mean_height=height,
    )


def test_pair_score_merges_same_person_fragments():
    a = _rt(1, 0, 10_000, _hist(3))
    b = _rt(2, 10_500, 20_000, _hist(3))
    score = pair_score(a, b)
    assert score is not None
    assert score > MERGE_THRESHOLD
    # Identical hist, same size, tiny gap; spatial is neutral 0.5 without
    # endpoint centers, so the ceiling is 0.35 + 0.125 + 0.2 + ~0.2.
    assert score > 0.85


def test_pair_score_rejects_large_temporal_overlap():
    a = _rt(1, 0, 10_000, _hist(3))
    b = _rt(2, 4_000, 20_000, _hist(3))  # 6s overlap -> co-present people
    assert pair_score(a, b) is None


def test_pair_score_allows_tiny_overlap():
    a = _rt(1, 0, 10_000, _hist(3))
    b = _rt(2, 9_500, 20_000, _hist(3))  # 500ms overlap < 1s tolerance
    assert pair_score(a, b) is not None


def test_pair_score_rejects_gap_over_ten_minutes():
    a = _rt(1, 0, 10_000, _hist(3))
    b = _rt(2, 10_000 + 600_001, 700_000 + 600_001, _hist(3))
    assert pair_score(a, b) is None


def test_different_histograms_stay_below_threshold():
    a = _rt(1, 0, 10_000, _hist(3))
    b = _rt(2, 11_000, 20_000, _hist(40))
    score = pair_score(a, b)
    assert score is not None
    assert score < MERGE_THRESHOLD
    mapping, identities = merge_tracks([a, b])
    assert mapping == {1: 1, 2: 2}
    assert len(identities) == 2


def test_hist_correlation_bounds_and_missing():
    assert hist_correlation(_hist(3), _hist(3)) == 1.0
    assert hist_correlation(_hist(3), _hist(40)) == 0.0  # negative corr clamped
    assert hist_correlation(None, _hist(3)) == 0.5  # neutral when missing


def test_chain_merge_into_single_identity():
    tracks = [
        _rt(1, 0, 10_000, _hist(3)),
        _rt(2, 11_000, 20_000, _hist(3)),
        _rt(3, 21_000, 30_000, _hist(3)),
    ]
    mapping, identities = merge_tracks(tracks)
    assert mapping == {1: 1, 2: 1, 3: 1}
    assert len(identities) == 1
    assert identities[0]["raw_track_ids"] == [1, 2, 3]
    assert identities[0]["first_ms"] == 0
    assert identities[0]["last_ms"] == 30_000


def test_track_no_ordered_by_first_appearance():
    # raw id 9 appears first, so it must become track_no 1
    tracks = [
        _rt(5, 5_000, 9_000, _hist(40)),
        _rt(9, 0, 4_000, _hist(3)),
    ]
    mapping, identities = merge_tracks(tracks)
    assert mapping[9] == 1
    assert mapping[5] == 2
    assert [i["track_no"] for i in identities] == [1, 2]


def test_multi_interval_cluster_still_rejects_overlap():
    # a1+a2 merge first (same hist, tiny gap); the resulting multi-interval
    # cluster overlaps b by 8s > 1s tolerance, so b must stay separate.
    a1 = _rt(1, 0, 10_000, _hist(3))
    a2 = _rt(2, 20_000, 30_000, _hist(3))
    b = _rt(3, 21_000, 29_000, _hist(3))
    mapping, identities = merge_tracks([a1, a2, b])
    assert mapping[1] == mapping[2]
    assert mapping[3] != mapping[1]
    assert len(identities) == 2


def test_merge_scales_to_many_fragments():
    # Perf regression: the old implementation re-scored all O(n^2) pairs
    # after every merge (~25s for n=250); the heap-based merge finishes in
    # well under a second. Also checks correctness at scale: a chain of
    # same-appearance fragments collapses into a single identity.
    import time

    tracks = [
        _rt(i + 1, i * 4_000, i * 4_000 + 3_000, _hist(3)) for i in range(250)
    ]
    t0 = time.perf_counter()
    mapping, identities = merge_tracks(tracks)
    elapsed = time.perf_counter() - t0
    assert len(identities) == 1
    assert set(mapping.values()) == {1}
    assert identities[0]["first_ms"] == 0
    assert identities[0]["last_ms"] == 249 * 4_000 + 3_000
    assert elapsed < 10.0, f"merge_tracks took {elapsed:.1f}s for 250 fragments"


def test_embed_cosine_veto_rejects_orthogonal_embeds():
    # Same hist + tiny gap merges without embeds (see
    # test_pair_score_merges_same_person_fragments); orthogonal CLIP embeds
    # (cos 0 < 0.35) must veto the pair outright.
    a = _rt(1, 0, 10_000, _hist(3), embed=_embed(0))
    b = _rt(2, 10_500, 20_000, _hist(3), embed=_embed(1))
    assert pair_score(a, b) is None


def test_identical_embeds_boost_appearance():
    # Uncorrelated hists alone score appearance 0; identical embeds lift the
    # blended appearance term (0.5 * cos_mapped + 0.5 * hist_corr), so the
    # pair score must strictly increase.
    without = pair_score(
        _rt(1, 0, 10_000, _hist(3)), _rt(2, 10_500, 20_000, _hist(40))
    )
    with_embeds = pair_score(
        _rt(1, 0, 10_000, _hist(3), embed=_embed(0)),
        _rt(2, 10_500, 20_000, _hist(40), embed=_embed(0)),
    )
    assert without is not None and with_embeds is not None
    assert with_embeds > without


# Mobile trajectories (endpoint spread 0.3 > MOBILE_RANGE) for the adult
# prior, which only applies to walking clusters. The pair's adjacent
# endpoints sit 0.6 apart, far beyond a plausible off-camera hop, so only
# the prior's 0.75 appearance floor can carry the pair over the threshold.
_WALK_OUT = ((0.05, 0.5), (0.35, 0.5))
_WALK_IN = ((0.95, 0.5), (0.65, 0.5))


def test_adult_pair_prior_merges_tall_mobile_fragments_only():
    # Two tall MOBILE fragments (>= p90 mean height, walking) with no
    # appearance evidence at all reunite via the adult-size prior; the
    # same-shaped short (child-height) pair stays split because children
    # never reach p90.
    tall_a = _rt(1, 0, 10_000, height=0.5, centers=_WALK_OUT)
    tall_b = _rt(2, 25_000, 35_000, height=0.5, centers=_WALK_IN)
    short_a = _rt(3, 0, 10_000, height=0.2, centers=_WALK_OUT)
    short_b = _rt(4, 25_000, 35_000, height=0.2, centers=_WALK_IN)
    mapping, identities = merge_tracks([tall_a, tall_b, short_a, short_b])
    assert mapping[1] == mapping[2]
    assert mapping[3] != mapping[4]
    assert mapping[3] != mapping[1] and mapping[4] != mapping[1]
    assert len(identities) == 3


def test_adult_pair_prior_requires_embed_agreement_over_hist_contradiction():
    # With contradicting hists the prior needs embed confirmation: cos 0.6
    # (>= 0.5) unlocks the floor and merges the tall walkers; cos 0.4
    # (surviving the veto but below agreement) leaves the contradiction in
    # charge and the pair split.
    e_base = np.zeros(512)
    e_base[0] = 1.0
    for cos_target, should_merge in ((0.6, True), (0.4, False)):
        e_other = np.zeros(512)
        e_other[0] = cos_target
        e_other[1] = math.sqrt(1.0 - cos_target**2)
        tall_a = _rt(
            1, 0, 10_000, _hist(3), embed=e_base, height=0.5, centers=_WALK_OUT
        )
        tall_b = _rt(
            2, 25_000, 35_000, _hist(40), embed=e_other, height=0.5, centers=_WALK_IN
        )
        short = _rt(3, 0, 35_000, _hist(10), height=0.2)
        mapping, _ = merge_tracks([tall_a, tall_b, short])
        assert (mapping[1] == mapping[2]) is should_merge, f"cos={cos_target}"


def test_adult_pair_prior_skips_stationary_tall_clusters():
    # Perspective-tall front-row students reach p90 while seated; without
    # mobility on BOTH sides the prior must stay off even with zero
    # appearance evidence, so the cross-room distance keeps them apart
    # (regression: the prior used to glue tall stationary clusters and
    # destroy the teacher's role margin on real footage).
    seat_a = ((0.18, 0.8), (0.22, 0.8))  # spread 0.04: not mobile, not seat-vetoed
    seat_b = ((0.82, 0.8), (0.78, 0.8))
    tall_a = _rt(1, 0, 10_000, height=0.5, centers=seat_a)
    tall_b = _rt(2, 25_000, 35_000, height=0.5, centers=seat_b)
    short = _rt(3, 0, 35_000, height=0.2)
    mapping, identities = merge_tracks([tall_a, tall_b, short])
    assert mapping[1] != mapping[2]
    assert len(identities) == 3


def test_adult_pair_prior_never_overrides_embed_veto():
    # Adult stature must not glue two different tall walkers together when
    # their embeds actively disagree (cos < 0.35 vetoes before the prior).
    tall_a = _rt(1, 0, 10_000, _hist(3), embed=_embed(0), height=0.5, centers=_WALK_OUT)
    tall_b = _rt(
        2, 25_000, 35_000, _hist(40), embed=_embed(1), height=0.5, centers=_WALK_IN
    )
    short = _rt(3, 0, 35_000, _hist(10), height=0.2)
    mapping, identities = merge_tracks([tall_a, tall_b, short])
    assert mapping[1] != mapping[2]
    assert len(identities) == 3


def _crouch_pair(c_first, c_last, c_centers, c_area):
    """Adult walker A plus a below-p90 fragment C with agreeing embeds
    (cos ~0.52 >= ADULT_PRIOR_MIN_COS) and contradicting hists, plus two
    tall-span seated fillers that pin the height p90 above C."""
    e_base = np.zeros(512)
    e_base[0] = 1.0
    e_other = np.zeros(512)
    e_other[0] = 0.52
    e_other[1] = math.sqrt(1.0 - 0.52**2)
    adult = _rt(
        1, 0, 20_000, _hist(3), embed=e_base, height=0.5, centers=_WALK_OUT
    )
    crouch = _rt(
        2, c_first, c_last, _hist(40), embed=e_other, height=0.2,
        area=c_area, centers=c_centers,
    )
    span_end = max(20_000, c_last)
    fillers = [
        _rt(3, 0, span_end, height=0.2, centers=((0.18, 0.8), (0.22, 0.8))),
        _rt(4, 0, span_end, height=0.2, centers=((0.82, 0.8), (0.78, 0.8))),
    ]
    return [adult, crouch, *fillers]


def test_crouch_reacquisition_joins_adult_prior():
    # The crouching-teacher fragment: kneeling at a desk drops her bbox
    # below the p90 height cut, so the both-adult prior misses her exactly
    # when the tracker re-acquires. A near-instant same-spot re-acquisition
    # (400ms, endpoint dist 0) with agreeing embeds must still get the
    # appearance floor despite the contradicting hist and dissimilar box
    # size; without the floor the pair scores below MERGE_THRESHOLD.
    tracks = _crouch_pair(
        20_400, 40_000, c_centers=((0.35, 0.5), (0.55, 0.5)), c_area=0.004
    )
    mapping, identities = merge_tracks(tracks)
    assert mapping[1] == mapping[2], "crouch re-acquisition must reunite"
    assert len(identities) == 3


def test_crouch_reacquisition_requires_small_gap():
    # Same fragment pair but 25s apart: a genuinely absent teacher stands
    # to walk out, which restores her height, so a long-gap one-adult pair
    # must NOT ride the prior (equal box sizes would merge if it fired).
    tracks = _crouch_pair(
        45_000, 65_000, c_centers=((0.95, 0.5), (0.75, 0.5)), c_area=0.02
    )
    mapping, identities = merge_tracks(tracks)
    assert mapping[1] != mapping[2]
    assert len(identities) == 4


def test_crouch_reacquisition_requires_same_spot():
    # 400ms gap but the fragment reappears 0.6 across the room: a teleport
    # is not a re-acquisition, so the prior stays off and the pair splits.
    tracks = _crouch_pair(
        20_400, 40_000, c_centers=((0.95, 0.5), (0.75, 0.5)), c_area=0.02
    )
    mapping, identities = merge_tracks(tracks)
    assert mapping[1] != mapping[2]
    assert len(identities) == 4


def test_leave_and_return_teacher_fragments_merge():
    # The M5 target scenario: the tall teacher walks out of frame and
    # returns 15s later on the far side of the room. The two fragments are
    # temporally disjoint, spatially discontinuous (endpoint dist ~0.5) and
    # both mobile, but share a CLIP embedding — they must fold into one
    # identity while the seated students stay separate.
    bbox_t = {"y": 0.2, "w": 0.1, "h": 0.5}
    dets = []
    for i, ts in enumerate(range(0, 20_001, 1_000)):
        dets.append(
            Detection(ts, 1, {"x": 0.05 + 0.015 * i, **bbox_t}, 0.9, True, False)
        )
    for i, ts in enumerate(range(35_000, 55_001, 1_000)):
        dets.append(
            Detection(ts, 2, {"x": 0.85 - 0.015 * i, **bbox_t}, 0.9, True, False)
        )
    for ts in range(0, 55_001, 1_000):
        dets.append(
            Detection(ts, 3, {"x": 0.3, "y": 0.6, "w": 0.08, "h": 0.2}, 0.9, False, False)
        )
        dets.append(
            Detection(ts, 4, {"x": 0.6, "y": 0.6, "w": 0.08, "h": 0.2}, 0.9, False, False)
        )
    embeds = {1: [_embed(0)], 2: [_embed(0)], 3: [_embed(5)], 4: [_embed(6)]}
    tracks = build_raw_tracks(dets, {}, embeds)
    mapping, identities = merge_tracks(tracks)
    assert mapping[1] == mapping[2], "teacher fragments must reunite after leave/return"
    assert len({mapping[1], mapping[3], mapping[4]}) == 3
    assert len(identities) == 3


def test_build_raw_tracks_summarizes_detections():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.4}
    dets = [
        Detection(0, 7, bbox, 0.9, True, False),
        Detection(500, 7, bbox, 0.8, True, False),
        Detection(2_000, 8, bbox, 0.7, False, False),
    ]
    hists = {7: [_hist(3), _hist(3)]}
    tracks = build_raw_tracks(dets, hists)
    assert [t.raw_id for t in tracks] == [7, 8]
    t7 = tracks[0]
    assert (t7.first_ms, t7.last_ms, t7.n_dets) == (0, 500, 2)
    assert t7.hist is not None and t7.hist.shape == (64,)
    assert tracks[1].hist is None
    assert abs(t7.mean_area - 0.04) < 1e-9
