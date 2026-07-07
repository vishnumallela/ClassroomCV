"""Unit tests for identity merge scoring and threshold behavior."""

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


def _rt(raw_id, first, last, hist=None, area=0.02, n=10) -> RawTrack:
    return RawTrack(
        raw_id=raw_id, first_ms=first, last_ms=last, hist=hist, mean_area=area, n_dets=n
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
