"""Identity merge: greedy agglomerative merging of raw tracker ids.

Raw BoT-SORT track ids fragment whenever a person is occluded or leaves the
frame. We merge fragments into stable identities:

- candidate pairs only when temporal overlap is tiny (< 1s) and the gap
  between them is < 10 min,
- score = 0.6 * appearance + 0.2 * size_similarity + 0.2 * temporal_proximity,
- appearance = torso-histogram correlation when BOTH fragments carry a
  histogram; otherwise SPATIAL CONTINUITY between the temporally-adjacent
  endpoints: exp(-(endpoint_dist / (0.04 + 0.004 * gap_s))^2). A missing
  histogram must NOT score a neutral 0.5 — on real classroom footage that
  floor (0.3 + 0.2*size + 0.2*temporal ~ 0.68) chains any two non-overlapping
  similar-size fragments with gap <= ~7.5 min into full-frame chimeras.
  Seated students reconnect at endpoint distances ~0.02 (84% < 0.10), so the
  gaussian gate keeps true reconnects and rejects cross-room jumps.
- greedy: repeatedly merge the highest-scoring candidate pair >= threshold (0.55),
- final identities get track_no 1..N ordered by first appearance.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.models import Detection

OVERLAP_TOLERANCE_MS = 1_000
MAX_GAP_MS = 600_000  # 10 minutes
MERGE_THRESHOLD = 0.55

W_HIST = 0.6  # appearance weight (histogram OR spatial continuity)
W_SIZE = 0.2
W_TEMPORAL = 0.2

# Spatial-continuity tolerance (normalized units): base allowance plus growth
# per second of gap. Calibrated on real classroom fragments: a stationary
# reconnect sits at ~0.02, but a fast-moving teacher jumps ~0.08-0.10 across
# even a ~1s sampling gap (she covers most of the frame width). A 0.04 base
# fragmented her walk into orphan identities that no longer reconnect; 0.08
# reunites the walking teacher into one identity while chimeric cross-room
# jumps (> ~0.16) still fall off the gaussian. Endpoint distances of genuine
# seated reconnects (~0.02) remain ~1.0 either way.
SPATIAL_BASE_TOL = 0.08
SPATIAL_TOL_PER_S = 0.004

Center = Optional[tuple[float, float]]
# Cluster interval: (start_ms, end_ms, start_center, end_center)
Interval = tuple[int, int, Center, Center]


@dataclass
class RawTrack:
    """Summary of one raw tracker id."""

    raw_id: int
    first_ms: int
    last_ms: int
    hist: Optional[np.ndarray]  # median torso HSV histogram (flattened) or None
    mean_area: float  # mean normalized bbox area (w*h)
    n_dets: int
    first_center: Center = None  # bbox center at first_ms (spatial continuity)
    last_center: Center = None  # bbox center at last_ms


def _bbox_center(d: Detection) -> tuple[float, float]:
    return (d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0)


def build_raw_tracks(
    detections: list[Detection], hists: dict[int, list[np.ndarray]]
) -> list[RawTrack]:
    """Group per-frame detections by raw_track_id into RawTrack summaries."""
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.raw_track_id, []).append(d)

    tracks: list[RawTrack] = []
    for raw_id, dets in by_id.items():
        areas = [max(0.0, d.bbox["w"] * d.bbox["h"]) for d in dets]
        first_det = min(dets, key=lambda d: d.video_ts_ms)
        last_det = max(dets, key=lambda d: d.video_ts_ms)
        samples = hists.get(raw_id) or []
        hist = None
        if samples:
            hist = np.median(np.stack([np.asarray(s).ravel() for s in samples]), axis=0)
        tracks.append(
            RawTrack(
                raw_id=raw_id,
                first_ms=first_det.video_ts_ms,
                last_ms=last_det.video_ts_ms,
                hist=hist,
                mean_area=float(np.mean(areas)) if areas else 0.0,
                n_dets=len(dets),
                first_center=_bbox_center(first_det),
                last_center=_bbox_center(last_det),
            )
        )
    tracks.sort(key=lambda t: t.first_ms)
    return tracks


def hist_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Pearson correlation of two histograms clamped to [0, 1].

    Neutral 0.5 when either histogram is missing (no appearance evidence).
    NOTE: _score_clusters only falls back to this neutral value when spatial
    endpoints are ALSO unavailable — see spatial_continuity.
    """
    if a is None or b is None:
        return 0.5
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, min(1.0, corr))


def spatial_continuity(ca: Center, cb: Center, gap_ms: int) -> float:
    """exp(-(dist/tol)^2) between the adjacent fragment endpoints.

    tol = SPATIAL_BASE_TOL + SPATIAL_TOL_PER_S * gap_seconds: a longer absence
    allows more drift. Neutral 0.5 when endpoint centers are unavailable
    (legacy callers constructing RawTrack without centers).
    """
    if ca is None or cb is None:
        return 0.5
    dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    tol = SPATIAL_BASE_TOL + SPATIAL_TOL_PER_S * (max(0, gap_ms) / 1000.0)
    return math.exp(-((dist / tol) ** 2))


@dataclass
class _Cluster:
    raw_ids: list[int]
    intervals: list[Interval]
    hist: Optional[np.ndarray]
    hist_weight: float
    mean_area: float
    n_dets: int
    first_ms: int = field(init=False)

    def __post_init__(self) -> None:
        self.first_ms = min(iv[0] for iv in self.intervals)


def _cluster_from_track(t: RawTrack) -> _Cluster:
    return _Cluster(
        raw_ids=[t.raw_id],
        intervals=[(t.first_ms, t.last_ms, t.first_center, t.last_center)],
        hist=None if t.hist is None else np.asarray(t.hist, dtype=np.float64).ravel(),
        hist_weight=float(t.n_dets if t.hist is not None else 0),
        mean_area=t.mean_area,
        n_dets=t.n_dets,
    )


def _coalesce(intervals: list[Interval]) -> list[Interval]:
    """Fuse overlapping/adjacent SORTED intervals into a disjoint sorted list.

    Cluster interval lists are kept coalesced so _overlap_ms/_gap_info can use
    linear two-pointer sweeps instead of O(k_a * k_b) pair loops. The fused
    interval keeps the earliest start_center and the latest end_center.
    """
    out: list[Interval] = []
    for s, e, sc, ec in intervals:
        if out and s <= out[-1][1]:
            ps, pe, psc, pec = out[-1]
            if e > pe:
                out[-1] = (ps, e, psc, ec)
        else:
            out.append((s, e, sc, ec))
    return out


def _overlap_ms(a: _Cluster, b: _Cluster) -> int:
    """Total overlap between two sorted, disjoint interval lists (two-pointer)."""
    ia, ib = a.intervals, b.intervals
    total = 0
    i = j = 0
    while i < len(ia) and j < len(ib):
        s1, e1 = ia[i][0], ia[i][1]
        s2, e2 = ib[j][0], ib[j][1]
        total += max(0, min(e1, e2) - max(s1, s2))
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return total


def _gap_info(a: _Cluster, b: _Cluster) -> tuple[int, Center, Center]:
    """(smallest gap, adjacent endpoint centers) between two interval lists.

    Two-pointer sweep over sorted, disjoint interval lists: advancing the
    interval that ends first is safe because its gap to every later interval
    of the other list can only be larger than the one just measured. The
    returned centers are (end center of the earlier interval, start center of
    the later interval) at the closest approach — the pair of points a real
    person would have to travel between during the gap.
    """
    ia, ib = a.intervals, b.intervals
    best: Optional[int] = None
    best_centers: tuple[Center, Center] = (None, None)
    i = j = 0
    while i < len(ia) and j < len(ib):
        s1, e1, sc1, ec1 = ia[i]
        s2, e2, sc2, ec2 = ib[j]
        if min(e1, e2) >= max(s1, s2):  # touching/overlapping
            return (0, ec1, sc2) if s1 <= s2 else (0, ec2, sc1)
        gap = max(s1, s2) - min(e1, e2)
        if best is None or gap < best:
            best = gap
            best_centers = (ec1, sc2) if e1 <= s2 else (ec2, sc1)
        if e1 <= e2:
            i += 1
        else:
            j += 1
    if best is None:
        return MAX_GAP_MS, None, None
    return best, best_centers[0], best_centers[1]


def _size_similarity(a: float, b: float) -> float:
    hi = max(a, b)
    if hi <= 0.0:
        return 1.0
    return max(0.0, min(1.0, min(a, b) / hi))


def _score_clusters(
    a: _Cluster,
    b: _Cluster,
    overlap_tolerance_ms: int = OVERLAP_TOLERANCE_MS,
    max_gap_ms: int = MAX_GAP_MS,
) -> Optional[float]:
    """Merge score, or None when (a, b) is not a merge candidate."""
    if _overlap_ms(a, b) >= overlap_tolerance_ms:
        return None
    gap, ca, cb = _gap_info(a, b)
    if gap >= max_gap_ms:
        return None
    temporal = 1.0 - gap / max_gap_ms
    if a.hist is not None and b.hist is not None:
        appearance = hist_correlation(a.hist, b.hist)
    else:
        # No appearance evidence (e.g. /rederive from stored detections —
        # histograms are never persisted): demand spatial continuity instead.
        appearance = spatial_continuity(ca, cb, gap)
    return (
        W_HIST * appearance
        + W_SIZE * _size_similarity(a.mean_area, b.mean_area)
        + W_TEMPORAL * temporal
    )


def pair_score(a: RawTrack, b: RawTrack) -> Optional[float]:
    """Merge score for two raw tracks (None when not a candidate pair)."""
    return _score_clusters(_cluster_from_track(a), _cluster_from_track(b))


def _merge_clusters(a: _Cluster, b: _Cluster) -> _Cluster:
    if a.hist is None and b.hist is None:
        hist, weight = None, 0.0
    elif a.hist is None:
        hist, weight = b.hist, b.hist_weight
    elif b.hist is None:
        hist, weight = a.hist, a.hist_weight
    else:
        wa, wb = max(a.hist_weight, 1.0), max(b.hist_weight, 1.0)
        hist = (a.hist * wa + b.hist * wb) / (wa + wb)
        weight = wa + wb
    n = a.n_dets + b.n_dets
    area = (a.mean_area * a.n_dets + b.mean_area * b.n_dets) / max(n, 1)
    return _Cluster(
        raw_ids=sorted(a.raw_ids + b.raw_ids),
        intervals=_coalesce(
            sorted(a.intervals + b.intervals, key=lambda iv: (iv[0], iv[1]))
        ),
        hist=hist,
        hist_weight=weight,
        mean_area=area,
        n_dets=n,
    )


def merge_tracks(
    raw_tracks: list[RawTrack],
    threshold: float = MERGE_THRESHOLD,
    overlap_tolerance_ms: int = OVERLAP_TOLERANCE_MS,
    max_gap_ms: int = MAX_GAP_MS,
) -> tuple[dict[int, int], list[dict]]:
    """Greedy agglomerative merge.

    Returns (mapping raw_track_id -> track_no,
             identities [{track_no, raw_track_ids, first_ms, last_ms}]).
    track_no is 1..N ordered by first appearance.

    Implemented with a lazy-deletion max-heap instead of a full O(n^2) rescan
    per merge: _score_clusters(a, b) depends only on a and b, so a merge only
    invalidates pairs involving the two merged clusters (their ids simply
    disappear from `alive`) and only pairs involving the new cluster need
    scoring. Total cost drops from ~O(merges * n^2 * k) to ~O(n^2 log n),
    which keeps heavily fragmented long videos out of multi-hour merge stalls.
    """
    alive: dict[int, _Cluster] = {
        cid: _cluster_from_track(t) for cid, t in enumerate(raw_tracks)
    }
    next_id = len(raw_tracks)
    heap: list[tuple[float, int, int]] = []  # (-score, cid_a, cid_b)

    def _push_pairs(cid: int, other_ids: list[int]) -> None:
        a = alive[cid]
        for oid in other_ids:
            score = _score_clusters(a, alive[oid], overlap_tolerance_ms, max_gap_ms)
            if score is not None and score >= threshold:
                heapq.heappush(heap, (-score, cid, oid))

    ids = list(alive)
    for k, cid in enumerate(ids):
        _push_pairs(cid, ids[k + 1 :])

    while heap:
        _, i, j = heapq.heappop(heap)
        if i not in alive or j not in alive:
            continue  # stale entry: one side was already merged away
        merged = _merge_clusters(alive.pop(i), alive.pop(j))
        mid = next_id
        next_id += 1
        others = list(alive)
        alive[mid] = merged
        _push_pairs(mid, others)

    clusters = sorted(alive.values(), key=lambda c: c.first_ms)
    mapping: dict[int, int] = {}
    identities: list[dict] = []
    for track_no, cluster in enumerate(clusters, start=1):
        for raw_id in cluster.raw_ids:
            mapping[raw_id] = track_no
        identities.append(
            {
                "track_no": track_no,
                "raw_track_ids": list(cluster.raw_ids),
                "first_ms": min(iv[0] for iv in cluster.intervals),
                "last_ms": max(iv[1] for iv in cluster.intervals),
            }
        )
    return mapping, identities
