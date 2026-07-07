"""Identity merge: greedy agglomerative merging of raw tracker ids.

Raw BoT-SORT track ids fragment whenever a person is occluded or leaves the
frame. We merge fragments into stable identities:

- candidate pairs only when temporal overlap is tiny (< 1s) and the gap
  between them is < 10 min,
- CLIP embed veto (plan M5): when both fragments carry a CLIP embedding and
  their cosine is < 0.35 they are different people, full stop,
- appearance = 0.5 * cosine (mapped from [0.35, 1] to [0, 1]) + 0.5 *
  torso-histogram correlation when both embeds AND both histograms exist;
  hist-only pairs keep the plain correlation; otherwise SPATIAL CONTINUITY
  between the temporally-adjacent endpoints:
  exp(-(endpoint_dist / (0.04 + 0.004 * gap_s))^2). A missing
  histogram must NOT score a neutral 0.5 — on real classroom footage that
  floor (0.3 + 0.2*size + 0.2*temporal ~ 0.68) chains any two non-overlapping
  similar-size fragments with gap <= ~7.5 min into full-frame chimeras.
  Seated students reconnect at endpoint distances ~0.02 (84% < 0.10), so the
  gaussian gate keeps true reconnects and rejects cross-room jumps.
- adult-size prior: two clusters that are BOTH adult-tall (mean bbox height
  >= the 90th percentile) AND mobile get an appearance floor of 0.75 when
  their embeds agree (cos >= 0.5) or no appearance evidence exists — the
  walking teacher who leaves the frame and returns far away reconnects even
  though her spatial continuity is near zero, while seated students (even
  the perspective-tall front row) never qualify; a one-adult mobile pair
  also qualifies when the two fragments are a near-instant same-spot
  re-acquisition (gap <= 1.5s, endpoint dist <= 0.10) — the crouching
  teacher whose bbox falls below the height cut mid-fragment,
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

# Appearance alone cannot separate uniformed students (inter-student histogram
# correlation runs ~0.9 when shirts repeat), so spatial continuity is ALWAYS a
# scoring term rather than a fallback, and seated far-apart pairs are hard
# vetoed regardless of histogram agreement.
W_HIST = 0.35
W_SPATIAL = 0.25
W_SIZE = 0.2
W_TEMPORAL = 0.2

# A cluster whose interval-endpoint centers span more than this is mobile (the
# walking teacher); mobile pairs evaluate spatial continuity with a widened
# tolerance so her long jumps between fragments still reconnect. This must
# stay DISTANCE-BASED: an earlier flat score floor for mobile pairs let one
# bad merge make a cluster "mobile", which then absorbed every non-overlapping
# fragment in the room, and every absorption made it more mobile (a chimera
# cascade that destroyed teacher classification on real footage).
MOBILE_RANGE = 0.15
MOBILE_TOL_SCALE = 3.0
# Seated+seated pairs whose approximate anchor centers sit farther apart than
# this are different desks, therefore different students: reject outright.
SEATED_RANGE = 0.02
SEATED_VETO_DIST = 0.10

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

# CLIP embed agreement (embeddings are stored L2-normalized, so a dot product
# IS the cosine). Below 0.35 two upper-body crops are different people no
# matter what hist/size/temporal say — the seat-swap veto from plan M5.
EMBED_VETO_COS = 0.35

# Adult-size prior: clusters whose mean bbox height reaches the 90th
# percentile across tracks are adult-tall candidates, but height alone is NOT
# adult on CCTV — perspective makes stationary front-row students just as
# tall (measured on the demo: 16 of 151 tracks reach p90, most of them seated
# kids near the camera, and flooring their pairwise appearance broke the
# teacher's role margin). The prior therefore only fires when BOTH clusters
# are also mobile (walking), which is exactly the leave-and-return teacher
# case it exists for. Appearance gets a 0.75 floor, and only when the CLIP
# embeds agree (cos >= 0.5) or there is no appearance evidence at all — a
# computable hist contradiction without embed confirmation must stand (the
# demo's verified teacher_present_ms regressed when the prior overrode it).
ADULT_HEIGHT_PERCENTILE = 90.0
ADULT_PRIOR_MIN_COS = 0.5
ADULT_PRIOR_APPEARANCE = 0.75
# Crouch re-acquisition: a teacher kneeling at a desk shrinks below the p90
# height cut exactly when the tracker tends to drop her, so her return
# fragment fails the both-adult test through posture, not identity. A
# one-adult pair is admitted to the adult tier anyway when the fragments are
# a near-instant same-spot re-acquisition (the tracker would have kept the id
# but for a momentary blip); long-gap one-adult pairs stay excluded because
# a genuinely absent teacher stands to walk, which restores her height.
CROUCH_REACQUIRE_MAX_GAP_MS = 1_500
CROUCH_REACQUIRE_MAX_DIST = 0.10

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
    embed: Optional[np.ndarray] = None  # L2-normalized median CLIP embedding
    mean_height: float = 0.0  # mean normalized bbox height (adult-size prior)


def _bbox_center(d: Detection) -> tuple[float, float]:
    return (d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0)


def build_raw_tracks(
    detections: list[Detection],
    hists: dict[int, list[np.ndarray]],
    embeds: Optional[dict[int, list]] = None,
) -> list[RawTrack]:
    """Group per-frame detections by raw_track_id into RawTrack summaries.

    embeds maps raw_track_id to a list of CLIP embedding samples (usually a
    single median from the detector or the persisted copy); like hists, the
    per-track summary is the median over samples, re-normalized so downstream
    dot products stay true cosines.
    """
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.raw_track_id, []).append(d)

    tracks: list[RawTrack] = []
    for raw_id, dets in by_id.items():
        areas = [max(0.0, d.bbox["w"] * d.bbox["h"]) for d in dets]
        heights = [max(0.0, d.bbox["h"]) for d in dets]
        first_det = min(dets, key=lambda d: d.video_ts_ms)
        last_det = max(dets, key=lambda d: d.video_ts_ms)
        samples = hists.get(raw_id) or []
        hist = None
        if samples:
            hist = np.median(np.stack([np.asarray(s).ravel() for s in samples]), axis=0)
        embed_samples = (embeds or {}).get(raw_id) or []
        embed = None
        if embed_samples:
            embed = np.median(
                np.stack(
                    [np.asarray(e, dtype=np.float64).ravel() for e in embed_samples]
                ),
                axis=0,
            )
            norm = float(np.linalg.norm(embed))
            embed = embed / norm if norm > 0 else None
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
                embed=embed,
                mean_height=float(np.mean(heights)) if heights else 0.0,
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


def spatial_continuity(
    ca: Center, cb: Center, gap_ms: int, tol_scale: float = 1.0
) -> float:
    """exp(-(dist/tol)^2) between the adjacent fragment endpoints.

    tol = SPATIAL_BASE_TOL + SPATIAL_TOL_PER_S * gap_seconds: a longer absence
    allows more drift; tol_scale widens the allowance for mobile clusters.
    Neutral 0.5 when endpoint centers are unavailable (legacy callers
    constructing RawTrack without centers).
    """
    if ca is None or cb is None:
        return 0.5
    dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    tol = (SPATIAL_BASE_TOL + SPATIAL_TOL_PER_S * (max(0, gap_ms) / 1000.0)) * tol_scale
    return math.exp(-((dist / tol) ** 2))


@dataclass
class _Cluster:
    raw_ids: list[int]
    intervals: list[Interval]
    hist: Optional[np.ndarray]
    hist_weight: float
    mean_area: float
    n_dets: int
    embed: Optional[np.ndarray] = None
    embed_weight: float = 0.0
    mean_height: float = 0.0
    # Stamped by merge_tracks against the video-wide height p90; pair_score
    # (no population context) leaves it False, disabling the adult prior.
    adult: bool = False
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
        embed=None if t.embed is None else np.asarray(t.embed, dtype=np.float64).ravel(),
        embed_weight=float(t.n_dets if t.embed is not None else 0),
        mean_height=t.mean_height,
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


def _endpoint_centers(cluster: "_Cluster") -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for _s, _e, sc, ec in cluster.intervals:
        if sc is not None:
            centers.append(sc)
        if ec is not None:
            centers.append(ec)
    return centers


def _center_stats(cluster: "_Cluster") -> Optional[tuple[float, tuple[float, float]]]:
    """(spread, mean center) of a cluster's interval-endpoint centers.

    Endpoint centers are a cheap stand-in for the full trajectory: a seated
    student's endpoints cluster at the desk, the walking teacher's spread
    across the room. None when the cluster carries no center data (legacy
    callers).
    """
    centers = _endpoint_centers(cluster)
    if not centers:
        return None
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    return spread, (sum(xs) / len(xs), sum(ys) / len(ys))


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

    stats_a = _center_stats(a)
    stats_b = _center_stats(b)
    # Seat veto: two stationary clusters anchored at different desks are
    # different students no matter how well their uniforms correlate.
    if stats_a is not None and stats_b is not None:
        spread_a, center_a = stats_a
        spread_b, center_b = stats_b
        if spread_a <= SEATED_RANGE and spread_b <= SEATED_RANGE:
            anchor_dist = math.hypot(
                center_a[0] - center_b[0], center_a[1] - center_b[1]
            )
            if anchor_dist > SEATED_VETO_DIST:
                return None

    mobile_a = stats_a is not None and stats_a[0] > MOBILE_RANGE
    mobile_b = stats_b is not None and stats_b[0] > MOBILE_RANGE
    mobile = mobile_a or mobile_b
    spatial = spatial_continuity(ca, cb, gap, MOBILE_TOL_SCALE if mobile else 1.0)

    cos = None
    if a.embed is not None and b.embed is not None:
        # Embeddings are stored L2-normalized, so the dot product IS cosine.
        cos = float(np.dot(a.embed, b.embed))
        if cos < EMBED_VETO_COS:
            # Different faces/hair/build: different people regardless of how
            # well uniforms correlate or how close the seats are.
            return None

    if cos is not None and a.hist is not None and b.hist is not None:
        # Map cos from the surviving [veto, 1] band onto [0, 1] so a
        # barely-above-veto pair does not still score 0.35 appearance credit.
        cos_mapped = min(1.0, (cos - EMBED_VETO_COS) / (1.0 - EMBED_VETO_COS))
        appearance = 0.5 * cos_mapped + 0.5 * hist_correlation(a.hist, b.hist)
    elif a.hist is not None and b.hist is not None:
        appearance = hist_correlation(a.hist, b.hist)
    else:
        # No appearance evidence (e.g. /rederive from rows persisted before
        # hists/embeds were stashed): spatial carries the appearance slot.
        appearance = spatial

    # Adult-size prior: two adult-tall WALKING fragments are almost certainly
    # the one teacher, whose leave-and-return breaks spatial continuity. Both
    # sides must be mobile (stationary front-row students reach p90 height on
    # perspective CCTV), and the prior may only override appearance when the
    # embeds agree (cos >= 0.5) or when no appearance evidence exists at all;
    # a contradicting hist without embed confirmation, or a cos in
    # [0.35, 0.5), still keeps different adults apart.
    if mobile_a and mobile_b:
        adult_pair = a.adult and b.adult
        if not adult_pair and (a.adult or b.adult) and ca is not None and cb is not None:
            # Crouch re-acquisition (see CROUCH_REACQUIRE_*): the non-adult
            # side joins the adult tier only through a near-instant same-spot
            # re-acquisition, never through appearance alone.
            adult_pair = (
                gap <= CROUCH_REACQUIRE_MAX_GAP_MS
                and math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= CROUCH_REACQUIRE_MAX_DIST
            )
        if adult_pair:
            embeds_agree = cos is not None and cos >= ADULT_PRIOR_MIN_COS
            no_appearance_evidence = cos is None and (a.hist is None or b.hist is None)
            if embeds_agree or no_appearance_evidence:
                appearance = max(appearance, ADULT_PRIOR_APPEARANCE)
    return (
        W_HIST * appearance
        + W_SPATIAL * spatial
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
    if a.embed is None and b.embed is None:
        embed, embed_weight = None, 0.0
    elif a.embed is None:
        embed, embed_weight = b.embed, b.embed_weight
    elif b.embed is None:
        embed, embed_weight = a.embed, a.embed_weight
    else:
        ea, eb = max(a.embed_weight, 1.0), max(b.embed_weight, 1.0)
        embed = (a.embed * ea + b.embed * eb) / (ea + eb)
        norm = float(np.linalg.norm(embed))
        # A blend of unit vectors is not unit; re-normalize so future dot
        # products against this cluster stay true cosines.
        embed = embed / norm if norm > 0 else None
        embed_weight = ea + eb
    n = a.n_dets + b.n_dets
    area = (a.mean_area * a.n_dets + b.mean_area * b.n_dets) / max(n, 1)
    height = (a.mean_height * a.n_dets + b.mean_height * b.n_dets) / max(n, 1)
    return _Cluster(
        raw_ids=sorted(a.raw_ids + b.raw_ids),
        intervals=_coalesce(
            sorted(a.intervals + b.intervals, key=lambda iv: (iv[0], iv[1]))
        ),
        hist=hist,
        hist_weight=weight,
        mean_area=area,
        n_dets=n,
        embed=embed,
        embed_weight=embed_weight,
        mean_height=height,
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
    # Adult-size prior population cut: the classroom height distribution is
    # bimodal (many seated children, one walking adult), so the 90th
    # percentile of per-track mean heights isolates the adult tier. Legacy
    # callers whose RawTracks carry no heights (all 0) disable the prior.
    heights = [t.mean_height for t in raw_tracks if t.mean_height > 0.0]
    p90 = float(np.percentile(heights, ADULT_HEIGHT_PERCENTILE)) if heights else 0.0

    alive: dict[int, _Cluster] = {}
    for cid, t in enumerate(raw_tracks):
        cluster = _cluster_from_track(t)
        cluster.adult = p90 > 0.0 and cluster.mean_height >= p90
        alive[cid] = cluster
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
        # Re-derive adulthood from the merged mean height against the fixed
        # population p90 (absorbing a short misdetection fragment can demote
        # a cluster; two adult halves stay adult).
        merged.adult = p90 > 0.0 and merged.mean_height >= p90
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
