"""Teacher timeline stitching: reclaim the teacher's trajectory from tracker steals.

A classroom has exactly one walking adult, and she spends the lesson visiting
seated students. Every visit follows the same fatal pattern for tracking: she
crouches or leans at a desk, her bbox collapses onto the student's, her raw
track dies — and when she stands back up the tracker resumes on the STUDENT's
raw id (which then walks away teacher-sized) or on a fresh id far from where
the merge expects her. Observed on real footage as three recurring failures:

STEAL-BY-CROUCH  a student's raw id that sat static for minutes suddenly
                 grows to teacher height and walks away, exactly at the
                 teacher's disappearance point.
CHAIN-CHIMERA    greedy merge chains her mid-video fragment into a student
                 identity via mobile-tolerance widening; nothing at the role
                 level could pull a raw fragment back out of a merged
                 identity.
FAKE-CONTINUATION identity-level absorption folded a corner-student fragment
                 into her identity because it started inside an absence
                 window near her trajectory — absorption never checked size.

The stitcher replaces identity-level absorption. It seeds from the most
teacher-like fragment of the teacher identity, fits a height-vs-y model on
the seed's standing detections (perspective makes "teacher-sized" depend on
the room row: ~0.36 mid-room vs ~0.45+ near the camera), then extends the
timeline fragment by fragment:

- CONTINUE: a fragment alive at the chain's end, teacher-sized there and
  co-located with her last position, is claimed from that point on (covers
  duplicate-box handoffs and merge chimeras; the fragment's earlier
  detections stay with their student identity).
- SPLIT: a fragment alive at her disappearance point that is NOT
  teacher-sized there, but shows a sustained rise to teacher height within
  the next few seconds (the seated student standing up = her), is claimed
  from the rise onward.
- START: a fragment born shortly after her disappearance, near it and
  teacher-sized, is claimed whole.
- RECOVER: when she crossed the room untracked (crouched walk, occlusion) no
  position anchor exists; a strongly teacher-like fragment (tall, mobile,
  embed-compatible) starting within the recovery window is claimed anyway.

Detection ranges of the ORIGINAL teacher identity that the chain never
claims are EVICTED to fresh student tracks — that is what un-does a fake
continuation. Everything is deterministic from detections (+ optional
embeds), so /analyze and /rederive produce identical timelines.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.models import Detection

# --- teacher-size test -------------------------------------------------------
# Ratios are det_height / model-predicted teacher height at that y.
# Leaning/crouching at a desk halves her height, so continuation at a tight
# spatial anchor tolerates a lean...
LEAN_RATIO = 0.50
# ...while claims without a tight anchor (fragment starts elsewhere) must look
# unambiguously teacher-sized...
START_RATIO = 0.60
# ...and anchor-free recovery claims must be beyond doubt: tall AND walking.
RECOVER_RATIO = 0.75
RECOVER_MIN_SPREAD = 0.05

# --- chaining geometry -------------------------------------------------------
# Allowed distance between her last seen position and where a claim begins,
# growing with the gap (she may drift while untracked), capped so a claim
# never teleports across the room without the recovery-grade evidence.
CHAIN_BASE_TOL = 0.10
CHAIN_TOL_PER_S = 0.008
CHAIN_TOL_MAX = 0.25
# A fragment alive at her disappearance must itself sit at the crouch point.
HANDOFF_DIST = 0.20
# How far past the disappearance a steal's stand-up rise may start. A teacher
# leaning over a pupil's work stays crouched for many seconds; measured ~22s
# on real footage, so the window must outlast a genuine desk visit.
HANDOFF_WINDOW_MS = 25_000
# Fragments starting later than this after her disappearance are a fresh
# scene, not a continuation; recovery shares the same horizon.
MAX_CHAIN_GAP_MS = 30_000
# Sustained = this many consecutive samples pass the ratio test.
SUSTAIN_SAMPLES = 5
# Claims and evictions below this size are noise, not segments.
MIN_CLAIM_DETS = 3
# The teacher is the one mobile adult: a fragment that sits at one spot for a
# sustained span is a seated student, never her, no matter how teacher-tall
# perspective makes it or how close it lands to her last position. (She can
# stand still at the board, but that is always part of a longer mobile
# fragment — walk in, write, walk off — so its overall spread stays high.)
SEATED_STATIC_MIN_MS = 6_000
SEATED_STATIC_SPREAD = 0.05
# When several fragments are alive at a handoff and all clear the height and
# proximity gates, the teacher is the one that MOVES: a fidgety seated student
# spreads ~0.1, she spreads most of the frame. Candidate cost subtracts a
# mobility reward so "follow the walking adult" beats "grab the nearest box"
# (the nearest box at a desk visit is usually the pupil she leaned over).
MOBILITY_REWARD = 0.35
# Embeds are weakly discriminative on classroom footage (different people
# reach cos 0.93), so they only ever veto, mirroring merge.EMBED_VETO_COS.
EMBED_VETO_COS = 0.35


def _center(d: Detection) -> tuple[float, float]:
    return d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0


def _cy(d: Detection) -> float:
    return d.bbox["y"] + d.bbox["h"] / 2.0


def _h(d: Detection) -> float:
    return max(0.0, d.bbox["h"])


@dataclass
class HeightModel:
    """Teacher height as a linear function of bbox-center y (perspective)."""

    a: float
    b: float
    floor: float
    cap: float  # her tallest observed standing bbox; the model never predicts above this

    def predict(self, cy: float) -> float:
        return min(self.cap, max(self.floor, self.a + self.b * cy))

    def ratio(self, d: Detection) -> float:
        return _h(d) / self.predict(_cy(d))


def fit_height_model(dets: list[Detection]) -> HeightModel:
    """Fit h ~ a + b*cy on the seed's STANDING detections.

    Standing-only keeps her crouches out of the fit. Degenerate fits (too few
    points, no y-spread, or a nonsensical negative slope — on CCTV people
    lower in the frame are closer, therefore taller) fall back to a flat
    p75-height model.

    The prediction is CAPPED at her tallest observed standing bbox: a raw
    linear fit trained on mid-room board points extrapolates absurdly at the
    frame bottom (predicts 0.68 where her real max is 0.51), which crushes the
    height ratio of a genuine crouch at a near-camera desk below the lean
    threshold. The cap keeps the perspective slope where it was fitted and
    flattens the runaway extrapolation.
    """
    standing = [d for d in dets if d.standing]
    sample = standing if len(standing) >= 10 else dets
    hs = np.array([_h(d) for d in sample], dtype=np.float64)
    p75 = float(np.percentile(hs, 75)) if len(hs) else 0.3
    cap = float(np.percentile(hs, 95)) * 1.05 if len(hs) else 0.5
    cys = np.array([_cy(d) for d in sample], dtype=np.float64)
    if len(sample) >= 10 and float(cys.max() - cys.min()) > 0.1:
        b, a = np.polyfit(cys, hs, 1)
        if b > 0:
            return HeightModel(a=float(a), b=float(b), floor=0.6 * p75, cap=cap)
    return HeightModel(a=p75, b=0.0, floor=0.6 * p75, cap=max(cap, p75))


@dataclass
class Fragment:
    """One raw tracker id's detections inside one merged identity."""

    raw_id: int
    host_track_no: int
    dets: list[Detection]  # ts-sorted
    ts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ts = [d.video_ts_ms for d in self.dets]

    @property
    def first_ms(self) -> int:
        return self.ts[0]

    @property
    def last_ms(self) -> int:
        return self.ts[-1]

    def alive_at(self, t: int, slack_ms: int = 1_500) -> bool:
        return self.first_ms <= t + slack_ms and self.last_ms >= t - slack_ms

    def index_at(self, t: int) -> int:
        """First det index with ts >= t (clamped to a valid index)."""
        return min(bisect_left(self.ts, t), len(self.dets) - 1)

    def center_near(self, t: int, window_ms: int = 1_000) -> Optional[tuple[float, float]]:
        lo = bisect_left(self.ts, t - window_ms)
        hi = bisect_right(self.ts, t + window_ms)
        if lo >= hi:
            return None
        xs = sorted(_center(d)[0] for d in self.dets[lo:hi])
        ys = sorted(_center(d)[1] for d in self.dets[lo:hi])
        return xs[len(xs) // 2], ys[len(ys) // 2]

    def spread(self, from_idx: int = 0) -> float:
        cs = [_center(d) for d in self.dets[from_idx:]]
        if not cs:
            return 0.0
        xs = [c[0] for c in cs]
        ys = [c[1] for c in cs]
        return max(max(xs) - min(xs), max(ys) - min(ys))


@dataclass
class Claim:
    fragment: Fragment
    from_idx: int  # claim covers dets[from_idx:]

    @property
    def dets(self) -> list[Detection]:
        return self.fragment.dets[self.from_idx :]


def _chain_tol(gap_ms: int) -> float:
    return min(
        CHAIN_TOL_MAX, CHAIN_BASE_TOL + CHAIN_TOL_PER_S * max(0, gap_ms) / 1000.0
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seated_static(frag: "Fragment", from_idx: int) -> bool:
    """True if the claimed portion sits at one spot long enough to be a student.

    A short portion is never judged (a brief handoff blip has no room to move);
    only a sustained, near-motionless span rules the teacher out.
    """
    dets = frag.dets[from_idx:]
    if len(dets) < 2:
        return False
    if dets[-1].video_ts_ms - dets[0].video_ts_ms < SEATED_STATIC_MIN_MS:
        return False
    return frag.spread(from_idx) < SEATED_STATIC_SPREAD


def _sustained_ratio(
    model: HeightModel, dets: list[Detection], start: int, threshold: float
) -> bool:
    window = dets[start : start + SUSTAIN_SAMPLES]
    # Require a FULL window of SUSTAIN_SAMPLES consecutive samples. (The old
    # guard compared len(window) against min(SUSTAIN_SAMPLES, len(dets)-start),
    # which is exactly len(window) for any valid start, so it never fired and a
    # 1-4 sample tail counted as "sustained".)
    if len(window) < SUSTAIN_SAMPLES:
        return False
    return all(model.ratio(d) >= threshold for d in window)


def _tail_anchor(
    model: HeightModel, claimed: list[Detection]
) -> tuple[int, tuple[float, float]]:
    """(last ts, robust last position) of the chain so far.

    A dying box shrinks and slides in its final samples (she crouched or the
    tracker lost her), so the position anchor is the median center of the
    last few detections that still look at least half teacher-sized.
    """
    t = claimed[-1].video_ts_ms
    tail = [d for d in claimed[-5:] if model.ratio(d) >= 0.5]
    if not tail:
        tail = claimed[-1:]
    xs = sorted(_center(d)[0] for d in tail)
    ys = sorted(_center(d)[1] for d in tail)
    return t, (xs[len(xs) // 2], ys[len(ys) // 2])


def _embed_ok(
    teacher_embed: Optional[np.ndarray],
    embeds_by_raw: Optional[dict[int, np.ndarray]],
    raw_id: int,
) -> bool:
    """Embeds only VETO (cos below the different-person floor); absence passes."""
    if teacher_embed is None or not embeds_by_raw:
        return True
    emb = embeds_by_raw.get(raw_id)
    if emb is None:
        return True
    return float(np.dot(teacher_embed, emb)) >= EMBED_VETO_COS


def build_fragments(dets_by_track: dict[int, list[Detection]]) -> list[Fragment]:
    frags: list[Fragment] = []
    for track_no, dets in dets_by_track.items():
        by_raw: dict[int, list[Detection]] = {}
        for d in dets:
            by_raw.setdefault(d.raw_track_id, []).append(d)
        for raw_id, group in by_raw.items():
            group.sort(key=lambda d: d.video_ts_ms)
            frags.append(Fragment(raw_id=raw_id, host_track_no=track_no, dets=group))
    frags.sort(key=lambda f: f.first_ms)
    return frags


def _pick_seed(model_frags: list[Fragment], teacher_no: int) -> Optional[Fragment]:
    """Most teacher-like fragment of the teacher identity: longest tall span."""
    own = [f for f in model_frags if f.host_track_no == teacher_no]
    if not own:
        return None

    def key(f: Fragment) -> float:
        span = f.last_ms - f.first_ms
        mean_h = float(np.mean([_h(d) for d in f.dets]))
        return span * mean_h

    return max(own, key=key)


def stitch_teacher(
    teacher_no: int,
    dets_by_track: dict[int, list[Detection]],
    embeds_by_raw: Optional[dict[int, np.ndarray]] = None,
) -> Optional[tuple[list[Claim], list[tuple[Fragment, int, int]]]]:
    """Stitch the teacher's timeline; returns (claims, evictions) or None.

    claims     ordered Claim list (includes the seed as claims[0]).
    evictions  (fragment, start_idx, end_idx) det ranges of the ORIGINAL
               teacher identity that the chain rejected — the caller must
               move these to fresh student tracks.
    """
    fragments = build_fragments(dets_by_track)
    seed = _pick_seed(fragments, teacher_no)
    if seed is None or len(seed.dets) < MIN_CLAIM_DETS:
        return None
    model = fit_height_model(seed.dets)
    teacher_embed = (embeds_by_raw or {}).get(seed.raw_id)

    claims: list[Claim] = [Claim(fragment=seed, from_idx=0)]
    claimed_from: dict[int, int] = {id(seed): 0}  # fragment -> claimed from_idx

    # --- backward: fragments ending just before the seed begins --------------
    start_t = seed.first_ms
    start_p = _center(seed.dets[0])
    while True:
        best: Optional[Fragment] = None
        best_d = None
        for f in fragments:
            if id(f) in claimed_from or f.last_ms >= start_t:
                continue
            gap = start_t - f.last_ms
            if gap > MAX_CHAIN_GAP_MS:
                continue
            tail = f.center_near(f.last_ms)
            if tail is None or _dist(tail, start_p) > _chain_tol(gap):
                continue
            if _seated_static(f, 0):
                continue
            if not _sustained_ratio(model, f.dets, max(0, len(f.dets) - SUSTAIN_SAMPLES), START_RATIO):
                continue
            if not _embed_ok(teacher_embed, embeds_by_raw, f.raw_id):
                continue
            d = _dist(tail, start_p) + gap / 1000.0 * 0.01
            if best is None or d < best_d:
                best, best_d = f, d
        if best is None:
            break
        claims.insert(0, Claim(fragment=best, from_idx=0))
        claimed_from[id(best)] = 0
        start_t, start_p = best.first_ms, _center(best.dets[0])

    # --- forward --------------------------------------------------------------
    while True:
        chain_dets = claims[-1].dets
        t, p = _tail_anchor(model, chain_dets)
        candidate: Optional[Claim] = None
        candidate_cost = None

        for f in fragments:
            if id(f) in claimed_from or f.last_ms <= t:
                continue

            if f.first_ms <= t:
                # CONTINUE: the fragment coexists with her last moment. Its
                # detections BEFORE the first teacher-height sample stay with
                # whatever identity owned them — either a student box she leaned
                # over (raw id swaps onto her only when she stands) or, when the
                # whole lead-in is sub-height, a different seated pupil at the
                # same corner. The claim begins at the first sustained
                # teacher-height sample within the desk-visit window, so a
                # genuine lean (tall enough to clear LEAN_RATIO once the height
                # model is capped) is kept while a short seated pupil is trimmed.
                pos = f.center_near(t)
                if pos is None or _dist(pos, p) > HANDOFF_DIST:
                    continue
                idx = f.index_at(t)
                horizon = max(idx + 1, bisect_right(f.ts, t + HANDOFF_WINDOW_MS))
                claim_idx = next(
                    (
                        i
                        for i in range(idx, horizon)
                        if _sustained_ratio(model, f.dets, i, LEAN_RATIO)
                    ),
                    None,
                )
                if claim_idx is None or _seated_static(f, claim_idx):
                    continue
                if len(f.dets) - claim_idx < MIN_CLAIM_DETS:
                    continue  # too short to be a real claim; skip, don't abort
                mobility = MOBILITY_REWARD * min(1.0, f.spread(claim_idx))
                # Prefer an immediate continuation and the more mobile fragment;
                # a long sub-height trim (she was out of frame a while) costs
                # more so a cleaner continuation wins when one exists.
                cost = (
                    _dist(pos, p)
                    + (f.ts[claim_idx] - t) / 1000.0 * 0.02
                    - mobility
                )
                if candidate is None or cost < candidate_cost:
                    candidate, candidate_cost = Claim(f, claim_idx), cost
                continue

            if f.first_ms - t > MAX_CHAIN_GAP_MS:
                continue
            # START: a fresh fragment born near her disappearance.
            gap = f.first_ms - t
            head = f.center_near(f.first_ms)
            if head is None or _dist(head, p) > _chain_tol(gap):
                continue
            if _seated_static(f, 0):
                continue
            if len(f.dets) < MIN_CLAIM_DETS:
                continue  # too short to be a real claim; skip, don't abort
            if not _sustained_ratio(model, f.dets, 0, START_RATIO):
                continue
            if not _embed_ok(teacher_embed, embeds_by_raw, f.raw_id):
                continue
            cost = (
                _dist(head, p)
                + gap / 1000.0 * 0.01
                - MOBILITY_REWARD * min(1.0, f.spread(0))
            )
            if candidate is None or cost < candidate_cost:
                candidate, candidate_cost = Claim(f, 0), cost

        if candidate is None:
            # RECOVER: she crossed the room untracked; no position anchor.
            for f in fragments:
                if id(f) in claimed_from or f.last_ms <= t:
                    continue
                idx = f.index_at(max(t, f.first_ms)) if f.first_ms <= t else 0
                start_ms = f.ts[idx]
                # Fragments of the teacher's own merged identity were already
                # vetted by the merge (embed veto / adult prior), so a genuine
                # leave-and-return beyond the recovery horizon still counts;
                # foreign fragments must start within the horizon.
                if start_ms - t > MAX_CHAIN_GAP_MS and f.host_track_no != teacher_no:
                    continue
                if len(f.dets) - idx < MIN_CLAIM_DETS:
                    continue
                rest = f.dets[idx:]
                ratios = [model.ratio(d) for d in rest]
                if float(np.mean(ratios)) < RECOVER_RATIO:
                    continue
                if f.spread(idx) < RECOVER_MIN_SPREAD:
                    continue
                if not _embed_ok(teacher_embed, embeds_by_raw, f.raw_id):
                    continue
                cost = (start_ms - t) / 1000.0
                if candidate is None or cost < candidate_cost:
                    candidate, candidate_cost = Claim(f, idx), cost

        # Every branch already rejects sub-MIN_CLAIM_DETS fragments during
        # selection, so a real winner is never too short here; only a genuinely
        # empty round ends the forward walk. (The old terminal length check
        # broke the whole loop when a tiny fragment won on cost, abandoning
        # every still-unclaimed valid claim after it.)
        if candidate is None:
            break
        claims.append(candidate)
        claimed_from[id(candidate.fragment)] = candidate.from_idx

    # --- evictions: teacher-identity ranges the chain rejected ----------------
    evictions: list[tuple[Fragment, int, int]] = []
    for f in fragments:
        if f.host_track_no != teacher_no:
            continue
        upto = claimed_from.get(id(f))
        if upto is None:
            evictions.append((f, 0, len(f.dets)))
        elif upto >= MIN_CLAIM_DETS:
            evictions.append((f, 0, upto))
    return claims, evictions
