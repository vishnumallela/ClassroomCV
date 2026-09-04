"""Which tracked body is THIS lesson's teacher. Phase 3 of the plan.

app/teacher.py hands over segments: bodies the motion model could follow
without ambiguity, cut wherever it could not. Three things are still wrong
with them for a lesson with more than one adult, and this module fixes each
with the evidence that decides it:

  1. A segment can quietly change person at an occluded crossing (the real
     handover did, once). APPEARANCE finds the change-point: split there.
  2. One person is many segments — she sat, she stood, she stepped out, the
     detector flickered. APPEARANCE plus CONTINUITY link them back into people.
     Neither alone is enough on real footage: a student standing in front of
     her recolours her box, and two people standing still are the same place.
  3. Nothing says which person the lesson assesses. The TIMETABLE decides,
     through one observable fact: the teacher whose period it is stays to the
     end, and the one who hands over leaves while the other remains.

The output is a choice plus a confidence and the reason, and `undetermined` is
an allowed answer that leaves Phase 0's refusal in force. Every number here is
a measurement from the two real lessons named in the plan; every threshold
names what it was set against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from app import appearance as app_mod
from app.models import Detection
from app.teacher import (
    FREE_GAP_MS,
    JUMP_BASE,
    MAX_SPEED_PER_S,
    SEGMENT_MIN_DETS,
    SEGMENT_MIN_MS,
    Segment,
    _center,
    _plausible,
)

# --- descriptors worth trusting ---------------------------------------------
# A box this short is a head, and its "torso band" is a face. Its descriptor
# says nothing about clothing and is not used for anything appearance decides.
CLEAN_MIN_H = 0.15
CLEAN_MIN_CONF = 0.5
# A piece may speak for its appearance only when a person's worth of its
# boxes are clean — as a count AND as a share. On the handover the seated
# teacher's head-only lane (median height 0.12) still had 28 boxes over the
# height floor, 24% of it, and a descriptor built from those landed 0.23 from
# the other teacher's doorway piece: linked, and her "arrival" became t=0.
CLEAN_MIN_COUNT = 8
CLEAN_MIN_SHARE = 0.5

# --- swap: did the tracker hand two bodies each other's future? ------------
# At an OCCLUDED crossing one body goes undetected for a moment and the motion
# model gives its next box to whichever lane is nearest, which is the other
# body's. From then on each lane carries the other person. On the full 45-minute
# handover recording a colleague in white walked in past the period's teacher
# (black) at 2381 s: the teacher's 35-minute lane followed the colleague to a
# desk, and the teacher's remaining five minutes became a new lane — which then
# could not rejoin her, because her own lane was still "alive" on the wrong
# body. The change-point statistic cannot cut that lane: black against white
# scores 0.33 under this descriptor, below the 0.36 a single teacher reaches on
# the baseline through an ordinary occlusion.
#
# What does settle it is the PAIR. At the instant of the swap, the two lanes'
# before/after windows read black|white and white|black; re-paired, they read
# black|black and white|white. Measured there: 0.33 + 0.44 as tracked against
# 0.11 + 0.25 re-paired — a margin of 0.4. The same test at the clean crossing
# (velocity carried the walker through) reads cream|cream and black|black
# against 0.3-0.48 crossed, so it leaves that alone, and a lane with no second
# substantial lane beside it is never examined, so a one-adult lesson is
# untouched by construction. A false swap would need BOTH cross terms small:
# a student recolouring one lane's window pushes one of them up, not down.
SWAP_WINDOW_MS = 6_000
SWAP_MARGIN = 0.25
SWAP_MIN_CLEAN = 5

# --- change-point: did this segment change person? -------------------------
# Mean descriptor of the W boxes before an instant against the W after. On
# the handover the swap peaks at 0.48-0.52 with a 20-box window; the highest
# the same statistic reaches anywhere else on either real lesson is below the
# threshold by the margin noted in the plan. A split needs the "after" side to
# be at least a person, or a flicker could halve a segment.
# Window 30, threshold 0.42: on the 37-minute single-teacher baseline the
# statistic never exceeds 0.355 (0.44 at window 20 — one occlusion event
# near 1034 s); on the handover the swap peaks at 0.48 and 0.52 and nothing
# else on either segment passes 0.29. Measured 2026-09-03 with
# tools/ab_tracker.py's feature dumps; re-measure before moving either.
SPLIT_WINDOW = 30
SPLIT_THRESHOLD = 0.42

# --- linking: are these two pieces one person? -------------------------------
# Cost = appearance distance between the pieces' clean-mean descriptors, plus a
# position term that is decisive when the gap is short and neutral once it is
# long enough that position carries no information (FREE_GAP_MS, same as the
# tracker). Overlapping pieces at the same place are one body seen twice.
OVERLAP_TOL_MS = 3_000
SAME_PLACE = 0.10
# A duplicate lane: at this share of shared instants, the narrower box's
# x-range is at least this contained in the other's, and their centres are
# within this much vertically (a head box sits above a torso box's centre).
SAME_COLUMN = 0.8
SAME_COLUMN_DY = 0.20
SAME_COLUMN_SHARE = 0.8
POS_WEIGHT = 0.30
POS_NEUTRAL = 0.10
# A non-adjacent link (gap past FREE_GAP_MS) is appearance-only and must clear
# this. Same-person pieces on the handover sit at 0.10-0.25; cream vs black at
# 0.21-0.47, and the one cross-person pair below 0.30 is adjacent in neither
# time nor place. Adjacent links are gated by position instead.
APP_LINK_MAX = 0.30

# --- attribution ---------------------------------------------------------------
# A person whose final sighting is followed by another adult's presence for at
# least this long has handed the room over. Ten seconds is two sampled seconds
# beyond any single-frame miss.
HANDOVER_MIN_MS = 10_000
# How much of the observable window the remaining adult must then hold alone
# for the handover to be called with high confidence. The 6-minute trim of the
# real handover leaves 24 s after the outgoing teacher exits — enough to name
# the right adult, not enough to grade her on; the full recording leaves 39
# minutes. "Alone" is not the only way to be sure, though: on the full
# recording a colleague sat at a desk for the last five minutes and stopped
# being detected 12 s before the end, which read as "held the room alone for
# 12 s". A candidate with LEAD_HIGH times the presence of anyone who left is
# also called with high confidence.
HANDOVER_HIGH_MS = 60_000
# Presence margin between the top two candidates when nobody handed over, and
# between the one who stayed and the ones who left.
LEAD_HIGH = 2.0
LEAD_MEDIUM = 1.3
# When did an adult who was in the room at the bell LEAVE? Not her last
# sighting: pieces of two light-dressed adults can link across a long gap
# (cream stripes against a white shirt sit at 0.16 under this descriptor), and
# then the period-2 teacher's departure would read as the colleague's. Her
# departure is the end of the presence run that contains the bell, bridging
# absences shorter than this — a teacher who steps out for a minute has not
# left; one gone for five has.
LEFT_BRIDGE_MS = 300_000
AT_BELL_TOL_MS = 10_000


@dataclass
class Person:
    """One adult: the segments that are them, in time order."""

    segments: list[Segment]
    track_no: Optional[int] = None

    @property
    def detections(self) -> list[Detection]:
        out = [d for s in self.segments for d in s.detections]
        out.sort(key=lambda d: d.video_ts_ms)
        return out

    @property
    def first_ms(self) -> int:
        return min(s.first_ms for s in self.segments)

    @property
    def last_ms(self) -> int:
        return max(s.last_ms for s in self.segments)

    @property
    def substantial(self) -> bool:
        n = sum(len(s.detections) for s in self.segments)
        return n >= SEGMENT_MIN_DETS and (self.last_ms - self.first_ms) >= SEGMENT_MIN_MS


@dataclass
class Candidate:
    track_no: int
    first_ms: int
    last_ms: int
    present_ms: int
    in_period_ms: int
    handed_over: bool
    segments: int
    # When an adult who was in the room at the start of the window left it:
    # the end of her presence run containing the bell (see LEFT_BRIDGE_MS).
    # None when she was not there at the bell.
    left_ms: Optional[int] = None


@dataclass
class Attribution:
    persons: list[Person]
    chosen: Optional[Person]
    confidence: str  # high | medium | low
    reason: str
    candidates: list[Candidate] = field(default_factory=list)
    period_known: bool = False
    splits: int = 0
    swaps: int = 0

    def as_report(self) -> dict:
        return {
            "confidence": self.confidence,
            "reason": self.reason,
            "chosen_track_no": self.chosen.track_no if self.chosen else None,
            "period_known": self.period_known,
            "splits": self.splits,
            "swaps": self.swaps,
            "candidates": [c.__dict__ for c in self.candidates],
        }


# --------------------------------------------------------------------------- #
# appearance helpers
# --------------------------------------------------------------------------- #


def _clean(dets: list[Detection]) -> list[list[float]]:
    return [
        d.app
        for d in dets
        if d.app is not None and d.bbox["h"] >= CLEAN_MIN_H and d.conf >= CLEAN_MIN_CONF
    ]


def _mean_app(dets: list[Detection]) -> Optional[list[float]]:
    clean = _clean(dets)
    if len(clean) < CLEAN_MIN_COUNT or len(clean) < CLEAN_MIN_SHARE * len(dets):
        return None
    return app_mod.mean_descriptor(clean)


def _robust_end(dets: list[Detection], first: bool, k: int = 5) -> tuple[float, float]:
    sel = dets[:k] if first else dets[-k:]
    xs = [_center(d)[0] for d in sel]
    ys = [_center(d)[1] for d in sel]
    return median(xs), median(ys)


# --------------------------------------------------------------------------- #
# 0. undo the tracker's swaps at occluded crossings
# --------------------------------------------------------------------------- #


def _window_app(dets: list[Detection]) -> Optional[list[float]]:
    """Mean clean descriptor of a short window, or None if too few speak."""
    clean = _clean(dets)
    if len(clean) < SWAP_MIN_CLEAN or len(clean) < CLEAN_MIN_SHARE * len(dets):
        return None
    return app_mod.mean_descriptor(clean)


def _swap_score(x: Segment, y: Segment, t: int) -> Optional[float]:
    """How much cheaper it is to give `x` and `y` each other's boxes from `t`
    on: (as tracked) minus (re-paired), in appearance distance. None when a
    side has too few clean boxes to speak."""
    lo, hi = t - SWAP_WINDOW_MS, t + SWAP_WINDOW_MS
    xb = _window_app([d for d in x.detections if lo <= d.video_ts_ms < t])
    xa = _window_app([d for d in x.detections if t <= d.video_ts_ms <= hi])
    yb = _window_app([d for d in y.detections if lo <= d.video_ts_ms < t])
    ya = _window_app([d for d in y.detections if t <= d.video_ts_ms <= hi])
    if not (xb and xa and yb and ya):
        return None
    as_tracked = app_mod.distance(xb, xa) + app_mod.distance(yb, ya)
    re_paired = app_mod.distance(xb, ya) + app_mod.distance(yb, xa)
    return as_tracked - re_paired


def _find_swap(x: Segment, y: Segment) -> Optional[int]:
    """The instant at which `x` most clearly took over `y`'s body, or None.

    Only instants where the two lanes were within reach of each other are
    examined — the box `x` took could have been `y`'s next box — because that
    is the only moment a motion model can confuse two bodies.
    """
    y_dets = y.detections
    best_t, best_score = None, SWAP_MARGIN
    j = 0
    for d in x.detections:
        t = d.video_ts_ms
        if t <= y.first_ms or t > y.last_ms:
            continue
        while j + 1 < len(y_dets) and y_dets[j + 1].video_ts_ms < t:
            j += 1
        y_prev = y_dets[j]
        if y_prev.video_ts_ms >= t or t - y_prev.video_ts_ms >= FREE_GAP_MS:
            continue
        if not _plausible(y_prev, d):
            continue
        score = _swap_score(x, y, t)
        if score is not None and score > best_score:
            best_t, best_score = t, score
    return best_t


def _swap_tails(x: Segment, y: Segment, t: int) -> None:
    x_before = [d for d in x.detections if d.video_ts_ms < t]
    x_after = [d for d in x.detections if d.video_ts_ms >= t]
    y_before = [d for d in y.detections if d.video_ts_ms < t]
    y_after = [d for d in y.detections if d.video_ts_ms >= t]
    x.detections = x_before + y_after
    y.detections = y_before + x_after


def resolve_swaps(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Give each lane back its own body wherever the tracker swapped two.

    Rescans after every swap because a swap changes both lanes; bounded, and
    a corrected pair scores below the margin, so it cannot oscillate.
    """
    subs = [s for s in segments if s.substantial]
    swaps = 0
    for _ in range(len(subs) * len(subs) + 1):
        found = None
        for x in subs:
            for y in subs:
                if x is y:
                    continue
                t = _find_swap(x, y)
                if t is not None:
                    found = (x, y, t)
                    break
            if found:
                break
        if not found:
            break
        _swap_tails(*found)
        swaps += 1
    return [s for s in segments if s.detections], swaps


# --------------------------------------------------------------------------- #
# 1. split segments where the person changes
# --------------------------------------------------------------------------- #


def split_at_changes(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Cut each segment wherever its appearance changes and stays changed."""
    out: list[Segment] = []
    splits = 0
    for seg in segments:
        dets = seg.detections
        if len(dets) < 2 * SPLIT_WINDOW or not seg.substantial:
            out.append(seg)
            continue
        apps = [d.app if (d.app is not None and d.bbox["h"] >= CLEAN_MIN_H) else None for d in dets]
        cut_at: list[int] = []
        i = SPLIT_WINDOW
        while i <= len(dets) - SPLIT_WINDOW:
            before = app_mod.mean_descriptor([a for a in apps[i - SPLIT_WINDOW:i] if a])
            after = app_mod.mean_descriptor([a for a in apps[i:i + SPLIT_WINDOW] if a])
            if before and after and app_mod.distance(before, after) >= SPLIT_THRESHOLD:
                # Take the sharpest point of this change, then skip past it so
                # one change yields one cut.
                best_i, best_d = i, app_mod.distance(before, after)
                for j in range(i + 1, min(i + SPLIT_WINDOW, len(dets) - SPLIT_WINDOW + 1)):
                    b2 = app_mod.mean_descriptor([a for a in apps[j - SPLIT_WINDOW:j] if a])
                    a2 = app_mod.mean_descriptor([a for a in apps[j:j + SPLIT_WINDOW] if a])
                    if b2 and a2:
                        d2 = app_mod.distance(b2, a2)
                        if d2 > best_d:
                            best_i, best_d = j, d2
                cut_at.append(best_i)
                i = best_i + SPLIT_WINDOW
            else:
                i += 1
        if not cut_at:
            out.append(seg)
            continue
        splits += len(cut_at)
        prev = 0
        for c in cut_at + [len(dets)]:
            out.append(Segment(dets[prev:c]))
            prev = c
    return out, splits


# --------------------------------------------------------------------------- #
# 2. link pieces into people
# --------------------------------------------------------------------------- #


def _link_cost(a: Segment, b: Segment, app_a, app_b) -> Optional[float]:
    """Cost of `b` continuing `a`, or None when they cannot be one person."""
    gap = b.first_ms - a.last_ms
    if gap < -OVERLAP_TOL_MS:
        return None
    ea = _robust_end(a.detections, first=False)
    sb = _robust_end(b.detections, first=True)
    dist = ((ea[0] - sb[0]) ** 2 + (ea[1] - sb[1]) ** 2) ** 0.5
    app_d = app_mod.distance(app_a, app_b) if (app_a and app_b) else None

    if gap <= 0:
        # Overlapping in time: one body seen twice only if at the same place.
        return (app_d if app_d is not None else 0.0) if dist <= SAME_PLACE else None
    if gap < FREE_GAP_MS:
        allowed = JUMP_BASE + MAX_SPEED_PER_S * gap / 1000.0
        if dist > allowed:
            return None
        pos = POS_WEIGHT * min(1.0, dist / allowed)
        return (app_d if app_d is not None else POS_NEUTRAL) + pos
    # Long gap: position says nothing; appearance must carry it alone.
    if app_d is None or app_d > APP_LINK_MAX:
        return None
    return app_d + POS_NEUTRAL


def _overlap_ms(a: Segment, b: Segment) -> int:
    return max(0, min(a.last_ms, b.last_ms) - max(a.first_ms, b.first_ms))


def merge_duplicate_lanes(segments: list[Segment]) -> list[Segment]:
    """Fold a piece into a longer one it shadows: same time, same COLUMN.

    The tracker dedups two boxes on one body per instant by containment, and a
    head-only box hanging above a torso box sometimes escapes that. The result
    is a second lane on the same person for as long as the model keeps drawing
    both. Such a lane overlaps the real one for most of its life in the same
    column: at nearly every shared instant its x-range sits inside the other
    box's x-range, because it is a smaller box on the same body. Two adults
    standing side by side never satisfy that, however close — the black
    teacher spent fifteen seconds within 0.05 of the cream teacher at the
    board, and a plain "same place" rule folded her into the wrong person and
    moved her arrival by fifteen seconds. At instants both lanes have a box,
    the larger box is kept.
    """
    pieces = sorted((s for s in segments if s.detections), key=lambda s: -(s.last_ms - s.first_ms))
    kept: list[Segment] = []
    for s in pieces:
        span = max(1, s.last_ms - s.first_ms)
        host = None
        for k in kept:
            if _overlap_ms(s, k) < 0.5 * span:
                continue
            k_by_ts = {d.video_ts_ms: d for d in k.detections}
            shared = [(d, k_by_ts[d.video_ts_ms]) for d in s.detections if d.video_ts_ms in k_by_ts]
            if len(shared) < SEGMENT_MIN_DETS:
                continue
            nested = 0
            for a, b in shared:
                ax0, ax1 = a.bbox["x"], a.bbox["x"] + a.bbox["w"]
                bx0, bx1 = b.bbox["x"], b.bbox["x"] + b.bbox["w"]
                inner = min(a.bbox["w"], b.bbox["w"])
                x_contained = max(0.0, min(ax1, bx1) - max(ax0, bx0)) / inner if inner > 0 else 0.0
                dy = abs(_center(a)[1] - _center(b)[1])
                if x_contained >= SAME_COLUMN and dy <= SAME_COLUMN_DY:
                    nested += 1
            if nested >= SAME_COLUMN_SHARE * len(shared):
                host = k
                break
        if host is None:
            kept.append(s)
            continue
        by_ts = {d.video_ts_ms: d for d in host.detections}
        for d in s.detections:
            cur = by_ts.get(d.video_ts_ms)
            if cur is None or d.bbox["w"] * d.bbox["h"] > cur.bbox["w"] * cur.bbox["h"]:
                by_ts[d.video_ts_ms] = d
        host.detections = [by_ts[t] for t in sorted(by_ts)]
    return kept


def link_people(segments: list[Segment]) -> list[Person]:
    """Each piece, in start order, joins the ended piece it most plausibly
    continues; pieces nobody continues stay their own person."""
    pieces = sorted(merge_duplicate_lanes(segments), key=lambda s: s.first_ms)
    apps = {id(s): _mean_app(s.detections) for s in pieces}
    person_of: dict[int, Person] = {}
    continued: set[int] = set()  # pieces that already have a successor
    people: list[Person] = []
    for b in pieces:
        best: Optional[tuple[float, Segment]] = None
        for a in pieces:
            if a is b or a.first_ms > b.first_ms or id(a) in continued:
                continue
            if a.last_ms > b.last_ms:  # b entirely inside a: a duplicate lane
                pass
            cost = _link_cost(a, b, apps[id(a)], apps[id(b)])
            if cost is None:
                continue
            if best is None or cost < best[0]:
                best = (cost, a)
        if best is None:
            p = Person([b])
            people.append(p)
            person_of[id(b)] = p
        else:
            a = best[1]
            p = person_of[id(a)]
            p.segments.append(b)
            person_of[id(b)] = p
            continued.add(id(a))
    for p in people:
        p.segments.sort(key=lambda s: s.first_ms)
    return people


# --------------------------------------------------------------------------- #
# 3. attribute
# --------------------------------------------------------------------------- #


def _presence_ms(dets: list[Detection], lo: int, hi: int, gap_ms: int = FREE_GAP_MS) -> int:
    """Time covered by detections inside [lo, hi], bridging gaps < gap_ms."""
    ts = sorted(d.video_ts_ms for d in dets if lo <= d.video_ts_ms <= hi)
    if not ts:
        return 0
    total, start, prev = 0, ts[0], ts[0]
    for t in ts[1:]:
        if t - prev >= gap_ms:
            total += prev - start
            start = t
        prev = t
    return total + (prev - start)


def _left_ms(dets: list[Detection], lo: int) -> Optional[int]:
    """When an adult who was in the room at `lo` left it: the end of the
    presence run containing `lo`, bridging absences under LEFT_BRIDGE_MS.
    None when she was not there at `lo`."""
    ts = sorted(d.video_ts_ms for d in dets)
    if not ts or ts[0] > lo + AT_BELL_TOL_MS:
        return None
    end = ts[0]
    for t in ts[1:]:
        if t - end >= LEFT_BRIDGE_MS:
            break
        end = t
    return end


def attribute(
    segments: list[Segment],
    duration_ms: int,
    period_ms: Optional[tuple[int, int]] = None,
) -> Attribution:
    """Decide which person the lesson assesses, and say why."""
    segments, swaps = resolve_swaps(list(segments))
    pieces, splits = split_at_changes([s for s in segments if s.substantial])
    people = [p for p in link_people(pieces) if p.substantial]
    people.sort(key=lambda p: p.first_ms)
    period_known = period_ms is not None
    lo = max(0, period_ms[0]) if period_ms else 0
    hi = min(duration_ms, period_ms[1]) if period_ms else duration_ms
    if hi <= lo:
        lo, hi = 0, duration_ms

    if not people:
        return Attribution([], None, "low", "No adult was tracked for long enough to assess.",
                           period_known=period_known, splits=splits, swaps=swaps)

    cands: list[Candidate] = []
    for i, p in enumerate(people, start=1):
        dets = p.detections
        last = max(d.video_ts_ms for d in dets if d.video_ts_ms <= hi) if any(d.video_ts_ms <= hi for d in dets) else p.last_ms
        # Handed over: after this person's last sighting inside the window,
        # somebody else is present for HANDOVER_MIN_MS with this person absent.
        others_after = 0
        for q in people:
            if q is p:
                continue
            others_after = max(others_after, _presence_ms(q.detections, last, hi))
        cands.append(Candidate(
            track_no=i, first_ms=p.first_ms, last_ms=p.last_ms,
            present_ms=_presence_ms(dets, 0, duration_ms),
            in_period_ms=_presence_ms(dets, lo, hi),
            # Handed over when at least HANDOVER_MIN_MS of the window remains
            # after this person's last sighting AND somebody else fills it.
            handed_over=others_after >= HANDOVER_MIN_MS and hi - last >= HANDOVER_MIN_MS,
            segments=len(p.segments),
            left_ms=_left_ms(dets, lo),
        ))

    if len(people) == 1:
        people[0].track_no = 1
        return Attribution(people, people[0], "high", "One adult was tracked; the lesson is hers.",
                           cands, period_known, splits, swaps)

    staying = [c for c in cands if not c.handed_over]
    pool = staying if staying else cands
    pool.sort(key=lambda c: -c.in_period_ms)
    top = pool[0]
    runner = pool[1] if len(pool) > 1 else None
    handed = [c for c in cands if c.handed_over]

    if staying and len(staying) == 1 and handed:
        # The handover signature: everyone else left while this one remained.
        left_desc = ", ".join(f"one at {c.last_ms / 1000:.0f}s" for c in handed)
        remaining = min(hi, top.last_ms) - max(h.last_ms for h in handed)
        lead = top.in_period_ms / max(max(h.in_period_ms for h in handed), 1)
        enough = top.in_period_ms >= HANDOVER_HIGH_MS
        alone = remaining >= HANDOVER_HIGH_MS
        conf = "high" if enough and (alone or lead >= LEAD_HIGH) else "medium"
        reason = (f"{len(handed)} other adult(s) left while this one remained ({left_desc}); "
                  f"the adult who stayed is attributed. She then held the room alone for "
                  f"{remaining / 1000:.0f}s")
        if conf == "medium":
            reason += " — too little to grade her on; the full recording would settle it."
        elif alone:
            reason += "."
        else:
            reason += f", and was present {lead:.0f}× longer than anyone who left."
    elif runner is None or runner.in_period_ms == 0:
        conf = "high" if period_known else "medium"
        reason = "Only one adult was present during the scheduled period."
    else:
        lead = top.in_period_ms / max(runner.in_period_ms, 1)
        if lead >= LEAD_HIGH:
            conf = "high" if period_known else "medium"
        elif lead >= LEAD_MEDIUM:
            conf = "medium"
        else:
            conf = "low"
        reason = (f"{len(cands)} adults; the attributed one was present {top.in_period_ms / 1000:.0f}s "
                  f"of the {'scheduled period' if period_known else 'recording'} against {runner.in_period_ms / 1000:.0f}s"
                  + ("" if conf != "low" else " — too close to call") + ".")
        if not period_known and conf == "high":
            conf = "medium"

    chosen = people[top.track_no - 1] if conf != "low" else None
    # Number people: the chosen one is 1, the rest keep their order after it.
    order = ([chosen] if chosen else []) + [p for p in people if p is not chosen]
    for n, p in enumerate(order, start=1):
        p.track_no = n
    for c in cands:
        c.track_no = people[c.track_no - 1].track_no
    cands.sort(key=lambda c: c.track_no)
    if chosen is None:
        reason = reason + " Reported as Not Observed."
    return Attribution(people, chosen, conf, reason, cands, period_known, splits, swaps)
