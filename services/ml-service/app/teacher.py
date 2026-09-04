"""The teacher's timeline: one detected class, followed through the lesson.

This module is what is left of a 2,750-line identity stack — an age model, a
ground-plane fit, an appearance merge, a tracklet DP and a vision-model vote —
once the detector started naming the teacher directly. Everything that stack
existed to infer ("which of these thirty bodies is the adult?") is now a class
id, so the questions left are the ones a tracker actually answers:

  WHICH BOX CONTINUES WHICH BODY when the model offers more than one, and
  IS SHE STILL HERE across a frame where it offered none.

Both are settled by continuity. Until 2026-09 that meant ONE chain: every
instant's boxes were treated as competing guesses about a single body, and the
most confident reachable one won. That is right for a room with one adult — the
model emits at most one teacher box per frame across 583 scored frames of the
held-out room — and wrong for a handover: a recording of period 3 that starts
with the period 2 teacher still in the room offers two boxes at 64% of instants
across the overlap, both 0.7-0.86, and the chain settled each one on confidence
alone. The result was a frame-by-frame blend of two people that every quality
signal scored high, because there was always *a* box.

So the chain became SEGMENTS. Two boxes at one instant that are not the same
body are two people, and each continues its own segment by nearest reachable
box — greedy assignment, one box per segment per instant, over the same motion
model the single chain used. Nothing here decides which segment is the lesson's
teacher; that is attribution (docs/teacher-attribution-plan.md, Phase 3). What
this module hands over is (a) the segments, with the noise filtered out, and
(b) an interim `primary` — the biggest one — which is exactly the old chain on
a one-adult lesson and is refused downstream on a two-adult one until Phase 3
can do better.

Three things the segments do NOT fix, stated so nobody assumes they do:

- An OCCLUDED crossing. On the real handover clip the two teachers crossed
  paths twice. Predicting each body's position from its velocity carries the
  walker through the first. The second happened while one teacher was briefly
  undetected behind the other, and the model then drew a partial box on the
  occluded one: a partial box's CENTRE sits higher than a full box's on the
  same standing person, so centre-distance matched each body to the other's
  box. Tracking the box top (the head) instead resolves exactly that crossing —
  and re-swaps the first, and splits the 37-minute baseline, because a
  two-frame false box then seeds a lane whose stale prediction later steals
  hers. Rejected on that evidence (tools/ab_tracker.py). What resolves an
  occluded crossing is appearance: attribution.resolve_swaps compares both
  lanes' windows on either side of the encounter and gives each lane back its
  own body. The full 45-minute recording had two such swaps (306 s and
  2381 s); both are undone there, and this module stays motion-only.

- A student the detector calls "teacher" for a few frames is a short segment
  and is dropped as noise, which is right. A student who stands still at the
  front for thirty seconds is a substantial segment, which is also right — a
  distinct body WAS tracked — and it is attribution's job to say it is not her.
- While she is briefly occluded, a student box within reach of her last
  position joins HER segment, because it is the only continuation on offer.
  The old chain did the same. Appearance can catch it; motion alone cannot.

There is still no ByteTrack/BoT-SORT here and no Re-ID encoder. With two or
three adults at 5 fps on a static camera, distance-gated greedy assignment is
sufficient, and the day a room shows five adults is the day a real tracker
earns its place.

Pure function of detections, so /analyze, /rederive and the offline harness all
produce identical timelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.models import CLASS_TEACHER, Detection

logger = logging.getLogger(__name__)

# The attributed teacher's identity number. Other substantial segments take
# 2, 3, ... in order of first appearance; noise segments take none.
TEACHER_TRACK_NO = 1

# --- plausible motion --------------------------------------------------------
# How far she can move, as a fraction of the frame per second. Measured from
# per-frame ground truth on both annotated lessons: her speed peaks at 0.55
# (demo) and 0.25 (khaitan) frame-fractions per second. 0.8 clears the faster
# room's maximum with room to spare while still rejecting a box that jumps
# across the room between two samples, which is never her.
MAX_SPEED_PER_S = 0.8
# Flat allowance on top, absorbing bbox jitter at short sampling intervals: at
# 5 fps the ground truth's own largest single-frame displacement is 0.17, and
# a gate that cannot admit that would reject her at her fastest.
JUMP_BASE = 0.05
# Past this gap her position carries no information at all — she may have left
# the room and come back through another door — so the motion gate switches
# off rather than insisting she reappear where she vanished. Matches the
# presence-gap semantics in heuristics.py: beyond this she was ABSENT, and a
# fresh start is correct.
FREE_GAP_MS = 5_000

# --- more than one adult -----------------------------------------------------
# Two teacher boxes at ONE instant is evidence of two people. It is not a tie to
# break: the model puts a correct box on the teacher in 95.5% of scored frames
# and emitted at most one per frame across 583 frames of the single-adult
# held-out room, so a second box is far more likely to be a second body than a
# duplicate of the first.
#
# It can still be a duplicate, or an adult crossing the doorway on their way
# past, so the flag is raised on DURATION rather than on any single contested
# frame. A passer-by is seconds; the measured handover ran ~90 s of overlap
# before the outgoing teacher left. 30 s sits an order of magnitude above the
# first and comfortably below the second.
CO_PRESENCE_MIN_MS = 30_000

# --- what makes a segment a person ------------------------------------------
# Two boxes at one instant where the smaller sits mostly INSIDE the larger are
# one body seen twice, and the more confident sighting stands for both.
#
# Containment, not IoU, and the 37-minute baseline is why: the detector often
# puts a half-height box on her upper body beside the full-body box — same x,
# same top edge, half the height. Their IoU is 0.41-0.45, so an IoU gate at 0.5
# called them two bodies, nearest-assignment gave each its own lane, and the
# phantom lane then reached across the room and collected a student. The
# smaller box is 100% inside the larger; two people side by side are not.
#
# 0.5, not 0.7, after the real handover clip: on a SEATED teacher the model
# emits a head-only box beside a torso box, and the head box hangs over the
# torso box's top edge, so its containment is 0.5-0.9 rather than 1.0. At 0.7
# she became two lanes for 25 seconds. Two adults in that room, even at the
# moment they pass each other, overlap by ~0 — so 0.5 costs nothing there.
SAME_BODY_OVERLAP = 0.5
# A segment shorter than this is a misdetection, not a person: the detector
# calling a student "teacher" for some frames. On the baseline the longest such
# run is 0.8 s; three seconds is fifteen frames at 5 fps, well past a flicker
# and well short of anyone who is actually in the room. A noise segment is
# never numbered, never surfaced as another adult, and never a candidate for
# the return-from-absence weld — any of those would let a flicker fracture or
# hijack her timeline.
SEGMENT_MIN_MS = 3_000
SEGMENT_MIN_DETS = 8

# --- predicting where a body is next -----------------------------------------
# Assignment ranks candidate boxes by distance from where each body is PREDICTED
# to be, extrapolating its last step, rather than from where it last was. The
# difference only matters when two bodies compete for boxes, and that is
# exactly when it matters most: on the real handover clip the two teachers
# crossed paths twice, and nearest-to-last-position swapped their segments both
# times — the one walking past inherited the one standing still, because at the
# crossing "where she was" is the other person's position. Predicting carries
# the walker through. A velocity is only trusted from two sightings at most
# this far apart, and is only extrapolated this far ahead, because per-frame
# box jitter turns a long lead into a random guess.
PREDICT_MAX_STEP_MS = 1_000
PREDICT_MAX_LEAD_MS = 600


@dataclass
class Segment:
    """One continuously tracked body: detections the motion model accepts as a
    single person, in time order, extended by nearest reachable box."""

    detections: list[Detection]
    track_no: Optional[int] = None

    @property
    def first_ms(self) -> int:
        return self.detections[0].video_ts_ms

    @property
    def last_ms(self) -> int:
        return self.detections[-1].video_ts_ms

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def span_ms(self) -> int:
        return self.last_ms - self.first_ms

    @property
    def mean_conf(self) -> float:
        return sum(d.conf for d in self.detections) / len(self.detections)

    @property
    def substantial(self) -> bool:
        """A person rather than a flicker. See SEGMENT_MIN_MS."""
        return len(self.detections) >= SEGMENT_MIN_DETS and self.span_ms >= SEGMENT_MIN_MS


@dataclass
class TeacherTrack:
    """Her accepted detections, in time order, plus why to trust them."""

    detections: list[Detection]
    coverage: float = 0.0
    mean_conf: float = 0.0
    rejected_jumps: int = 0
    notes: list[str] = field(default_factory=list)
    # How often the detector offered more than one teacher box at a single
    # instant, the most it ever offered at once, and roughly how long that
    # lasted. These describe the ROOM, not her: they are the evidence that
    # `detections` above may not all belong to the same person.
    contested_instants: int = 0
    max_simultaneous: int = 1
    co_presence_ms: int = 0
    # Every body the tracker followed, primary included, in order of first
    # appearance. `detections` above is the primary segment's list.
    segments: list[Segment] = field(default_factory=list)
    # Boxes discarded as a second sighting of the same body at one instant.
    duplicates: int = 0

    @property
    def found(self) -> bool:
        return bool(self.detections)

    def retarget(self, detections: list[Detection], sampled_frames: int) -> "TeacherTrack":
        """The same lesson-level signals (co-presence, notes, duplicates) with
        the timeline replaced by `detections` — attribution's chosen person,
        which may be several segments. Coverage and confidence are recomputed
        over the new timeline; rejected_jumps keeps the tracker's count."""
        dets = sorted(detections, key=lambda d: d.video_ts_ms)
        mean_conf = sum(d.conf for d in dets) / max(len(dets), 1)
        return TeacherTrack(
            detections=dets,
            coverage=round(min(1.0, len(dets) / max(sampled_frames, 1)), 4),
            mean_conf=round(mean_conf, 4),
            rejected_jumps=self.rejected_jumps,
            notes=list(self.notes),
            contested_instants=self.contested_instants,
            max_simultaneous=self.max_simultaneous,
            co_presence_ms=self.co_presence_ms,
            segments=self.segments,
            duplicates=self.duplicates,
        )

    @property
    def others(self) -> list[Segment]:
        """Substantial segments that are not the primary: the other adults the
        tracker followed, numbered 2.. and handed to attribution."""
        return [
            s for s in self.segments
            if s.track_no is not None and s.track_no != TEACHER_TRACK_NO
        ]

    @property
    def multiple_adults(self) -> bool:
        """Was a second adult in the room long enough to matter?

        Deliberately not "were two boxes ever offered": a single contested
        instant is a duplicate box or someone passing the doorway, and refusing
        a lesson's numbers over that would make the refusal worthless. Sustained
        co-presence is the thing that makes this timeline a blend.
        """
        return self.co_presence_ms >= CO_PRESENCE_MIN_MS

    @property
    def first_ms(self) -> int:
        return self.detections[0].video_ts_ms if self.detections else 0

    @property
    def last_ms(self) -> int:
        return self.detections[-1].video_ts_ms if self.detections else 0

    @property
    def confidence(self) -> Optional[float]:
        """0..1 trust in the teacher labelling, on the scale quality.py tiers.

        Built from the two things that can actually be wrong here — how much of
        the lesson she was found in, and how sure the detector was when it found
        her — rather than from a margin over a runner-up, because with a named
        class there is no runner-up to measure against.
        """
        if not self.detections:
            return None
        return round(min(1.0, 0.5 * self.coverage + 0.5 * self.mean_conf), 4)


def _center(d: Detection) -> tuple[float, float]:
    return d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0


def _plausible(prev: Detection, cand: Detection) -> bool:
    """Could one person be at `prev` and then at `cand`?

    Distance-based rather than IoU-based on purpose: at 5 fps a walking teacher
    genuinely produces boxes that do not overlap at all, so an IoU gate would
    cut her track every time she crossed the room.
    """
    gap_ms = cand.video_ts_ms - prev.video_ts_ms
    if gap_ms >= FREE_GAP_MS:
        return True
    px, py = _center(prev)
    cx, cy = _center(cand)
    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
    allowed = JUMP_BASE + MAX_SPEED_PER_S * max(0, gap_ms) / 1000.0
    return dist <= allowed


def _predicted(seg: Segment, ts: int) -> tuple[float, float]:
    """Where `seg` should be at `ts`: its last centre carried forward by the
    velocity of its last step, when that step is recent enough to trust."""
    last = seg.last
    lx, ly = _center(last)
    if len(seg.detections) < 2:
        return lx, ly
    prev = seg.detections[-2]
    dt = last.video_ts_ms - prev.video_ts_ms
    if dt <= 0 or dt > PREDICT_MAX_STEP_MS:
        return lx, ly
    px, py = _center(prev)
    lead = min(ts - last.video_ts_ms, PREDICT_MAX_LEAD_MS)
    return lx + (lx - px) / dt * lead, ly + (ly - py) / dt * lead


def _dist_from(point: tuple[float, float], d: Detection) -> float:
    cx, cy = _center(d)
    return ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** 0.5


def _recent_height(seg: Segment) -> float:
    """Median box height over the segment's last SIZE_WINDOW sightings."""
    hs = sorted(d.bbox["h"] for d in seg.detections[-SIZE_WINDOW:])
    return hs[len(hs) // 2]


def _size_cost(ref_h: float, d: Detection) -> float:
    h = d.bbox["h"]
    if ref_h <= 0 or h <= 0:
        return 0.0
    return SIZE_WEIGHT * abs(math.log(h / ref_h))


def _containment(a: Detection, b: Detection) -> float:
    """Share of the SMALLER box's area that lies inside the other.

    1.0 for a half-body box on a full-body box (the smaller is entirely
    inside); near 0 for two people side by side. IoU cannot make that
    distinction — it penalises the size difference that is the whole tell.
    """
    ax0, ay0 = a.bbox["x"], a.bbox["y"]
    ax1, ay1 = ax0 + a.bbox["w"], ay0 + a.bbox["h"]
    bx0, by0 = b.bbox["x"], b.bbox["y"]
    bx1, by1 = bx0 + b.bbox["w"], by0 + b.bbox["h"]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min(a.bbox["w"] * a.bbox["h"], b.bbox["w"] * b.bbox["h"])
    return (iw * ih) / smaller if smaller > 0 else 0.0


def _dedup_instant(boxes: list[Detection]) -> list[Detection]:
    """One box per body at this instant, most confident first.

    A second box sitting mostly inside a kept one (SAME_BODY_OVERLAP) is the
    same body seen twice, and the more confident sighting stands for it —
    which is also what the old single chain chose when both were reachable, so
    a one-adult lesson keeps exactly the boxes it had.
    """
    kept: list[Detection] = []
    for d in sorted(boxes, key=lambda d: d.conf, reverse=True):
        if all(_containment(d, k) < SAME_BODY_OVERLAP for k in kept):
            kept.append(d)
    return kept


def _weld_target(segments: list[Segment], ts: int) -> Optional[Segment]:
    """The person a box after a long gap belongs to, or None if that is unclear.

    Replaces the old chain's rule that any box past FREE_GAP_MS simply continued
    the chain wherever it reappeared. That rule is exactly right when there is
    one adult — she stepped out and came back — and it is how two adults were
    welded into one: the outgoing teacher's chain reached forward and claimed
    the incoming one's first box.

    So the weld happens only when it is UNAMBIGUOUS: there is a dormant person
    to return to, and no other person was around at the end of their absence.
    "Around" means any other substantial segment alive after, or within
    FREE_GAP_MS before, the dormant one's last sighting — including one alive
    right now. Noise segments do not count either way, or a three-frame
    student misdetection during her absence would split her timeline in two.
    When it IS ambiguous the box starts a new segment and the question of
    whether that is her returning goes to attribution, which has evidence
    (appearance, the timetable) that position after a five-second gap does not.
    """
    dormant = [
        s for s in segments if ts - s.last_ms >= FREE_GAP_MS and s.substantial
    ]
    if not dormant:
        return None
    target = max(dormant, key=lambda s: s.last_ms)
    for s in segments:
        if s is target or not s.substantial:
            continue
        if s.last_ms > target.last_ms - FREE_GAP_MS:
            return None
    return target


def _track(stamps: list[int], by_ts: dict[int, list[Detection]]) -> tuple[list[Segment], int]:
    """Follow every body through the lesson. Returns (segments, duplicates).

    Per instant: drop same-body duplicates; match boxes to ACTIVE segments (a
    sighting within FREE_GAP_MS) by nearest reachable box, one box per segment;
    any box left over either welds onto the one person who is unambiguously
    returning from an absence, or starts a new segment.

    Nearest to where each body is PREDICTED to be, not most confident.
    Confidence is a statement about whether this is a teacher at all, which the
    class id has already settled; it says nothing about WHICH teacher. Distance
    from where a body was heading does. Choosing on confidence is the identity
    swap this rewrite exists to end; choosing on last position instead of
    predicted position is the swap the real handover clip then showed at every
    crossing. Reachability (_plausible) is still judged from the last sighting,
    so a body is never refused a box it could have reached.
    """
    segments: list[Segment] = []
    duplicates = 0
    for ts in stamps:
        boxes = _dedup_instant(by_ts[ts])
        duplicates += len(by_ts[ts]) - len(boxes)

        active = [s for s in segments if ts - s.last_ms < FREE_GAP_MS]
        predicted = [_predicted(seg, ts) for seg in active]
        pairs = sorted(
            (_dist_from(predicted[si], box), si, bi)
            for si, seg in enumerate(active)
            for bi, box in enumerate(boxes)
            if _plausible(seg.last, box)
        )
        taken_s: set[int] = set()
        taken_b: set[int] = set()
        for _, si, bi in pairs:
            if si in taken_s or bi in taken_b:
                continue
            active[si].detections.append(boxes[bi])
            taken_s.add(si)
            taken_b.add(bi)

        for bi, box in enumerate(boxes):
            if bi in taken_b:
                continue
            target = _weld_target(segments, ts)
            if target is not None:
                target.detections.append(box)
            else:
                segments.append(Segment([box]))
    return segments, duplicates


def _sample_interval_ms(stamps: list[int]) -> int:
    """Median gap between consecutive sampled instants, or 0 if unknowable.

    The median rather than the mean because a lesson's stamps are not evenly
    spaced end to end — a stretch where nothing at all was detected leaves one
    enormous gap, and an average over that would inflate every duration built
    from it.
    """
    if len(stamps) < 2:
        return 0
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    return gaps[len(gaps) // 2]


def _co_presence(
    by_ts: dict[int, list[Detection]], sample_ms: int
) -> tuple[int, int, int]:
    """(contested instants, most boxes at once, ms with two or more in frame).

    Reported as a duration as well as a count because a count means nothing
    without the sample rate: "289 contested frames" is unreadable, and "58
    seconds with two adults in the room" is a fact somebody can act on.
    """
    contested = 0
    most = 0
    for boxes in by_ts.values():
        n = len(boxes)
        if n > most:
            most = n
        if n >= 2:
            contested += 1
    return contested, most, contested * sample_ms


def build_teacher_track(
    detections: list[Detection],
    duration_ms: int,
    conf_threshold: Optional[float] = None,
) -> TeacherTrack:
    """Follow every teacher-class body through the lesson; stamps track_no in place.

    Returns the PRIMARY segment's detections as the track — the biggest body,
    which on a one-adult lesson is her chain exactly — with every segment the
    tracker built on `.segments` and the other substantial ones on `.others`,
    numbered 2.. for attribution to choose between.

    track_no after this call: 1 on the primary's boxes, 2.. on other adults',
    None on noise segments and same-body duplicates. All teacher-class boxes
    are persisted whatever their number (migration 0014); None means
    "detected as a teacher, attributed to nobody".
    """
    threshold = (
        conf_threshold if conf_threshold is not None else get_settings().teacher_conf
    )
    teacher_dets = sorted(
        (d for d in detections if d.cls == CLASS_TEACHER and d.conf >= threshold),
        key=lambda d: d.video_ts_ms,
    )
    for d in detections:
        d.track_no = None

    if not teacher_dets:
        return TeacherTrack(
            detections=[],
            notes=["The detector never found a teacher in this lesson."],
        )

    by_ts: dict[int, list[Detection]] = {}
    for d in teacher_dets:
        by_ts.setdefault(d.video_ts_ms, []).append(d)
    stamps = sorted(by_ts)

    segments, duplicates = _track(stamps, by_ts)

    # INTERIM attribution: the biggest segment is "the teacher". On a one-adult
    # lesson that is her chain, unchanged. On a two-adult lesson it is a guess
    # that Phase 0 refuses downstream (co-presence -> Not Observed) until
    # attribution proper replaces this line. Ties broken by confidence, then
    # by appearing first, so a lone contested instant is decided the way the
    # old chain decided it.
    primary = max(
        segments,
        key=lambda s: (len(s.detections), s.mean_conf, -s.first_ms),
    )
    primary.track_no = TEACHER_TRACK_NO
    others = sorted(
        (s for s in segments if s is not primary and s.substantial),
        key=lambda s: s.first_ms,
    )
    for i, seg in enumerate(others, start=TEACHER_TRACK_NO + 1):
        seg.track_no = i
    for seg in segments:
        for d in seg.detections:
            d.track_no = seg.track_no  # None for noise: unattributed, still stored

    accepted = primary.detections
    # Instants that offered a teacher box her timeline could not reach. Same
    # meaning the old chain gave the number; a box that went to ANOTHER person
    # is not a rejection, it is a different person.
    covered = {d.video_ts_ms for d in accepted}
    rejected = sum(1 for ts in stamps if ts not in covered)

    # Over every instant the detector LOOKED at, not just the ones offering a
    # teacher box: the sampling interval is a property of the pass, not of how
    # often she happened to be found.
    #
    # KNOWN ASYMMETRY between the two callers. /analyze passes a fresh detector
    # pass, where the board is detected in ~every frame, so this median IS the
    # sample rate. /rederive replays stored rows, which are teacher-class only
    # by construction, so on a heavily occluded lesson its stamps are sparse and
    # the median reads high — inflating co_presence_ms and making a rederive
    # more willing to refuse than the analysis that preceded it. It takes a
    # lesson where she is found rarely AND contested often to matter, and it
    # errs toward refusing rather than toward a wrong number, so it is recorded
    # here rather than worked around. The real fix is to persist the pass's
    # sampled-frame count instead of re-deriving it from whatever survived.
    sample_ms = _sample_interval_ms(sorted({d.video_ts_ms for d in detections}))
    contested, max_simultaneous, co_presence_ms = _co_presence(by_ts, sample_ms)

    notes: list[str] = []
    if co_presence_ms >= CO_PRESENCE_MIN_MS:
        notes.append(
            f"Two or more adults were in the room together for about "
            f"{co_presence_ms / 1000:.0f}s (up to {max_simultaneous} at once); "
            "each was tracked as a separate person."
        )
    if rejected:
        notes.append(
            f"{rejected} teacher detection(s) were rejected as implausible jumps "
            "and are not part of her timeline."
        )

    # Coverage is over every instant the detector looked at, not just the ones
    # that offered a teacher box. Dividing by the latter would report a lesson
    # where she was found in a tenth of the frames as perfectly covered, which
    # is precisely the reassuring-but-wrong number the quality report exists to
    # avoid. Frames with no detection of ANY class are invisible here, but the
    # board is detected in ~all of them, so the denominator is the sampled
    # frame count in practice.
    span = duration_ms if duration_ms > 0 else (accepted[-1].video_ts_ms or 1)
    sampled = len({d.video_ts_ms for d in detections})
    coverage = len(accepted) / max(sampled, 1)
    mean_conf = sum(d.conf for d in accepted) / max(len(accepted), 1)

    track = TeacherTrack(
        detections=accepted,
        coverage=round(min(1.0, coverage), 4),
        mean_conf=round(mean_conf, 4),
        rejected_jumps=rejected,
        notes=notes,
        contested_instants=contested,
        max_simultaneous=max_simultaneous,
        co_presence_ms=co_presence_ms,
        segments=segments,
        duplicates=duplicates,
    )
    logger.info(
        "teacher track: %d detections %d-%dms of a %dms lesson "
        "(mean conf %.2f, %d rejected jumps); %d segment(s), %d other adult(s)",
        len(accepted),
        track.first_ms,
        track.last_ms,
        span,
        mean_conf,
        rejected,
        len(segments),
        len(track.others),
    )
    if track.multiple_adults:
        logger.warning(
            "video has %d contested instants (~%.0fs, up to %d teacher boxes at "
            "once): this track may blend more than one adult",
            contested,
            co_presence_ms / 1000.0,
            max_simultaneous,
        )
    return track
