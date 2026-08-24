"""The teacher's timeline: one detected class, followed through the lesson.

This module is what is left of a 2,750-line identity stack — an age model, a
ground-plane fit, an appearance merge, a tracklet DP and a vision-model vote —
once the detector started naming the teacher directly. Everything that stack
existed to infer ("which of these thirty bodies is the adult?") is now a class
id, so the only questions left are the two a tracker actually answers:

  WHICH BOX IS HERS when the model offers more than one, and
  IS SHE STILL HERE across a frame where it offered none.

Both are settled by continuity, and the measurements say that is enough:

  - At the configured threshold the model emits AT MOST ONE teacher box per
    frame (0 frames with two, over 583 scored frames of the held-out room), so
    the disambiguation path is a safety net rather than the common case.
  - Its longest unbroken miss on that lesson is ~5.4 s, which a gap bridge
    covers; there is no re-identification problem to solve because there is no
    competing identity to confuse her with.

That is why there is no ByteTrack/BoT-SORT here, and no Re-ID encoder. Adding
either would be adding machinery to fix a problem the detector no longer has.
If a future room DOES show identity switching, the place to add it is
_pick_candidate, which is already the single point where competing boxes meet.

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

# The teacher is one person, so she gets one identity number.
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


@dataclass
class TeacherTrack:
    """Her accepted detections, in time order, plus why to trust them."""

    detections: list[Detection]
    coverage: float = 0.0
    mean_conf: float = 0.0
    rejected_jumps: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.detections)

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


def _pick_candidate(
    candidates: list[Detection], prev: Optional[Detection]
) -> Optional[Detection]:
    """Which of this frame's teacher boxes continues the track.

    Continuity FIRST, confidence second. Taking the highest-confidence box every
    frame is exactly the identity-switch bug this ordering exists to prevent: a
    momentarily more confident box on somebody else would steal the track and
    then keep it. A box that cannot be reached from the previous one is not her,
    however confident the model is about it.
    """
    if not candidates:
        return None
    if prev is None:
        return max(candidates, key=lambda d: d.conf)
    reachable = [d for d in candidates if _plausible(prev, d)]
    if not reachable:
        return None
    return max(reachable, key=lambda d: d.conf)


def _seed_index(stamps: list[int], by_ts: dict[int, list[Detection]]) -> int:
    """Index of the instant to start the chain from: the first UNCONTESTED one.

    Seeding on the first frame and breaking its tie by confidence looks
    reasonable and is a trap: a single spurious box that happens to outscore her
    on frame one captures the chain, and every real detection afterwards is then
    rejected as an implausible jump back. The whole lesson follows the wrong
    body from a one-frame accident.

    An instant offering exactly one teacher box has no such ambiguity, so the
    chain starts there and grows in both directions. On real footage this is
    almost always the very first instant — the detector emits at most one
    teacher box per frame at the configured threshold — so this costs nothing
    in the normal case and only matters in the one it exists for.
    """
    for i, ts in enumerate(stamps):
        if len(by_ts[ts]) == 1:
            return i
    return 0  # every instant contested: fall back to the first


def _chain(
    stamps: list[int], by_ts: dict[int, list[Detection]], seed: int
) -> tuple[list[Detection], int]:
    """Grow the track outwards from `seed`. Returns (accepted, rejected count)."""
    first = _pick_candidate(by_ts[stamps[seed]], None)
    if first is None:  # unreachable: every instant carries at least one box
        return [], 0
    accepted = [first]
    rejected = 0

    prev = first
    for ts in stamps[seed + 1 :]:
        chosen = _pick_candidate(by_ts[ts], prev)
        if chosen is None:
            rejected += 1
            continue
        accepted.append(chosen)
        prev = chosen

    # Backwards from the seed. _plausible is symmetric in distance but reads the
    # gap from its arguments' order, so the earlier detection stays first.
    prev = first
    before: list[Detection] = []
    for ts in reversed(stamps[:seed]):
        candidates = by_ts[ts]
        reachable = [d for d in candidates if _plausible(d, prev)]
        if not reachable:
            rejected += 1
            continue
        chosen = max(reachable, key=lambda d: d.conf)
        before.append(chosen)
        prev = chosen

    before.reverse()
    return before + accepted, rejected


def build_teacher_track(
    detections: list[Detection],
    duration_ms: int,
    conf_threshold: Optional[float] = None,
) -> TeacherTrack:
    """Follow the teacher class through the lesson; stamps track_no in place.

    Returns her accepted detections in time order. Detections rejected as
    implausible jumps keep their original class but are left without a
    track_no, so nothing downstream mistakes them for her.
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

    # Group by sampled instant: several boxes at one timestamp are competing
    # claims on the same body, and exactly one of them can be her.
    by_ts: dict[int, list[Detection]] = {}
    for d in teacher_dets:
        by_ts.setdefault(d.video_ts_ms, []).append(d)

    stamps = sorted(by_ts)
    accepted, rejected = _chain(stamps, by_ts, _seed_index(stamps, by_ts))
    for d in accepted:
        d.track_no = TEACHER_TRACK_NO

    notes: list[str] = []
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
    )
    logger.info(
        "teacher track: %d detections %d-%dms of a %dms lesson "
        "(mean conf %.2f, %d rejected jumps)",
        len(accepted),
        track.first_ms,
        track.last_ms,
        span,
        mean_conf,
        rejected,
    )
    return track
