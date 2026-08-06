"""Who is the teacher, and when — solved once, globally, over the whole lesson.

The old pipeline asked the question twice in the wrong order: merge fragments
into identities by appearance/geometry, then pick the most teacher-LOOKING
identity, then try to repair that identity by chaining forward from a seed.
On real classroom footage every stage of that fails in the same way, and the
failures compound:

  measured on an 11.5-minute lesson (28 people, 208 raw tracker ids), the
  teacher was tracked almost perfectly by three long raw ids — 0-208s,
  265-632s, 632-692s — yet the merge scattered them across three different
  merged identities, those identities then COMPETED with each other for the
  teacher slot, none could clear the relative-margin rule, and the video
  finished with no teacher at all: every KPI zero.

The fix is to stop deriving the teacher from generic identities and to solve
her timeline directly, as one global assignment:

1. TRACKLETS, not raw ids. A raw id is not a person: it is a person until the
   tracker hands it to someone else. Raw ids are split wherever the body they
   describe changes size in a sustained way (the classic classroom steal — the
   teacher crouches at a desk, her box collapses onto a pupil's, and the id
   walks away on the pupil, or vice versa).

2. AGE, not behaviour, as the anchor (app/adult.py). Behaviour is what breaks:
   in the first minutes of a lesson the whole class is standing, so "stands a
   lot" says nothing; a pupil sent to the board walks and stands; a teacher
   sitting with a group stops looking like a teacher. She is, however, the
   only adult in the room for the entire hour, in every frame she appears.

3. ONE GLOBAL CHOICE over her whole timeline, by dynamic programming over
   time-disjoint tracklets, instead of a greedy chain walking forward from a
   seed. This is what handles the cases a chain cannot:
     - she is on camera from the first frame (a forward chain seeded on her
       longest fragment can only reach backwards through a 30 s horizon),
     - she leaves the room and returns minutes later at a different door
       (no spatial continuity to chain through, and any gap length is fine),
     - she is occluded for a stretch of a crowded room (a gap costs nothing;
       the DP simply resumes),
     - two candidates are alive at once (impossible for one person, and the
       disjointness constraint rules it out by construction rather than by
       a tolerance).

4. ITERATED refinement. The first pass scores tracklets on age and behaviour
   alone; the winning set then defines a teacher appearance prototype and a
   teacher-specific height model, and the pass is repeated. Two or three
   rounds converge and give her own appearance a vote in what counts as her,
   without ever letting appearance bootstrap itself from nothing.

Everything here is a pure function of detections plus optional appearance, so
/analyze, /rederive and the offline harness produce identical timelines.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app import adult as adult_mod
from app.geometry import expand_bbox, polygon_bbox
from app.models import Detection

# --- tracklet splitting -----------------------------------------------------
# A raw id that goes quiet for this long and comes back has, more often than
# not, come back on somebody else.
SPLIT_GAP_MS = 8_000
# Sustained change in perspective-normalized height that marks a handoff. 0.28
# is well beyond posture noise (a walking body breathes +/-0.06) and below the
# ~0.45 step of an adult-to-child swap.
SPLIT_STATURE_DELTA = 0.28
# Samples that must agree on each side of a split before it is believed. At
# 5 fps this is 1.2 s of consistent evidence, which a crouch (transient) does
# not produce but a steal (permanent) does.
SPLIT_SUSTAIN = 6
# ...and the change must arrive as a STEP: this fraction of it has to happen
# between two consecutive samples. Walking towards the camera produces the same
# total change spread over seconds, and splitting on that is actively harmful.
SPLIT_STEP_FRACTION = 0.5
# ...and the perspective model has to be worth trusting before a height change
# is read as a change of person at all (see adult.GroundPlane.confidence).
SPLIT_MIN_PLANE_CONFIDENCE = 0.4
# Never cut a tracklet shorter than this; below it there is nothing to score.
MIN_TRACKLET_DETS = 3
# Appearance split: two halves of one raw id whose mean CLIP views disagree
# this much are two people. Deliberately far below the same-person band
# (measured p50 0.82 / p90 0.88 BETWEEN different people on real footage,
# while two views of one person sit well above 0.9), so only a real change of
# clothing triggers it.
APPEARANCE_SPLIT_COS = 0.86
APPEARANCE_SPLIT_MIN_SIDE = 2

# --- seeding ----------------------------------------------------------------
# The bootstrap only trusts LONG tracklets, because the signals that identify a
# teacher without knowing what she looks like — she walks the room, she is on
# her feet — are properties of a stretch of time, not of a moment. A
# four-second fragment of a seated pupil has a mobility of zero and a standing
# ratio of one and means nothing by either.
# A tracklet this long is fully trusted as a seed; shorter ones are trusted
# proportionally rather than excluded. A hard bar cannot work: in a packed room
# the teacher's longest unbroken stretch may be 25 seconds, and demanding 45
# would mean refusing to look for her precisely when she is hardest to find.
SEED_FULL_SPAN_MS = 45_000
SEED_MIN_SPAN_MS = 6_000
SEED_MIN_DETS = 15
SEED_W_MOBILITY = 0.40
SEED_W_STANDING = 0.25
SEED_W_ADULT = 0.35

# --- scoring ----------------------------------------------------------------
# Once there IS a seed, matching her to herself carries the most weight: the
# question for every other fragment is "is this the same person", and that is
# what appearance answers. Age keeps the answer honest when appearance is
# ambiguous (uniform crowds compress CLIP similarity), and behaviour breaks
# remaining ties.
W_APPEARANCE = 0.40
W_ADULT = 0.25
W_MOBILITY = 0.20
W_STANDING = 0.15
# Standing AT the board is the one piece of behaviour that means "teaching"
# rather than merely "moving", and it is what keeps a teacher who lectures
# from one spot — no mobility to speak of — identifiable at all.
#
# It is applied as a BONUS towards 1.0 rather than as another weighted term:
# as a weight it diluted the appearance evidence everywhere in the room to buy
# one case, and cost 4 points of coverage on real footage. Corroboration
# should be able to rescue a candidate without being able to punish one.
# Swept against real footage: up to 0.30 leaves the assignment untouched,
# while 0.35 hands the lesson to a pupil who stood in the board's half of the
# room for three minutes. 0.30 is what a teacher who lectures from one spot
# needs to clear the claim bar on geometry alone. Corroboration, not authority.
BOARD_BONUS = 0.30
# How far below the board's bottom edge her feet may be while still counting
# as standing AT it, in frame heights. Wide enough for the depth of the strip
# of floor in front of a board on this camera, narrow enough to exclude the
# first row of desks.
BOARD_FOOT_BAND = 0.25
# Mobility saturates here (fraction of the frame the tracklet's centre covers).
MOBILITY_NORM = 0.25
# A tracklet must beat this to be worth claiming at all, so that a room with no
# adult in it yields no teacher rather than a confident wrong one. Swept
# against real footage AND the scenario suite: below ~0.52 an unsupervised
# class elects its most distinctive pupil, above ~0.56 a teacher whose track is
# shredded by occlusion falls under the bar. 0.54 is the widest point where
# both hold.
CLAIM_THRESHOLD = 0.54
# Co-presence tolerance: one person cannot hold two boxes, but a handoff frame
# or two of overlap is tracker noise, not a second body.
OVERLAP_TOLERANCE_MS = 1_500
# When a raw id is split because it changed person, WHERE the change happened
# is only known to within the spacing of the evidence that revealed it. Two
# consecutive claims may therefore overlap by up to this much (and up to this
# share of the shorter claim), with the earlier claim trimmed back to the start
# of the later one. Without it the disjointness rule turns an uncertain handoff
# instant into an either/or, and half the teacher's timeline is discarded to
# satisfy an artefact of the split granularity.
MAX_CLAIM_OVERLAP_MS = 15_000
MAX_CLAIM_OVERLAP_FRACTION = 0.25

# --- transition plausibility ------------------------------------------------
# How far she may have moved between the end of one claimed tracklet and the
# start of the next, before the jump is charged as evidence against.
TRANSIT_BASE = 0.12
TRANSIT_PER_S = 0.02
TRANSIT_MAX = 0.90
# Beyond this a jump is free: she has been gone long enough (or left through a
# door) that her position carries no information at all.
TRANSIT_FREE_GAP_MS = 20_000
# Weight of the transition term relative to a tracklet's own value.
W_TRANSIT = 0.35

# --- appearance -------------------------------------------------------------
# CLIP cosines on CCTV crops sit in a narrow band (p50 0.82, p90 0.88 measured
# between DIFFERENT people on real footage), so appearance is used as a RANK
# within the video, never as an absolute threshold.
APPEARANCE_TOP_K = 3
# Robust standard deviations above the population's median affinity that count
# as a full appearance match. Her own fragments clear this comfortably; a room
# of strangers produces a flat field where nobody does.
AFFINITY_Z_FULL = 1.5
# Two modalities, because they fail on different frames. CLIP carries pose and
# build and survives a change of lighting; the torso colour histogram carries
# what she is wearing and is far sharper when it works — measured against her
# own prototype, colour scored her early fragments 0.71-0.91 while the pupil
# who stood at the door for three minutes scored 0.09. Each is ranked
# separately and averaged, so either can vouch for a fragment the other misses
# (colour recognised her first three minutes; CLIP recognised her last minute,
# where the light had changed).
W_AFFINITY_CLIP = 0.5
W_AFFINITY_HIST = 0.5
APPEARANCE_SHRINKAGE = 0.25
# The prototype is rebuilt each round from the STRONGEST claims only. Letting
# every claim vote drags the prototype towards whatever was wrongly claimed in
# the previous round, and the error then reinforces itself — the standard
# failure of iterating an assignment against its own output.
PROTOTYPE_TOP_K = 5
# A claim only joins the prototype if it genuinely resembles the SEED — by
# this many robust sigmas above the field. Admitting claims by score alone let
# one wrongly claimed pupil into the reference; his classmates then scored
# well against the polluted prototype and the search walked away from the
# teacher, turning a correct first round into a wrong fourth one.
PROTOTYPE_ADMIT_Z = 1.0
# Refinement rounds. Convergence is detected by the selection repeating, so
# this is only a ceiling.
REFINE_ROUNDS = 4


# raw track id -> (sample timestamps or None, unit vectors)
Gallery = dict[int, tuple[Optional[np.ndarray], np.ndarray]]


def _center(d: Detection) -> tuple[float, float]:
    return d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0


@dataclass
class Tracklet:
    """A maximal stretch of one raw tracker id that describes ONE body."""

    tid: int
    raw_id: int
    dets: list[Detection]
    # False only when this piece came from cutting a raw id whose APPEARANCE
    # changed, i.e. the id was worn by two differently-dressed people: its
    # torso histogram is one median with no timestamps and belongs to neither
    # half. Cuts for a time gap or a change of posture do not void it — the
    # tracker held the same person across those, so the colour still describes
    # them, and treating every cut as disqualifying threw away the sharpest
    # evidence in the room (on real footage it cost 12 points of coverage,
    # because the pupils who compete with her are exactly the ones whose ids
    # get split when they sit down).
    hist_attributable: bool = True
    ts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ts = [d.video_ts_ms for d in self.dets]

    @property
    def first_ms(self) -> int:
        return self.ts[0]

    @property
    def last_ms(self) -> int:
        return self.ts[-1]

    @property
    def span_ms(self) -> int:
        return self.ts[-1] - self.ts[0]

    def head_center(self, n: int = 3) -> tuple[float, float]:
        pts = [_center(d) for d in self.dets[:n]]
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    def tail_center(self, n: int = 3) -> tuple[float, float]:
        pts = [_center(d) for d in self.dets[-n:]]
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    def spread(self) -> float:
        pts = [_center(d) for d in self.dets]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def standing_ratio(self) -> float:
        return sum(1 for d in self.dets if d.standing) / len(self.dets)

    def mean_occlusion(self) -> float:
        return float(np.mean([d.occlusion for d in self.dets]))


# --------------------------------------------------------------------------- #
# 1. Tracklets
# --------------------------------------------------------------------------- #


def _sustained_levels(ratios: list[float], sustain: int, delta: float) -> list[int]:
    """Indices where the height level STEPS, abruptly, and stays stepped.

    Compares the median of the `sustain` samples before an index with the
    median of the `sustain` samples after it. A crouch dips and recovers, so
    both windows agree; a handoff to another body changes the level for good.

    The abruptness test is what makes this safe. A person walking towards the
    camera also changes apparent height by a lot — smoothly, over seconds —
    and an earlier version without this test shredded a teacher's 136-second
    track into nineteen pieces on a second lesson, destroying the very
    behaviour evidence the search bootstraps from. A tracker handing a box
    from one body to another does it BETWEEN two frames, so a genuine handoff
    puts most of the level change into a single step.
    """
    cuts: list[int] = []
    n = len(ratios)
    if n < 2 * sustain:
        return cuts
    i = sustain
    while i <= n - sustain:
        before = float(np.median(ratios[i - sustain : i]))
        after = float(np.median(ratios[i : i + sustain]))
        change = abs(after - before)
        if change >= delta:
            # The largest single-sample jump anywhere near the boundary must
            # account for most of the change; a gradual walk cannot do that.
            window = ratios[max(0, i - 2) : min(n, i + 3)]
            step = max(
                (abs(b - a) for a, b in zip(window, window[1:])), default=0.0
            )
            if step >= SPLIT_STEP_FRACTION * change:
                cuts.append(i)
                i += sustain  # one cut per transition, not one per sample
                continue
        i += 1
    return cuts


def _appearance_change_ms(entry: tuple[Optional[np.ndarray], np.ndarray]) -> Optional[int]:
    """When, if ever, this raw id's appearance changed to a different person.

    A single change point over the track's own timestamped crops: split the
    gallery at every position, and take the split whose two halves disagree
    most. The dip has to be big in absolute terms — CLIP cosines between two
    views of ONE person on this footage stay high, while the white-shirted
    teacher against the blue-shirted pupil her id was handed to does not.

    This is the cut that stature cannot make: the observed handoff was between
    two people of almost identical height, and only their clothes gave it away.
    """
    ts, mat = entry
    if ts is None or len(mat) < 2 * APPEARANCE_SPLIT_MIN_SIDE:
        return None
    best: Optional[tuple[float, int]] = None
    for i in range(APPEARANCE_SPLIT_MIN_SIDE, len(mat) - APPEARANCE_SPLIT_MIN_SIDE + 1):
        a = mat[:i].mean(axis=0)
        b = mat[i:].mean(axis=0)
        a /= max(float(np.linalg.norm(a)), 1e-9)
        b /= max(float(np.linalg.norm(b)), 1e-9)
        cos = float(np.dot(a, b))
        if best is None or cos < best[0]:
            best = (cos, i)
    if best is None or best[0] > APPEARANCE_SPLIT_COS:
        return None
    i = best[1]
    return int((ts[i - 1] + ts[i]) // 2)


def build_tracklets(
    dets_by_raw: dict[int, list[Detection]],
    plane: Optional[adult_mod.GroundPlane] = None,
    galleries: Optional[Gallery] = None,
) -> list[Tracklet]:
    """Split raw tracker ids wherever they stop describing one body.

    Three cuts — a long absence, a sustained change of size, a change of
    clothes — all conservative in the direction that matters: over-splitting
    costs the assignment nothing (the DP re-joins adjacent pieces of the same
    person at zero transition cost) while under-splitting welds a pupil to the
    teacher permanently.
    """
    plane = plane or adult_mod.fit_ground_plane(dets_by_raw)
    galleries = galleries or {}
    out: list[Tracklet] = []
    tid = 0
    for raw_id in sorted(dets_by_raw):
        dets = sorted(dets_by_raw[raw_id], key=lambda d: d.video_ts_ms)
        if not dets:
            continue
        # Cut 1: the id went away and came back.
        pieces: list[list[Detection]] = [[dets[0]]]
        for prev, cur in zip(dets, dets[1:]):
            if cur.video_ts_ms - prev.video_ts_ms >= SPLIT_GAP_MS:
                pieces.append([])
            pieces[-1].append(cur)

        # Cut 3: the person wearing this id changed clothes, i.e. changed.
        change_ms = (
            _appearance_change_ms(galleries[raw_id]) if raw_id in galleries else None
        )

        for piece in pieces:
            cuts: set[int] = set()
            wore_two_outfits = False
            # Cut 2: the body it describes changed size and stayed changed.
            # Only where the perspective model is good enough to make "size"
            # mean something. With a flat or unfitted plane, apparent height
            # tracks how far down the room somebody walked, and cutting on it
            # shreds the one person who walks — the teacher.
            if (
                plane.ok
                and plane.confidence >= SPLIT_MIN_PLANE_CONFIDENCE
                and len(piece) >= 2 * SPLIT_SUSTAIN
            ):
                ratios = [
                    d.bbox["h"] / plane.predict(d.bbox["y"] + d.bbox["h"])
                    for d in piece
                ]
                cuts.update(_sustained_levels(ratios, SPLIT_SUSTAIN, SPLIT_STATURE_DELTA))
            wore_two_outfits = (
                change_ms is not None
                and piece[0].video_ts_ms < change_ms < piece[-1].video_ts_ms
            )
            if wore_two_outfits:
                cuts.add(bisect_left([d.video_ts_ms for d in piece], change_ms))
            bounds = [0, *sorted(cuts), len(piece)]
            for lo, hi in zip(bounds, bounds[1:]):
                chunk = piece[lo:hi]
                if len(chunk) >= MIN_TRACKLET_DETS:
                    out.append(
                        Tracklet(
                            tid=tid,
                            raw_id=raw_id,
                            dets=chunk,
                            hist_attributable=not wore_two_outfits,
                        )
                    )
                    tid += 1
    out.sort(key=lambda t: (t.first_ms, t.last_ms))
    return out


# --------------------------------------------------------------------------- #
# 2. Appearance
# --------------------------------------------------------------------------- #


def normalize_galleries(raw: Optional[dict]) -> Gallery:
    """Accept either timestamped or bare appearance samples, per raw track id.

    /analyze hands over (ts, vector) samples so a split tracklet can keep only
    the crops taken while it was that body. /rederive and older fixtures only
    have the track's median vector with no timestamp; those apply to the whole
    raw track, which is the best that evidence can support.
    """
    out: Gallery = {}
    for raw_id, samples in (raw or {}).items():
        if samples is None or len(samples) == 0:
            continue
        stamps: list[int] = []
        vecs: list[list[float]] = []
        if np.isscalar(samples[0]):  # a single bare vector (the stored median)
            samples = [samples]
        for s in samples:
            if (
                isinstance(s, (tuple, list))
                and len(s) == 2
                and np.isscalar(s[0])
                and not np.isscalar(s[1])
            ):
                stamps.append(int(s[0]))
                vecs.append([float(v) for v in s[1]])
            else:
                vecs.append([float(v) for v in s])
        mat = np.asarray(vecs, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-9)
        ts = np.asarray(stamps, dtype=np.int64) if len(stamps) == len(vecs) else None
        out[int(raw_id)] = (ts, mat)
    return out


def _gallery_for(tracklet: Tracklet, galleries: Gallery) -> np.ndarray:
    """Unit vectors sampled from this tracklet's own time range."""
    entry = galleries.get(tracklet.raw_id)
    if entry is None:
        return np.zeros((0, 0))
    ts, mat = entry
    if ts is None:
        return mat
    keep = (ts >= tracklet.first_ms) & (ts <= tracklet.last_ms)
    if not keep.any():
        # A short tracklet may contain no sample instant of its own; fall back
        # to the parent's nearest sample so it is not left blind.
        mid = (tracklet.first_ms + tracklet.last_ms) // 2
        return mat[int(np.argmin(np.abs(ts - mid))) : int(np.argmin(np.abs(ts - mid))) + 1]
    return mat[keep]


def _affinity(gallery: np.ndarray, prototype: np.ndarray) -> Optional[float]:
    """Best-of-gallery cosine against the teacher prototype's samples.

    Max over pairs rather than mean-to-mean: she is half-occluded, back-turned
    or blown out by the doorway in most of her crops, and the question that
    matters for re-identification is whether her BEST view of one stretch
    matches her BEST view of another.
    """
    if gallery.size == 0 or prototype.size == 0:
        return None
    sims = gallery @ prototype.T
    k = min(APPEARANCE_TOP_K, sims.size)
    return float(np.mean(np.sort(sims.ravel())[-k:]))


def _fuse_appearance(
    tracklets: list[Tracklet],
    clip_rank: dict[int, float],
    hist_rank: dict[int, float],
) -> dict[int, float]:
    """Combine the two modalities into one 0..1 appearance score per tracklet.

    Averaging two ranks compresses the result towards the middle — a tracklet
    ranked top on colour and mid on CLIP lands at 0.7 and is no longer
    distinguishable from one that is mediocre at both. So the AVERAGE IS
    RANKED AGAIN, which restores the full spread, and only then scaled by how
    far the field actually separates (see _rank_normalize): a room where
    nobody resembles the prototype produces a flat combined field, which
    collapses to zero rather than crowning its least-bad member.
    """
    combined: dict[int, float] = {}
    for t in tracklets:
        vals = [
            (W_AFFINITY_CLIP, clip_rank.get(t.tid)),
            (W_AFFINITY_HIST, hist_rank.get(t.tid)),
        ]
        present = [(w, v) for w, v in vals if v is not None]
        if not present:
            continue
        weight = sum(w for w, _v in present)
        # Shrunk towards "no opinion" by how much evidence is missing, so a
        # fragment vouched for by BOTH colour and CLIP outranks one that only
        # CLIP likes — which matters because CLIP barely separates people in a
        # uniformed room (measured p50 0.82 / p90 0.88 between strangers) while
        # colour separates them decisively.
        combined[t.tid] = (
            sum(w * v for w, v in present) + APPEARANCE_SHRINKAGE * 0.5
        ) / (weight + APPEARANCE_SHRINKAGE)
    return _rank_normalize(combined)


def _outliers(values: dict[int, float], z: float) -> set[int]:
    """Keys whose value stands out from the field by at least `z` robust sigmas."""
    if len(values) < 4:
        return set(values)
    arr = np.array(list(values.values()), dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-9:
        return set()
    return {k for k, v in values.items() if (v - med) / mad >= z}


def _seed_affinity(
    t: Tracklet,
    seed: Tracklet,
    galleries: Gallery,
    hists: dict[int, np.ndarray],
    attributable: set[int],
) -> float:
    """How much this tracklet looks like the seed, over both modalities."""
    vals = [_affinity(_gallery_for(t, galleries), _gallery_for(seed, galleries))]
    if t.tid in attributable and seed.tid in attributable:
        vals.append(_hist_affinity(_hist_for(t, hists), _hist_for(seed, hists)))
    present = [v for v in vals if v is not None]
    return float(np.mean(present)) if present else 0.0


def _affinities(
    tracklets: list[Tracklet],
    reference: list[Tracklet],
    galleries: Gallery,
    hists: dict[int, np.ndarray],
    kind: str,
    attributable: Optional[set[int]] = None,
) -> dict[int, float]:
    """Raw affinity of every tracklet to the reference set, in one modality.

    A tracklet that is itself in the reference set is compared against the
    reference WITHOUT its own contribution, or it would simply confirm itself
    and the refinement would never be able to drop a bad claim.
    """
    ref_tids = {t.tid for t in reference}
    out: dict[int, float] = {}
    for t in tracklets:
        others = [p for p in reference if p.tid != t.tid] if t.tid in ref_tids else reference
        # Never compared against itself: a self-match is a perfect score for
        # free, and in a room with no teacher that self-confirmation alone
        # elects whichever pupil happened to seed the search. A tracklet with
        # no comparison available simply has no appearance term.
        if not others:
            continue
        if kind == "clip":
            a = _affinity(_gallery_for(t, galleries), _prototype(others, galleries))
        elif attributable is not None and t.tid not in attributable:
            # This tracklet's raw id was split because it described more than
            # one body, and its histogram is a single median over the whole id
            # with no timestamps — there is no way to know which body it came
            # from. Evidence that cannot be attributed is not used; the CLIP
            # gallery, which IS timestamped, carries these on its own.
            continue
        else:
            a = _hist_affinity(_hist_for(t, hists), _hist_prototype(others, hists))
        if a is not None:
            out[t.tid] = a
    return out


def _rank_normalize(values: dict[int, float]) -> dict[int, float]:
    """Map raw affinities onto 0..1 by rank AND by how much they stand out.

    Rank alone is a trap here: it always awards 1.0 to somebody, so in a room
    where NOBODY resembles the prototype the least-dissimilar stranger scores a
    perfect appearance match. Scaling the rank by a robust z-score of the same
    affinities means "best of a uniformly bad field" collapses to nearly zero
    and the decision falls back to age and behaviour — which is the correct
    answer for a room with no teacher in it.
    """
    if not values:
        return {}
    if len(values) == 1:
        return {k: 1.0 for k in values}
    rank = adult_mod._ranks(values)
    arr = np.array(list(values.values()), dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-9:
        return {k: 0.0 for k in values}
    return {
        k: rank[k] * float(min(1.0, max(0.0, (values[k] - med) / mad / AFFINITY_Z_FULL)))
        for k in values
    }


# --------------------------------------------------------------------------- #
# 3. Scoring + global selection
# --------------------------------------------------------------------------- #


def _board_proximity(dets: list[Detection], board: Optional[list[list[float]]]) -> float:
    """Fraction of detections that are STANDING AT the board.

    Three conditions, and the third is the one that matters: the feet have to
    be just below the board, not merely below it. A wall-mounted board is above
    everybody's feet, so "bottom edge below the board" is true of the entire
    room — a test that loose handed the bonus to half the class and cost more
    than it bought. Requiring the feet inside a band under the board is what
    actually means "at the board" rather than "somewhere in front of it".
    """
    if not board:
        return 0.0
    x0, _y0, x1, y1 = polygon_bbox(board)
    hits = 0
    for d in dets:
        cx = d.bbox["x"] + d.bbox["w"] / 2.0
        foot = d.bbox["y"] + d.bbox["h"]
        if d.standing and x0 <= cx <= x1 and y1 <= foot <= y1 + BOARD_FOOT_BAND:
            hits += 1
    return hits / len(dets)


def seed_scores(
    tracklets: list[Tracklet],
    adult_scores: dict[int, float],
    duration_ms: int = 0,
) -> dict[int, float]:
    """Teacher-likeness of LONG tracklets only, used to bootstrap the search.

    Nothing here needs to know what she looks like, which is the point: the
    seed has to be findable before any appearance prototype exists. Short
    tracklets are excluded rather than scored badly — their behaviour readings
    are undefined, not low — with the length bar scaled down for short clips
    so a two-minute excerpt is still analysable.
    """
    out: dict[int, float] = {}
    for t in tracklets:
        if t.span_ms < SEED_MIN_SPAN_MS or len(t.dets) < SEED_MIN_DETS:
            continue
        behaviour = (
            SEED_W_MOBILITY * min(1.0, t.spread() / MOBILITY_NORM)
            + SEED_W_STANDING * t.standing_ratio()
            + SEED_W_ADULT * adult_scores.get(t.tid, 0.0)
        )
        # Scaled by how much time stands behind the reading, so a long
        # confident stretch outranks a lucky four-second fragment without
        # short stretches being ruled out altogether.
        out[t.tid] = behaviour * min(1.0, t.span_ms / SEED_FULL_SPAN_MS)
    return out


def score_tracklets(
    tracklets: list[Tracklet],
    adult_scores: dict[int, float],
    appearance: Optional[dict[int, float]] = None,
    board_polygon: Optional[list[list[float]]] = None,
) -> dict[int, float]:
    """Per-tracklet 0..1 teacher likelihood.

    Appearance affinity to the current prototype leads, age keeps it honest,
    behaviour breaks ties. A tracklet with no appearance evidence at all is
    scored on the remaining signals rather than being punished for it — it can
    still be claimed on age and behaviour, and its short duration bounds how
    much it can win or lose.

    `appearance` is expected ALREADY normalized to 0..1 by the caller (see
    _rank_normalize). Re-normalizing it here silently destroyed the signal
    whenever her fragments agreed with each other: identical affinities have
    zero spread, and a second pass mapped the whole tied group to zero.
    """
    mobility_raw = {t.tid: min(1.0, t.spread() / MOBILITY_NORM) for t in tracklets}
    app_rank = appearance or {}

    out: dict[int, float] = {}
    for t in tracklets:
        parts = [
            (W_ADULT, adult_scores.get(t.tid, 0.0)),
            (W_MOBILITY, mobility_raw[t.tid]),
            (W_STANDING, t.standing_ratio()),
        ]
        if t.tid in app_rank:
            parts.append((W_APPEARANCE, app_rank[t.tid]))
        total = sum(w for w, _v in parts)
        score = sum(w * v for w, v in parts) / total
        if board_polygon is not None:
            at_board = _board_proximity(t.dets, board_polygon)
            score += BOARD_BONUS * at_board * (1.0 - score)
        out[t.tid] = score
    return out


def _transit_cost(a: Tracklet, b: Tracklet) -> float:
    """0..1 implausibility of the same person going from `a` to `b`.

    Zero once the gap is long enough that she could be anywhere (she may have
    left the room entirely) — the point of a global assignment is that a long
    absence is NOT evidence against, which is exactly what a spatial-continuity
    chain gets wrong about a teacher who steps out and comes back.
    """
    gap = max(0, b.first_ms - a.last_ms)
    if gap >= TRANSIT_FREE_GAP_MS:
        return 0.0
    ax, ay = a.tail_center()
    bx, by = b.head_center()
    dist = math.hypot(bx - ax, by - ay)
    allowed = min(TRANSIT_MAX, TRANSIT_BASE + TRANSIT_PER_S * gap / 1000.0)
    if dist <= allowed:
        return 0.0
    # Fade in with distance rather than a cliff: tracking noise at a handoff
    # should not read the same as a jump across the room.
    return min(1.0, (dist - allowed) / max(allowed, 1e-6))


def _allowed_overlap(a: Tracklet, b: Tracklet) -> int:
    """How much two consecutive claims may overlap before they are two people."""
    share = MAX_CLAIM_OVERLAP_FRACTION * min(a.span_ms, b.span_ms)
    return int(max(OVERLAP_TOLERANCE_MS, min(MAX_CLAIM_OVERLAP_MS, share)))


def select_timeline(
    tracklets: list[Tracklet], scores: dict[int, float]
) -> list[Tracklet]:
    """Best time-disjoint set of tracklets — the teacher's timeline.

    Weighted interval scheduling: value is (score - CLAIM_THRESHOLD) * duration,
    so a tracklet only earns its place by looking more like the teacher than the
    threshold for as long as it lasts, and a chain of confident short fragments
    can outweigh one long mediocre one. Transitions between consecutive picks
    are charged for implausibility. Solved exactly by DP in O(n^2) over the few
    hundred tracklets a lesson produces.

    Disjointness is the constraint that encodes "there is one of her": two
    boxes alive at the same instant cannot both be the teacher, so the tracker's
    duplicate boxes and the merge's chimeras can never both be claimed.
    """
    if not tracklets:
        return []
    order = sorted(tracklets, key=lambda t: (t.last_ms, t.first_ms))
    n = len(order)
    value = [
        (scores.get(t.tid, 0.0) - CLAIM_THRESHOLD) * max(t.span_ms, 1)
        for t in order
    ]
    # Scale so transition penalties are commensurate with value.
    scale = max(1.0, max(abs(v) for v in value))

    best = [0.0] * n
    prev = [-1] * n
    for i, t in enumerate(order):
        if value[i] <= 0:
            best[i] = -math.inf  # never worth claiming on its own merits
            continue
        best[i] = value[i]
        for j in range(i):
            if best[j] == -math.inf:
                continue
            if order[j].last_ms > t.first_ms + _allowed_overlap(order[j], t):
                continue  # genuinely co-present: cannot be the same person
            cand = (
                best[j]
                + value[i]
                - W_TRANSIT * scale * _transit_cost(order[j], t)
            )
            if cand > best[i]:
                best[i] = cand
                prev[i] = j

    end = max(range(n), key=lambda i: best[i])
    if best[end] <= 0:
        return []
    chain: list[Tracklet] = []
    i = end
    while i >= 0:
        chain.append(order[i])
        i = prev[i]
    chain.reverse()
    return chain


# --------------------------------------------------------------------------- #
# 4. Public entry point
# --------------------------------------------------------------------------- #


@dataclass
class TeacherTimeline:
    tracklets: list[Tracklet]
    confidence: float
    scores: dict[int, float]
    adult: dict[int, adult_mod.AdultEvidence]
    all_tracklets: list[Tracklet]
    notes: list[str] = field(default_factory=list)

    @property
    def det_count(self) -> int:
        return sum(len(t.dets) for t in self.tracklets)

    @property
    def raw_ids(self) -> list[int]:
        return sorted({t.raw_id for t in self.tracklets})

    def detections(self) -> list[Detection]:
        """Her detections, with overlapping claim tails trimmed.

        Consecutive claims may overlap slightly because the instant a tracker
        handed her id over is only known approximately; the later claim wins
        that stretch, so the timeline never reports her in two places at once.
        """
        out: list[Detection] = []
        for i, t in enumerate(self.tracklets):
            stop = (
                self.tracklets[i + 1].first_ms
                if i + 1 < len(self.tracklets)
                else None
            )
            out.extend(
                d for d in t.dets if stop is None or d.video_ts_ms < stop
            )
        return out

    def covers(self, ts_ms: int) -> bool:
        return any(t.first_ms <= ts_ms <= t.last_ms for t in self.tracklets)


def _prototype(picked: list[Tracklet], galleries: Gallery) -> np.ndarray:
    """Appearance prototype: every claimed view of her, as a gallery."""
    parts = [g for g in (_gallery_for(t, galleries) for t in picked) if g.size]
    if not parts:
        return np.zeros((0, 0))
    return np.vstack(parts)


def _hist_for(tracklet: Tracklet, hists: dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """This tracklet's torso colour histogram, L1-normalized. Keyed by TRACKLET id.

    Histograms are stored per RAW id and carry no timestamps, so a tracklet
    inherits its parent's colour (find_teacher does that mapping once). That is
    a real limitation for a raw id that changed person mid-way — which is why
    such ids are split on their CLIP gallery, which IS timestamped, before this
    is ever consulted.
    """
    h = hists.get(tracklet.tid)
    if h is None:
        return None
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    total = float(arr.sum())
    return arr / total if total > 0 else None


def _hist_prototype(
    picked: list[Tracklet], hists: dict[int, np.ndarray]
) -> Optional[np.ndarray]:
    parts = [h for h in (_hist_for(t, hists) for t in picked) if h is not None]
    if not parts:
        return None
    proto = np.mean(np.stack(parts), axis=0)
    total = float(proto.sum())
    return proto / total if total > 0 else None


def _hist_affinity(
    hist: Optional[np.ndarray], prototype: Optional[np.ndarray]
) -> Optional[float]:
    """Bhattacharyya overlap of two colour histograms, or None."""
    if hist is None or prototype is None or hist.shape != prototype.shape:
        return None
    return float(np.sqrt(hist * prototype).sum())


def find_teacher(
    dets_by_raw: dict[int, list[Detection]],
    galleries: Optional[dict] = None,
    hists: Optional[dict] = None,
    zero_shot: Optional[dict[int, float]] = None,
    duration_ms: int = 0,
    zones: Optional[list[dict]] = None,
) -> Optional[TeacherTimeline]:
    """The teacher's timeline over the whole video, or None when there is no adult.

    `galleries` (CLIP crops, for matching her to herself), `hists` (torso
    colour, for telling a uniform from whatever she wears) and `zero_shot` are
    all keyed by RAW track id — what the detector and the database store — and
    all optional: without them the assignment falls back to stature, body
    proportions and behaviour.
    """
    dets_by_raw = {k: v for k, v in dets_by_raw.items() if v}
    if not dets_by_raw:
        return None

    board_polygon = next(
        (z["polygon"] for z in (zones or []) if z.get("kind") == "board"), None
    )
    plane = adult_mod.fit_ground_plane(dets_by_raw)
    galleries = normalize_galleries(galleries)
    tracklets = build_tracklets(dets_by_raw, plane, galleries)
    if not tracklets:
        return None

    by_tid = {t.tid: t.dets for t in tracklets}
    # Appearance evidence is per raw id; a split tracklet inherits its parent's
    # embedding only for the samples taken inside its own time range.
    hists_by_tid: dict[int, np.ndarray] = {}
    for t in tracklets:
        h = (hists or {}).get(t.raw_id)
        if h is None:
            continue
        arr = np.asarray(h, dtype=np.float64)
        # A stored median arrives flat; a sampled set arrives as rows.
        hists_by_tid[t.tid] = arr.mean(axis=0) if arr.ndim > 1 else arr
    zero_by_tid = (
        {t.tid: zero_shot[t.raw_id] for t in tracklets if t.raw_id in zero_shot}
        if zero_shot
        else None
    )

    attributable = {t.tid for t in tracklets if t.hist_attributable}

    evidence = adult_mod.score_tracks(
        by_tid,
        hists={k: v for k, v in hists_by_tid.items() if k in attributable},
        zero_shot=zero_by_tid,
        plane=plane,
    )
    adult_scores = {tid: e.score for tid, e in evidence.items()}

    notes: list[str] = []
    if not plane.ok:
        notes.append(
            "Too few standing people to fit a perspective model; adult "
            "evidence falls back to body proportions and appearance."
        )

    # --- bootstrap: the one long stretch that behaves like a teacher --------
    seeds = seed_scores(tracklets, adult_scores, duration_ms)
    if not seeds:
        notes.append(
            "No tracklet is long enough to judge teaching behaviour; the "
            "lesson may be too short or too fragmented to identify a teacher."
        )
        return None
    seed_tid = max(seeds, key=lambda tid: seeds[tid])
    seed = next(t for t in tracklets if t.tid == seed_tid)

    # --- refine: prototype -> rescore -> reselect ---------------------------
    reference = [seed]
    picked: list[Tracklet] = []
    scores: dict[int, float] = {}
    for _round in range(REFINE_ROUNDS):
        clip_rank = adult_mod._ranks(
            _affinities(tracklets, reference, galleries, hists_by_tid, "clip")
        )
        hist_rank = adult_mod._ranks(
            _affinities(
                tracklets, reference, galleries, hists_by_tid, "hist", attributable
            )
        )
        appearance = _fuse_appearance(tracklets, clip_rank, hist_rank)

        scores = score_tracklets(
            tracklets, adult_scores, appearance or None, board_polygon
        )
        nxt = select_timeline(tracklets, scores)
        if not nxt:
            break
        converged = [t.tid for t in nxt] == [t.tid for t in picked]
        picked = nxt
        if converged:
            break
        # The prototype stays ANCHORED on the seed and grows only by claims
        # that look like the seed. Rebuilding it from "the highest-scoring
        # claims" instead lets one wrongly claimed pupil into the reference,
        # after which the prototype is partly him, his classmates score well
        # against it, and the search walks away from the teacher entirely —
        # measured as a drop from 95% coverage to 78% on real footage.
        affinity_to_seed = {
            t.tid: _seed_affinity(t, seed, galleries, hists_by_tid, attributable)
            for t in tracklets
        }
        admitted = _outliers(affinity_to_seed, PROTOTYPE_ADMIT_Z)
        rest = sorted(
            (t for t in picked if t.tid != seed.tid and t.tid in admitted),
            key=lambda t: -affinity_to_seed[t.tid],
        )
        reference = [seed, *rest[: PROTOTYPE_TOP_K - 1]]

    if not picked:
        return None

    confidence = _confidence(picked, tracklets, scores)
    return TeacherTimeline(
        tracklets=picked,
        confidence=confidence,
        scores=scores,
        adult=evidence,
        all_tracklets=tracklets,
        notes=notes,
    )


def _confidence(
    picked: list[Tracklet], tracklets: list[Tracklet], scores: dict[int, float]
) -> float:
    """0.5 + lead, on the scale quality.py tiers (high >= 0.65, medium >= 0.55).

    The lead compares the duration-weighted score of the claimed timeline with
    the best CONTENDER — the strongest tracklet that was alive at the same time
    as one of the claims and therefore lost the slot to it. That is the real
    question behind trusting these KPIs: was there another body in this room
    that looked nearly as much like the teacher?
    """
    total = sum(max(t.span_ms, 1) for t in picked)
    mine = sum(scores.get(t.tid, 0.0) * max(t.span_ms, 1) for t in picked) / max(total, 1)

    picked_tids = {t.tid for t in picked}
    rival = 0.0
    for t in tracklets:
        if t.tid in picked_tids:
            continue
        overlaps = any(
            min(t.last_ms, p.last_ms) - max(t.first_ms, p.first_ms)
            > OVERLAP_TOLERANCE_MS
            for p in picked
        )
        if overlaps:
            rival = max(rival, scores.get(t.tid, 0.0))
    return round(min(1.0, max(0.0, 0.5 + (mine - rival))), 4)
