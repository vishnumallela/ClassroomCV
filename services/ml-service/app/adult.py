"""Adult-vs-child evidence per tracked person — the cue that actually names the teacher.

A classroom is the one surveillance scene where the target is defined by AGE:
twenty-odd children and one adult. Every behavioural proxy the pipeline used
before (stands a lot, moves a lot, present a lot, near the board) breaks on
real lessons — during the first minutes the whole class is on its feet, a
pupil sent to the board stands and walks, and a teacher who sits with a group
stops looking like a teacher entirely. Age does not break: she is an adult for
the whole hour, in every frame she appears, whether she is walking, crouching
at a desk, or half-occluded behind a row.

Four independent measurements, each scale-free and each computed RELATIVE TO
THE POPULATION IN THIS VIDEO (no absolute pixel or colour constants, so the
same code works on any room, camera height, or uniform):

STATURE   How tall this person is compared with what a body standing at that
          spot SHOULD measure. A ground-plane model (height vs the y of the
          feet) is fitted across everyone, then each track reports the median
          ratio of its own height to the prediction. This is the fix for the
          single most misleading signal in a wide-angle classroom: the raw
          pixel height of a front-row child exceeds the teacher's at the
          board, which is why an unnormalized "adult = tallest" rule
          (merge.ADULT_HEIGHT_PERCENTILE) keeps electing seated pupils.

PROPORTION  Children are not scaled-down adults: their heads are larger
          relative to the torso and their legs shorter (roughly 1/6 of stature
          at 6 years vs 1/8 for an adult). Ratios of pose-keypoint distances
          carry that difference at any distance from the camera, and survive
          the lower body being hidden by a desk.

DISTINCTIVENESS  In a uniformed school the children dress alike and the
          teacher does not. Rather than hard-coding a colour, each track is
          scored on how far its appearance sits from the crowd's — the
          population's own consensus defines "uniform", so this works whether
          the uniform is a green polo or a grey blazer.

ZERO-SHOT  Optional CLIP adult-vs-child reading of the same crops the re-ID
          already embeds (detector.zero_shot_adult). Free when CLIP ran,
          simply absent otherwise.

Every signal is optional and every signal is weighted by how much data stands
behind it, with the total shrunk towards a child prior. That combination is
load-bearing: on the first real run, tracks with NO usable measurement scored
higher than the teacher, because dropping their missing signals left one noisy
signal to speak for them. Absence of evidence must read as "probably a pupil,
like everyone else in the room", never as a free pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.models import Detection

# --- measurement gates ------------------------------------------------------
# A body a third hidden behind someone else has a meaningless bbox height and
# a poisoned appearance crop (see detector._occlusions), so stature and colour
# are read only off clean views.
MAX_OCCLUSION = 0.35
# Pose ratios survive far more occlusion than either: the keypoints that ARE
# visible still belong to this person, and `vis` already reports how many of
# them the model was confident about. Gating limb proportions as strictly as
# height threw away the age evidence exactly when it was most needed — in a
# packed room, where the teacher is half-hidden almost every frame and nothing
# else identifies her.
MAX_OCCLUSION_PROPORTION = 0.6
# A person on their feet is the only one whose bbox height means stature; a
# seated body's box is cut off by the desk in front of it.
MIN_STANDING_SAMPLES = 5
# Pose ratios need keypoints the model is actually confident about.
MIN_KEYPOINT_VIS = 0.5
MIN_PROPORTION_SAMPLES = 5

# --- ground-plane model -----------------------------------------------------
# Feet lower in the frame are nearer the camera, so height grows with foot y.
# Fitted by a Theil-Sen style median of pairwise slopes: with one adult among
# thirty children a least-squares fit is dragged by whichever group is nearer
# the camera, while the median slope ignores both tails.
GROUND_MIN_TRACKS = 4
GROUND_MIN_Y_SPREAD = 0.08
GROUND_SLOPE_PAIRS = 400

# --- fusion -----------------------------------------------------------------
# Stature leads: it is the measurement with the most samples behind it and the
# widest adult/child separation once perspective is removed. Proportions are
# noisier per sample but independent of clothing and lighting. Distinctiveness
# is a strong classroom prior but fails for a teacher who wears the school
# polo, so it never dominates. Zero-shot is a genuine age reading but CLIP is
# weak on 40-pixel CCTV crops.
W_STATURE = 0.40
W_PROPORTION = 0.25
W_DISTINCT = 0.20
W_ZERO_SHOT = 0.15

# Stature ratio mapped through this band: 1.0 = exactly room-typical for that
# spot. The band is deliberately wide because how much taller the adult is
# depends on the year group — measured on a senior class, the teacher stands
# only ~10% above her pupils, and a band tuned on infants would have scored
# her as a child. Stature is evidence, not proof, which is why it is one
# weighted signal rather than the rule.
STATURE_LO = 1.00
STATURE_HI = 1.35

# How many robust standard deviations from the crowd's average appearance
# count as fully "not wearing the uniform". Measured: the teacher's fragments
# sit 1.3-1.5 MADs below the population's similarity-to-consensus.
DISTINCT_Z_FULL = 1.5

# Shrinkage towards the population. With every signal present at full
# reliability the prior contributes about a fifth of the score; with one weak
# signal it dominates, which is the point.
CHILD_PRIOR = 0.30
SHRINKAGE = 0.25


@dataclass
class AdultEvidence:
    """Per-track age evidence. `score` is 0..1, higher = more adult-like."""

    key: int
    score: float
    stature: Optional[float] = None
    head: Optional[float] = None
    leg: Optional[float] = None
    distinct: Optional[float] = None
    zero_shot: Optional[float] = None
    samples: int = 0
    # Total reliability-weighted evidence behind `score`. Near zero means the
    # score IS the prior: nothing was measurable for this track.
    evidence: float = 0.0
    signals: list[str] = field(default_factory=list)

    def explain(self) -> str:
        bits = [f"{s}={getattr(self, s):.2f}" for s in self.signals]
        return f"adult={self.score:.2f} (" + " ".join(bits) + ")"


# --------------------------------------------------------------------------- #
# Ground-plane (perspective) model
# --------------------------------------------------------------------------- #


@dataclass
class GroundPlane:
    """Expected standing height of a body whose feet are at image row y."""

    a: float
    b: float
    ok: bool = True
    # 0..1 confidence in the fit. A camera whose people all stand at a similar
    # depth, or a room where almost nobody stands, yields a nearly flat slope:
    # the model then predicts one height everywhere and "stature" degenerates
    # into raw box height, which is exactly the measurement perspective was
    # supposed to correct. Consumers weight the signal by this rather than
    # trusting a degenerate fit at face value.
    confidence: float = 1.0

    def predict(self, foot_y: float) -> float:
        return max(1e-4, self.a + self.b * foot_y)


def _clean_standing(dets: list[Detection]) -> list[Detection]:
    return [
        d
        for d in dets
        if d.standing and d.occlusion <= MAX_OCCLUSION and d.bbox.get("h", 0) > 0
    ]


def fit_ground_plane(dets_by_key: dict[int, list[Detection]]) -> GroundPlane:
    """Fit height ~ a + b * foot_y over everyone's clean standing detections.

    One (median) point per track, so a pupil tracked for the whole lesson does
    not outvote thirty briefly-seen classmates, and the slope is the median of
    pairwise slopes rather than a least-squares fit: with a single adult in the
    room the residuals are deliberately non-Gaussian and least squares would
    partly absorb her into the model it is supposed to measure her against.
    """
    points: list[tuple[float, float]] = []
    for dets in dets_by_key.values():
        clean = _clean_standing(dets)
        if len(clean) < MIN_STANDING_SAMPLES:
            continue
        foot = float(np.median([d.bbox["y"] + d.bbox["h"] for d in clean]))
        height = float(np.median([d.bbox["h"] for d in clean]))
        points.append((foot, height))

    degraded = False
    if len(points) < GROUND_MIN_TRACKS:
        # A lesson where almost nobody stands leaves too few clean standing
        # tracks to fit anything. Fall back to each person's TALLEST posture
        # (p90 height), which is the closest thing to their standing height
        # the footage contains, and mark the fit degraded so consumers weight
        # it down rather than trusting it.
        degraded = True
        points = []
        for dets in dets_by_key.values():
            usable = [d for d in dets if d.occlusion <= MAX_OCCLUSION and d.bbox.get("h", 0) > 0]
            if len(usable) < MIN_STANDING_SAMPLES:
                continue
            hs = np.array([d.bbox["h"] for d in usable])
            tall = np.argsort(hs)[-max(1, len(hs) // 10) :]
            points.append(
                (
                    float(np.median([usable[i].bbox["y"] + usable[i].bbox["h"] for i in tall])),
                    float(np.median(hs[tall])),
                )
            )

    if len(points) < GROUND_MIN_TRACKS:
        return GroundPlane(
            a=float(np.median([p[1] for p in points])) if points else 0.2,
            b=0.0,
            ok=False,
            confidence=0.0,
        )

    ys = np.array([p[0] for p in points])
    hs = np.array([p[1] for p in points])
    if float(ys.max() - ys.min()) < GROUND_MIN_Y_SPREAD:
        return GroundPlane(a=float(np.median(hs)), b=0.0, ok=False, confidence=0.0)

    slopes: list[float] = []
    n = len(points)
    step = max(1, (n * (n - 1) // 2) // GROUND_SLOPE_PAIRS)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            k += 1
            if k % step:
                continue
            dy = ys[j] - ys[i]
            if abs(dy) > 1e-6:
                slopes.append((hs[j] - hs[i]) / dy)
    if not slopes:
        return GroundPlane(a=float(np.median(hs)), b=0.0, ok=False, confidence=0.0)
    b = float(np.median(slopes))
    # Perspective can only make nearer (lower) bodies taller; a negative slope
    # means the fit found noise, not geometry.
    if b < 0:
        b = 0.0
    a = float(np.median(hs - b * ys))
    # How much of the height variation the model actually explains: the height
    # difference the slope predicts across the observed rows, against the
    # spread of the heights themselves.
    explained = b * float(ys.max() - ys.min())
    spread = float(np.percentile(hs, 90) - np.percentile(hs, 10))
    confidence = float(min(1.0, explained / spread)) if spread > 1e-6 else 0.0
    if degraded:
        confidence *= 0.5
    return GroundPlane(a=a, b=b, ok=True, confidence=max(0.1, confidence))


# --------------------------------------------------------------------------- #
# Per-track measurements
# --------------------------------------------------------------------------- #


def stature_ratio(dets: list[Detection], plane: GroundPlane) -> Optional[float]:
    """Median height / ground-plane prediction over clean standing detections."""
    clean = _clean_standing(dets)
    if len(clean) < MIN_STANDING_SAMPLES:
        return None
    ratios = [
        d.bbox["h"] / plane.predict(d.bbox["y"] + d.bbox["h"]) for d in clean
    ]
    return float(np.median(ratios))


def proportions(
    dets: list[Detection],
) -> tuple[Optional[float], Optional[float], int]:
    """(head/torso, leg/torso, samples behind the leg measurement)."""
    heads: list[float] = []
    legs: list[float] = []
    for d in dets:
        body = d.body
        if not body or d.occlusion > MAX_OCCLUSION_PROPORTION:
            continue
        if float(body.get("vis") or 0.0) < MIN_KEYPOINT_VIS:
            continue
        if body.get("head") is not None:
            heads.append(float(body["head"]))
        if body.get("leg") is not None:
            legs.append(float(body["leg"]))
    head = float(np.median(heads)) if len(heads) >= MIN_PROPORTION_SAMPLES else None
    leg = float(np.median(legs)) if len(legs) >= MIN_PROPORTION_SAMPLES else None
    return head, leg, len(legs)


def distinctiveness(hists: dict[int, np.ndarray]) -> dict[int, float]:
    """How far each track's COLOUR sits from the population consensus, 0..1.

    Torso histograms, not CLIP: a school uniform is a colour, and the
    population's own average histogram is therefore a picture of the uniform.
    Bhattacharyya similarity to that consensus, ranked across tracks, puts the
    one person not wearing it at the top without anyone naming a colour.

    Measured on real footage, this is the strongest single appearance cue for
    finding the adult (her main fragments rank in the most-distinctive tenth,
    against a population median of 0.53 similarity), and it clearly beats the
    same construction over CLIP embeddings, where crops are dominated by
    background and her fragments land mid-pack. CLIP still earns its place
    elsewhere — matching HER to HERSELF across the lesson (teacher_track's
    prototype affinity) is a different question from telling her apart from a
    room of uniforms.
    """
    keys = [k for k, v in hists.items() if v is not None and np.any(v)]
    if len(keys) < 4:
        return {}
    mat = np.stack([np.asarray(hists[k], dtype=np.float64).ravel() for k in keys])
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)

    # "How similar is this person to a TYPICAL OTHER person in the room" —
    # the median of their pairwise similarities — rather than similarity to an
    # averaged consensus histogram. Two reasons, both learned the hard way:
    # an average consensus is dragged towards whoever is most fragmented (a
    # teacher split into a dozen tracks stops being an outlier against a
    # picture that is partly her), and averaging sparse histograms with
    # different peaks produces a consensus that resembles nobody at all.
    root = np.sqrt(mat)
    sim_matrix = root @ root.T  # Bhattacharyya coefficients, pairwise
    np.fill_diagonal(sim_matrix, np.nan)
    sim = np.nanmedian(sim_matrix, axis=1)

    # Rank alone always crowns somebody, even in a room where everyone is
    # dressed identically — which is precisely a room with no teacher in it.
    # Multiplying the rank by HOW FAR the outlier really sits from the crowd
    # (robust z, so a handful of odd crops cannot set the scale) keeps the
    # ordering while letting the whole signal fall to nothing when no one
    # genuinely stands out.
    med = float(np.median(sim))
    mad = float(np.median(np.abs(sim - med))) * 1.4826
    if mad < 1e-6:
        return {k: 0.0 for k in keys}
    rank = _ranks({k: -float(sim[i]) for i, k in enumerate(keys)})
    out: dict[int, float] = {}
    for i, k in enumerate(keys):
        z = (med - float(sim[i])) / mad
        out[k] = rank[k] * float(min(1.0, max(0.0, z / DISTINCT_Z_FULL)))
    return out


def _ranks(values: dict[int, float]) -> dict[int, float]:
    """Rank values within this video onto 0..1 (1.0 = highest value).

    Rank rather than magnitude wherever a measurement's scale depends on the
    camera rather than the person — how foreshortened a limb looks, how
    compressed appearance similarities are on this lens. The question worth
    asking is always "who in THIS room is most adult-like", never "does this
    number exceed a constant somebody wrote down against another camera".
    """
    if not values:
        return {}
    if len(values) == 1:
        return {k: 0.5 for k in values}
    keys = list(values)
    arr = np.array([values[k] for k in keys], dtype=np.float64)
    # Average rank for ties. Breaking ties by position instead would spread a
    # group of identical measurements across the whole 0..1 range — which is
    # how a roomful of pupils with identical body proportions came to include
    # one that outranked the teacher.
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return {k: float(ranks[i]) / (len(keys) - 1) for i, k in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def _band(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Map value onto 0..1 across [lo, hi] (inverted when hi < lo)."""
    if value is None:
        return None
    if hi == lo:
        return 0.5
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def score_tracks(
    dets_by_key: dict[int, list[Detection]],
    hists: Optional[dict[int, np.ndarray]] = None,
    zero_shot: Optional[dict[int, float]] = None,
    plane: Optional[GroundPlane] = None,
) -> dict[int, AdultEvidence]:
    """Adult evidence per key (raw track id or tracklet id).

    Two properties matter more than the exact weights:

    RELIABILITY-WEIGHTED. Every measurement carries how much data stands
    behind it — a stature read off 200 clean standing frames is not the same
    claim as one read off 5. Without this the ranking is topped by
    three-detection fragments whose single noisy sample happened to look
    adult, which is exactly what happened on the first real run.

    SHRUNK TOWARDS THE POPULATION. Scores are pulled towards a child prior in
    proportion to how little evidence stands behind them, so absence of
    evidence can never read as evidence of adulthood — the failure where a
    track with no usable measurement outscored the teacher because dropping
    its missing signals left one noisy signal unopposed.

    Pure function of detections plus optional appearance, so /analyze,
    /rederive and the offline harness all produce identical scores.
    """
    plane = plane or fit_ground_plane(dets_by_key)
    distinct = distinctiveness(hists or {})

    stature_raw: dict[int, Optional[float]] = {}
    leg_raw: dict[int, Optional[float]] = {}
    head_raw: dict[int, Optional[float]] = {}
    clean_n: dict[int, int] = {}
    leg_n: dict[int, int] = {}
    for key, dets in dets_by_key.items():
        clean_n[key] = len(_clean_standing(dets))
        stature_raw[key] = stature_ratio(dets, plane) if plane.ok else None
        head_raw[key], leg_raw[key], leg_n[key] = proportions(dets)

    # Limb proportions are RANKED, not banded: how long a leg looks depends on
    # the camera's pitch as much as on the person, so only the ordering within
    # one video is meaningful. (head/torso is kept as a diagnostic but not
    # scored — on a ceiling-mounted camera it measures head TILT, and the
    # measured adult/child values came out the opposite way round from the
    # anthropometry it was meant to encode. A prior the data contradicts is
    # not a prior worth keeping.)
    leg_rank = _ranks({k: v for k, v in leg_raw.items() if v is not None})

    out: dict[int, AdultEvidence] = {}
    for key, dets in dets_by_key.items():
        parts: list[tuple[float, float, str]] = []  # (weight, value, name)

        s = _band(stature_raw[key], STATURE_LO, STATURE_HI)
        if s is not None:
            parts.append(
                (
                    W_STATURE * _reliability(clean_n[key], 30) * plane.confidence,
                    s,
                    "stature",
                )
            )
        if key in leg_rank:
            # Weighted by how many limb measurements exist, NOT by how often
            # this person stood: proportions are exactly the age cue that still
            # works for a teacher who spends the lesson sitting with a group.
            parts.append(
                (W_PROPORTION * _reliability(leg_n[key], 20), leg_rank[key], "leg")
            )
        if key in distinct:
            parts.append((W_DISTINCT, distinct[key], "distinct"))
        z = (zero_shot or {}).get(key)
        if z is not None:
            parts.append((W_ZERO_SHOT, float(z), "zero_shot"))

        weight = sum(w for w, _v, _n in parts)
        score = (sum(w * v for w, v, _n in parts) + SHRINKAGE * CHILD_PRIOR) / (
            weight + SHRINKAGE
        )

        out[key] = AdultEvidence(
            key=key,
            score=round(float(score), 4),
            stature=stature_raw[key],
            head=head_raw[key],
            leg=leg_raw[key],
            distinct=distinct.get(key),
            zero_shot=z,
            samples=len(dets),
            evidence=round(float(weight), 3),
            signals=[n for _w, _v, n in parts],
        )
    return out


def _reliability(n: int, full: int) -> float:
    """0..1 confidence in a measurement made from n samples (1.0 at `full`)."""
    return float(min(1.0, n / max(1, full)))
