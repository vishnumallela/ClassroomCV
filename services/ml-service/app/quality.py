"""Per-run data-quality assessment: how much can you trust these analytics?

Every dashboard number is an estimate over sampled, occluded CCTV. A school
leader reading "42 minutes at the board" deserves to know whether that figure
came from a clean, well-covered lesson or from a camera that lost the teacher
for a third of it. This module turns the pipeline's own signals into an honest,
additive confidence report. It NEVER changes a derived number — it only
annotates them.

The three signals changed with the detector, because two of the old ones stopped
meaning anything. "Distinct identities" and "tracker ids merged per identity"
measured how hard the old appearance merge had to work to reassemble one person
out of fragments; with the teacher detected as a named class there is no merge,
no fragments and no rival identity, so reporting a fragmentation of 1.0 every
time would be theatre. What can still go wrong is different, and this is it:

1. COVERAGE: of the frames sampled across the lesson, in how many was she
   actually found? This is the honest denominator behind every duration: at 80%
   coverage, "time at the board" is a floor rather than a measurement.

2. CONTINUITY: how many times did her timeline break, and what was the longest
   break? Entries and exits are counted from those breaks, so a lesson full of
   short dropouts is one where that KPI specifically should be distrusted —
   even when total coverage looks healthy.

3. DETECTION CONFIDENCE: how sure the model was when it did find her. A lesson
   held together by 0.3-confidence boxes is a lesson to be careful with.

4. ATTRIBUTION: was there only one adult to follow in the first place?

The fourth is not like the other three, and the difference is the reason it
exists. Coverage, continuity and confidence all ask HOW WELL one person was
followed, and all three degrade visibly when the answer is "badly". Attribution
asks whether following one person was the right thing to do at all — and when
the answer is no, the other three do not degrade. They improve. A timeline
blended from two adults is unbroken, well covered and confidently detected,
because there is always *a* box to find: the measured handover scored high on
every axis while reporting the wrong teacher's arrival. A trust report that
could not say this would be most reassuring in exactly the case it exists for.
"""

from __future__ import annotations

from typing import Optional

from app.models import Detection

# A gap in her detections at least this long is a break in the timeline rather
# than a sampling wobble. Matches heuristics.PRESENCE_GAP_MS, so the breaks
# counted here are exactly the ones entry/exit is derived from.
BREAK_GAP_MS = 5_000

# Coverage of the lesson (fraction of sampled frames she was found in). Below
# these, duration-based metrics undercount.
COVERAGE_HIGH = 0.85
COVERAGE_LOW = 0.6

# Timeline breaks per lesson. A handful is normal (she leaves the room); dozens
# means the detector kept losing her and the entry/exit count is inflated.
BREAKS_CLEAN = 6
BREAKS_NOISY = 20

# Mean detection confidence tiers.
CONF_HIGH = 0.7
CONF_LOW = 0.5

Tier = str  # "high" | "medium" | "low"


def _worst(*tiers: Tier) -> Tier:
    """Weakest-link tier over the given tiers (low < medium < high)."""
    order = {"low": 0, "medium": 1, "high": 2}
    present = [t for t in tiers if t in order]
    if not present:
        return "low"
    return min(present, key=lambda t: order[t])


def _breaks(ts_sorted: list[int], gap_ms: int) -> tuple[int, int]:
    """(number of breaks, longest break in ms) in a sorted timestamp list."""
    breaks = 0
    longest = 0
    for prev, cur in zip(ts_sorted, ts_sorted[1:]):
        gap = cur - prev
        if gap >= gap_ms:
            breaks += 1
            longest = max(longest, gap)
    return breaks, longest


def assess(
    teacher_dets: list[Detection],
    sampled_frames: int,
    duration_ms: int,
    teacher_confidence: Optional[float] = None,
    mean_conf: Optional[float] = None,
    gap_ms: int = BREAK_GAP_MS,
    multiple_adults: bool = False,
    co_presence_ms: int = 0,
    max_simultaneous: int = 1,
    attribution: Optional[dict] = None,
) -> dict:
    """Additive data-quality report for one analysed video.

    Pure function of the same teacher detections the analytics are derived
    from, so it is exact and free to compute. Returns a JSON-friendly dict;
    callers attach it to the analytics payload untouched.
    """
    dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)
    ts = [d.video_ts_ms for d in dets]
    frames = len(set(ts))
    coverage = frames / sampled_frames if sampled_frames > 0 else 0.0
    breaks, longest_gap = _breaks(ts, gap_ms)
    if mean_conf is None:
        mean_conf = (sum(d.conf for d in dets) / len(dets)) if dets else 0.0

    notes: list[str] = []

    # --- coverage ---------------------------------------------------------- #
    if not dets:
        coverage_tier: Tier = "low"
        notes.append(
            "The teacher was never detected in this lesson; teacher metrics are "
            "unavailable."
        )
    elif coverage >= COVERAGE_HIGH:
        coverage_tier = "high"
    elif coverage >= COVERAGE_LOW:
        coverage_tier = "medium"
        notes.append(
            f"The teacher was visible in {coverage * 100:.0f}% of sampled frames; "
            "durations are a floor rather than an exact measurement."
        )
    else:
        coverage_tier = "low"
        notes.append(
            f"The teacher was found in only {coverage * 100:.0f}% of sampled frames "
            "(heavy occlusion or dropout), so time-based numbers undercount."
        )

    # --- continuity -------------------------------------------------------- #
    if not dets:
        continuity_tier: Tier = "low"
    elif breaks <= BREAKS_CLEAN:
        continuity_tier = "high"
    elif breaks <= BREAKS_NOISY:
        continuity_tier = "medium"
        notes.append(
            f"Her timeline breaks {breaks} times; some entries and exits may be "
            "tracking dropouts rather than real door crossings."
        )
    else:
        continuity_tier = "low"
        notes.append(
            f"Her timeline breaks {breaks} times (longest {longest_gap / 1000:.0f}s): "
            "treat the entry/exit count as an upper bound."
        )

    # --- teacher identification -------------------------------------------- #
    if teacher_confidence is None or not dets:
        teacher_tier: Tier = "low"
    elif mean_conf >= CONF_HIGH:
        teacher_tier = "high"
    elif mean_conf >= CONF_LOW:
        teacher_tier = "medium"
    else:
        teacher_tier = "low"
        notes.append(
            "The teacher was detected at low confidence throughout; the labelling "
            "is tentative."
        )

    # --- attribution: is this timeline one person, and the right one? -------- #
    # With Phase 3's report the tier IS its confidence: high means one adult, or
    # several with a decided answer; medium means decided on thin evidence (the
    # dashboard still withholds); low means undetermined. Without a report (rows
    # predating Phase 3) the Phase 0 rule stands: two adults and no answer is
    # low. multiple_adults itself is still app/teacher.py's verdict, not
    # re-derived here, so there is one definition of it.
    if attribution is not None:
        attribution_tier: Tier = attribution["confidence"]
        if multiple_adults or attribution_tier != "high":
            notes.append(attribution["reason"])
    elif multiple_adults:
        attribution_tier = "low"
        notes.append(
            f"Two or more adults were in the room together for about "
            f"{co_presence_ms / 1000:.0f}s. Nothing yet decides which of them "
            "this lesson assesses, so arrival, departure, time in the room and "
            "the entry/exit counts are reported as Not Observed rather than "
            "measured against a timeline that may blend them."
        )
    elif not dets:
        attribution_tier = "low"
    else:
        attribution_tier = "high"

    overall = _worst(coverage_tier, continuity_tier, teacher_tier, attribution_tier)

    return {
        "detections": len(dets),
        "frames": frames,
        "sampled_frames": sampled_frames,
        "coverage": round(coverage, 3),
        "mean_confidence": round(float(mean_conf), 3),
        "breaks": breaks,
        "longest_gap_ms": longest_gap,
        # Additive, and describing the ROOM rather than her: how many adults the
        # detector saw at once, and for how long. Present on every report so a
        # reader can tell "measured, and there was one adult" from "this lesson
        # predates the check", which a bare boolean could not express.
        "multiple_adults_detected": multiple_adults,
        "max_simultaneous_adults": max_simultaneous,
        "co_presence_ms": co_presence_ms,
        "attribution": attribution,
        "confidence": {
            "overall": overall,
            "coverage": coverage_tier,
            "continuity": continuity_tier,
            "teacher": teacher_tier,
            "attribution": attribution_tier,
        },
        "notes": notes,
    }
