"""Per-run data-quality assessment: how much can you trust these analytics?

Every dashboard number is an estimate over sampled, occluded CCTV. A school
leader reading "average 27 students" deserves to know whether that figure came
from a clean, well-covered lesson or from a half-occluded camera whose tracker
fragmented every child into three identities. This module turns the pipeline's
own internal signals into an honest, additive confidence report. It NEVER
changes a derived number — it only annotates them.

Three independent signals drive the report:

1. Coverage: over the lesson's active span (first to last detection), what
   fraction of time buckets actually contained a detected person. Low coverage
   means the camera went dark, the frame was occluded, or the model dropped
   out — every presence/occupancy number is then a floor, not a measurement.

2. Fragmentation: raw tracker ids per final identity. The tracker mints a new
   id every time a person is occluded or leaves frame; the merge stage reunites
   them. A ratio near 1 means clean tracking; a high ratio means the merge did
   heavy lifting and identity-derived counts (max/avg students) carry more
   estimation error.

3. Concurrent crowd count: the number of non-teacher boxes visible in a single
   frame can never double-count one person (one body is one box per frame),
   so its peak/typical values are a re-identification-INDEPENDENT cross-check
   on the identity-based occupancy. When the two agree, occupancy is solid;
   when they diverge, the report says so.

The teacher-identification margin (roles.assign_roles' role_confidence) is
folded in as a fourth tier so the "who is the teacher" decision carries its own
trust level.
"""

from __future__ import annotations

from typing import Optional

from app.models import Detection

BUCKET_MS = 5_000

# Fragmentation (raw tracks / identity): <=2 is clean tracking, 2..4 means the
# merge stage did real work, >4 means identity-derived counts are estimates.
FRAG_CLEAN = 2.0
FRAG_NOISY = 4.0

# Coverage of the active span (fraction of buckets with any detection). Below
# these, the camera dropped out often enough that time-based metrics undercount.
COVERAGE_HIGH = 0.9
COVERAGE_LOW = 0.7

# Teacher-identification margin tiers (roles emits 0.5 + lead, capped at 1.0).
TEACHER_CONF_HIGH = 0.65
TEACHER_CONF_MED = 0.55

# How far the identity-based crowd count may sit below the fragmentation-immune
# concurrent peak before occupancy is downgraded: a large shortfall means whole
# students were missed by the identity count (over-merging or dropout).
OCCUPANCY_AGREEMENT_TOL = 3

Tier = str  # "high" | "medium" | "low"


def _percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _concurrent_counts(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
) -> list[int]:
    """Non-teacher box count per distinct frame timestamp (identity-free).

    One physical person contributes exactly one box per frame, so per-frame
    counts cannot double-count a fragmented identity the way distinct-track_no
    counting can. Returned unsorted, one entry per occupied frame.
    """
    per_frame: dict[int, int] = {}
    for track_no, dets in dets_by_track.items():
        if roles_map.get(track_no, ("unknown", None))[0] == "teacher":
            continue
        for d in dets:
            per_frame[d.video_ts_ms] = per_frame.get(d.video_ts_ms, 0) + 1
    return list(per_frame.values())


def _coverage(dets_by_track: dict[int, list[Detection]], bucket_ms: int) -> tuple[float, int, int]:
    """(fraction, occupied_buckets, span_buckets) over the active detection span.

    Measures dropout WITHIN the lesson (first to last detection), not the idle
    bookends before/after class, so a genuinely empty pre-lesson stretch does
    not read as poor data.
    """
    all_ts = [d.video_ts_ms for dets in dets_by_track.values() for d in dets]
    if not all_ts:
        return 0.0, 0, 0
    first, last = min(all_ts), max(all_ts)
    span_buckets = max(1, (last - first) // bucket_ms + 1)
    occupied = {(ts - first) // bucket_ms for ts in all_ts}
    return len(occupied) / span_buckets, len(occupied), span_buckets


def _worst(*tiers: Tier) -> Tier:
    """Weakest-link tier over the given tiers (low < medium < high)."""
    order = {"low": 0, "medium": 1, "high": 2}
    present = [t for t in tiers if t in order]
    if not present:
        return "low"
    return min(present, key=lambda t: order[t])


def assess(
    dets_by_track: dict[int, list[Detection]],
    roles_map: dict[int, tuple[str, Optional[float]]],
    duration_ms: int,
    teacher_confidence: Optional[float] = None,
    identity_max_students: int = 0,
    bucket_ms: int = BUCKET_MS,
) -> dict:
    """Additive data-quality report for one analysed video.

    Pure function of the same merged, role-labelled detections the analytics are
    derived from, so it is exact and free to compute. Returns a JSON-friendly
    dict; callers attach it to the analytics payload untouched.
    """
    detections = sum(len(d) for d in dets_by_track.values())
    identities = len(dets_by_track)
    raw_tracks = len(
        {d.raw_track_id for dets in dets_by_track.values() for d in dets}
    )
    frames = len({d.video_ts_ms for dets in dets_by_track.values() for d in dets})
    fragmentation = raw_tracks / identities if identities else 0.0

    coverage, occupied_buckets, span_buckets = _coverage(dets_by_track, bucket_ms)

    concurrent = sorted(_concurrent_counts(dets_by_track, roles_map))
    concurrent_peak = int(round(_percentile(concurrent, 0.95))) if concurrent else 0
    concurrent_typical = int(round(_percentile(concurrent, 0.5))) if concurrent else 0

    notes: list[str] = []

    # --- identity-tracking confidence (fragmentation) ---------------------- #
    if fragmentation <= FRAG_CLEAN:
        identity_tier = "high"
    elif fragmentation <= FRAG_NOISY:
        identity_tier = "medium"
        notes.append(
            f"Tracker fragmented people into ~{fragmentation:.1f} ids each; "
            "identity counts are re-id estimates."
        )
    else:
        identity_tier = "low"
        notes.append(
            f"Heavy fragmentation (~{fragmentation:.1f} ids per person): treat "
            "per-identity counts as approximate."
        )

    # --- coverage contribution --------------------------------------------- #
    if coverage >= COVERAGE_HIGH:
        coverage_tier = "high"
    elif coverage >= COVERAGE_LOW:
        coverage_tier = "medium"
        notes.append(
            f"The camera saw people in only {coverage * 100:.0f}% of the lesson's "
            "active span; presence and occupancy may undercount."
        )
    else:
        coverage_tier = "low"
        notes.append(
            f"Low coverage ({coverage * 100:.0f}% of the active span): frequent "
            "dropout or occlusion, so time-based numbers are a floor."
        )

    # --- occupancy confidence: coverage AND identity/concurrent agreement -- #
    shortfall = concurrent_peak - identity_max_students
    if concurrent_peak > 0 and shortfall > OCCUPANCY_AGREEMENT_TOL:
        notes.append(
            f"Up to {concurrent_peak} people were visible at once but only "
            f"{identity_max_students} distinct identities formed; the crowd was "
            "likely larger than the identity count."
        )
        agreement_tier: Tier = "low"
    elif concurrent_peak > 0 and shortfall > 1:
        agreement_tier = "medium"
    else:
        agreement_tier = "high"
    occupancy_tier = _worst(coverage_tier, agreement_tier, identity_tier)

    # --- teacher-identification confidence --------------------------------- #
    if teacher_confidence is None:
        teacher_tier: Tier = "low"
        notes.append(
            "No identity was a clear behavioural outlier, so no teacher was "
            "labelled; teacher metrics are unavailable."
        )
    elif teacher_confidence >= TEACHER_CONF_HIGH:
        teacher_tier = "high"
    elif teacher_confidence >= TEACHER_CONF_MED:
        teacher_tier = "medium"
    else:
        teacher_tier = "low"
        notes.append(
            "The teacher led the runner-up by a narrow margin; the teacher "
            "labelling is tentative."
        )

    overall = _worst(occupancy_tier, identity_tier, teacher_tier, coverage_tier)

    return {
        "detections": detections,
        "frames": frames,
        "identities": identities,
        "raw_tracks": raw_tracks,
        "fragmentation": round(fragmentation, 2),
        "coverage": round(coverage, 3),
        "occupied_buckets": occupied_buckets,
        "span_buckets": span_buckets,
        "concurrent_peak": concurrent_peak,
        "concurrent_typical": concurrent_typical,
        "confidence": {
            "overall": overall,
            "occupancy": occupancy_tier,
            "identity": identity_tier,
            "coverage": coverage_tier,
            "teacher": teacher_tier,
        },
        "notes": notes,
    }
