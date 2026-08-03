"""Per-run data-quality assessment: how much can you trust these analytics?

Every dashboard number is an estimate over sampled, occluded CCTV. A school
leader reading "42 minutes at the board" deserves to know whether that figure
came from a clean, well-covered lesson or from a half-occluded camera whose
tracker fragmented the teacher into a dozen identities. This module turns the
pipeline's own internal signals into an honest, additive confidence report.
It NEVER changes a derived number — it only annotates them.

Three signals drive the report, each one a direct trust input for the three
teacher KPIs (entry/exit, board time, heatmap):

1. Coverage: over the lesson's active span (first to last detection), what
   fraction of time buckets actually contained a detected person. Low coverage
   means the camera went dark, the frame was occluded, or the model dropped
   out — presence-derived numbers are then a floor, not a measurement.

2. Fragmentation: raw tracker ids per final identity. The tracker mints a new
   id every time a person is occluded or leaves frame; the merge stage reunites
   them. A ratio near 1 means clean tracking; a high ratio means the merge did
   heavy lifting and the teacher's timeline (entries/exits, board sessions,
   heatmap path) carries more stitching error.

3. Teacher identification: the margin behind the "who is the teacher" decision
   (roles.assign_roles' role_confidence, raised by the teacher_id vote when it
   confirms). Every KPI hangs off this one label.
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

Tier = str  # "high" | "medium" | "low"


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

    notes: list[str] = []

    # --- identity-tracking confidence (fragmentation) ---------------------- #
    if fragmentation <= FRAG_CLEAN:
        identity_tier = "high"
    elif fragmentation <= FRAG_NOISY:
        identity_tier = "medium"
        notes.append(
            f"Tracker fragmented people into ~{fragmentation:.1f} ids each; "
            "the teacher's timeline is a re-id estimate."
        )
    else:
        identity_tier = "low"
        notes.append(
            f"Heavy fragmentation (~{fragmentation:.1f} ids per person): treat "
            "the teacher's entry/exit and board sessions as approximate."
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

    overall = _worst(identity_tier, teacher_tier, coverage_tier)

    return {
        "detections": detections,
        "frames": frames,
        "identities": identities,
        "raw_tracks": raw_tracks,
        "fragmentation": round(fragmentation, 2),
        "coverage": round(coverage, 3),
        "occupied_buckets": occupied_buckets,
        "span_buckets": span_buckets,
        "confidence": {
            "overall": overall,
            "identity": identity_tier,
            "coverage": coverage_tier,
            "teacher": teacher_tier,
        },
        "notes": notes,
    }
