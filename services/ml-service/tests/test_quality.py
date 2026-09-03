"""Unit tests for the additive data-quality assessment (app/quality.py)."""

from app import quality
from app.models import CLASS_TEACHER, Detection


def _det(ts, conf=0.9, x=0.5, y=0.5, w=0.05, h=0.1):
    return Detection(
        video_ts_ms=ts,
        cls=CLASS_TEACHER,
        bbox={"x": x, "y": y, "w": w, "h": h},
        conf=conf,
        track_no=1,
    )


def _lesson(frames=120, step=1000, conf=0.9):
    """The teacher found in every sampled frame of a clean lesson."""
    return [_det(t * step, conf=conf) for t in range(frames)]


def test_clean_lesson_is_high_confidence():
    dets = _lesson()
    r = quality.assess(dets, sampled_frames=len(dets), duration_ms=120_000,
                       teacher_confidence=0.9)
    assert r["coverage"] == 1.0
    assert r["breaks"] == 0
    assert r["confidence"]["overall"] == "high"
    assert r["confidence"]["coverage"] == "high"
    assert r["confidence"]["continuity"] == "high"
    assert r["confidence"]["teacher"] == "high"
    assert r["notes"] == []


def test_low_coverage_is_flagged():
    """She is found in a third of the frames: durations become a floor."""
    dets = _lesson(frames=40)
    r = quality.assess(dets, sampled_frames=120, duration_ms=120_000,
                       teacher_confidence=0.9)
    assert r["coverage"] < quality.COVERAGE_LOW
    assert r["confidence"]["coverage"] == "low"
    assert r["confidence"]["overall"] == "low"
    assert any("undercount" in n for n in r["notes"])


def test_broken_timeline_downgrades_continuity():
    """Many gaps means the entry/exit count is inflated by dropouts."""
    dets = []
    for block in range(25):
        base = block * 60_000
        dets += [_det(base + i * 1000) for i in range(10)]
    r = quality.assess(dets, sampled_frames=len(dets), duration_ms=1_500_000,
                       teacher_confidence=0.9)
    assert r["breaks"] > quality.BREAKS_NOISY
    assert r["confidence"]["continuity"] == "low"
    assert any("upper bound" in n for n in r["notes"])


def test_low_detection_confidence_is_tentative():
    dets = _lesson(conf=0.3)
    r = quality.assess(dets, sampled_frames=len(dets), duration_ms=120_000,
                       teacher_confidence=0.4)
    assert r["confidence"]["teacher"] == "low"
    assert any("tentative" in n for n in r["notes"])


def test_no_teacher_is_low_and_does_not_crash():
    r = quality.assess([], sampled_frames=100, duration_ms=120_000,
                       teacher_confidence=None)
    assert r["detections"] == 0
    assert r["coverage"] == 0.0
    assert r["confidence"]["overall"] == "low"
    assert any("never detected" in n for n in r["notes"])


def test_empty_lesson_does_not_divide_by_zero():
    r = quality.assess([], sampled_frames=0, duration_ms=0, teacher_confidence=None)
    assert r["coverage"] == 0.0
    assert r["breaks"] == 0


# --------------------------------------------------------------------------- #
# Attribution: was there one adult to follow at all?
# --------------------------------------------------------------------------- #


def test_one_adult_scores_high_attribution():
    dets = _lesson()
    r = quality.assess(dets, sampled_frames=len(dets), duration_ms=120_000,
                       teacher_confidence=0.9)
    assert r["confidence"]["attribution"] == "high"
    assert r["multiple_adults_detected"] is False
    assert r["max_simultaneous_adults"] == 1
    assert r["co_presence_ms"] == 0


def test_multiple_adults_drags_overall_down_from_an_otherwise_perfect_lesson():
    """The case the signal exists for, stated as bluntly as it can be.

    Every other input here is the clean lesson from the first test in this
    file: full coverage, no breaks, 0.9 confidence. Without the attribution
    tier this scores "high" on all four axes while its arrival time belongs to
    whichever adult happened to be alone first.
    """
    dets = _lesson()
    clean = quality.assess(dets, sampled_frames=len(dets), duration_ms=120_000,
                           teacher_confidence=0.9)
    assert clean["confidence"]["overall"] == "high"

    blended = quality.assess(dets, sampled_frames=len(dets), duration_ms=120_000,
                             teacher_confidence=0.9, multiple_adults=True,
                             co_presence_ms=58_000, max_simultaneous=2)

    # The three original signals are untouched — they cannot see this.
    assert blended["coverage"] == clean["coverage"]
    assert blended["breaks"] == clean["breaks"]
    assert blended["confidence"]["coverage"] == "high"
    assert blended["confidence"]["continuity"] == "high"
    assert blended["confidence"]["teacher"] == "high"
    # But the report as a whole must not read as trustworthy.
    assert blended["confidence"]["attribution"] == "low"
    assert blended["confidence"]["overall"] == "low"


def test_multiple_adults_says_which_numbers_it_withheld_and_why():
    r = quality.assess(_lesson(), sampled_frames=120, duration_ms=120_000,
                       teacher_confidence=0.9, multiple_adults=True,
                       co_presence_ms=58_000, max_simultaneous=2)
    note = next(n for n in r["notes"] if "adults" in n)
    assert "58s" in note
    assert "Not Observed" in note
    assert r["multiple_adults_detected"] is True
    assert r["max_simultaneous_adults"] == 2
    assert r["co_presence_ms"] == 58_000


def test_a_lesson_with_no_teacher_cannot_claim_high_attribution():
    """Nothing was followed, so "the right person was followed" is not a claim
    this report gets to make."""
    r = quality.assess([], sampled_frames=100, duration_ms=120_000,
                       teacher_confidence=None)
    assert r["confidence"]["attribution"] == "low"
