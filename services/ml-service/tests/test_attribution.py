"""Unit tests for attribution (app/attribution.py): pieces -> people -> the one
the lesson assesses.

Descriptors here are synthetic 16-float histograms: two "colours" that are far
apart, and small noise on top so nothing is byte-identical. Every threshold in
the module was set from the two real lessons; these tests pin the BEHAVIOUR
those thresholds were set to produce, on shapes small enough to read.
"""

import random

import pytest

from app import attribution as X
from app import appearance as A
from app.models import CLASS_TEACHER, Detection
from app.teacher import Segment


def _colour(seed: str) -> list[float]:
    """A stable, distinct descriptor per name (each part L1-normalised)."""
    rng = random.Random(seed)
    parts = []
    for n in (A.H_BINS, A.S_BINS, A.V_BINS):
        raw = [rng.random() ** 3 for _ in range(n)]
        t = sum(raw)
        parts.extend(v / t for v in raw)
    return parts


CREAM = _colour("cream-stripes")
BLACK = _colour("black-kurta")


def _noisy(desc: list[float], seed: int, amp: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    out = [max(0.0, v + rng.uniform(-amp, amp)) for v in desc]
    # keep each part normalised
    res = []
    for lo, hi in ((0, A.H_BINS), (A.H_BINS, A.H_BINS + A.S_BINS), (A.H_BINS + A.S_BINS, A.DIMS)):
        part = out[lo:hi]
        t = sum(part) or 1.0
        res.extend(v / t for v in part)
    return res


def _det(ts, x=0.5, y=0.4, w=0.08, h=0.3, conf=0.85, app=None, seed=0):
    return Detection(
        video_ts_ms=ts, cls=CLASS_TEACHER, bbox={"x": x, "y": y, "w": w, "h": h},
        conf=conf, app=(_noisy(app, seed + ts) if app is not None else None),
    )


def _seg(t0, t1, x=0.5, app=CREAM, step=200, **kw):
    return Segment([_det(ts, x=x, app=app, **kw) for ts in range(t0, t1 + 1, step)])


class TestSplit:
    def test_a_segment_that_changes_person_is_cut_there(self):
        """Boxes look cream for 20 s then black for 20 s: one cut, at the change."""
        dets = [_det(ts, app=CREAM) for ts in range(0, 20_001, 200)]
        dets += [_det(ts, app=BLACK) for ts in range(20_200, 40_001, 200)]
        pieces, n = X.split_at_changes([Segment(dets)])
        assert n == 1 and len(pieces) == 2
        assert abs(pieces[0].last_ms - 20_000) <= X.SPLIT_WINDOW * 200
        assert all(d.app is not None for d in pieces[1].detections)

    def test_one_person_is_not_cut(self):
        """Forty seconds of one colour with per-frame noise: no cut."""
        pieces, n = X.split_at_changes([_seg(0, 40_000)])
        assert n == 0 and len(pieces) == 1

    def test_a_brief_occlusion_is_not_a_person_change(self):
        """Two seconds of a different colour inside 40 s of one: the window
        is wider than the blip, so nothing crosses the threshold."""
        dets = [_det(ts, app=(BLACK if 20_000 <= ts <= 22_000 else CREAM)) for ts in range(0, 40_001, 200)]
        _, n = X.split_at_changes([Segment(dets)])
        assert n == 0


class TestLink:
    def test_pieces_across_a_long_gap_link_by_appearance(self):
        """She steps out for a minute. Position says nothing after five
        seconds; appearance links the two pieces."""
        a = _seg(0, 20_000, x=0.2)
        b = _seg(80_000, 100_000, x=0.8)
        people = X.link_people([a, b])
        assert len(people) == 1 and len(people[0].segments) == 2

    def test_a_different_person_across_a_long_gap_stays_separate(self):
        a = _seg(0, 20_000, x=0.2, app=CREAM)
        b = _seg(80_000, 100_000, x=0.8, app=BLACK)
        assert len(X.link_people([a, b])) == 2

    def test_adjacent_pieces_link_by_position_despite_a_muddied_descriptor(self):
        """From the real clip: a student in green stood in front of the black
        teacher for 40 s and her descriptor went strange. The piece before ends
        0.4 s earlier at the same spot, so continuity carries the link that
        appearance would have refused."""
        a = _seg(0, 15_000, x=0.5, app=BLACK)
        b = _seg(15_400, 40_000, x=0.51, app=CREAM)  # "wrong" colour, right place
        people = X.link_people([a, b])
        assert len(people) == 1

    def test_a_piece_with_too_few_clean_boxes_cannot_speak_for_itself(self):
        """The seated teacher's head-only lane: a handful of boxes over the
        height floor must not manufacture a descriptor."""
        head = Segment([_det(ts, h=0.12, app=CREAM) for ts in range(0, 20_001, 200)])
        for d in head.detections[:5]:
            d.bbox["h"] = 0.2  # five "clean" boxes out of 101
        assert X._mean_app(head.detections) is None

    def test_a_head_lane_nested_in_a_torso_lane_is_one_body(self):
        torso = Segment([_det(ts, x=0.40, y=0.62, w=0.08, h=0.26) for ts in range(0, 30_001, 200)])
        head = Segment([_det(ts, x=0.41, y=0.58, w=0.05, h=0.12) for ts in range(2_000, 25_001, 200)])
        kept = X.merge_duplicate_lanes([torso, head])
        assert len(kept) == 1
        # the larger box wins at shared instants
        assert all(d.bbox["h"] == 0.26 for d in kept[0].detections if 2_000 <= d.video_ts_ms <= 25_000)

    def test_two_adults_side_by_side_are_not_merged(self):
        """Fifteen seconds within 0.05 of each other, but different columns."""
        a = Segment([_det(ts, x=0.50, w=0.07) for ts in range(0, 30_001, 200)])
        b = Segment([_det(ts, x=0.58, w=0.07) for ts in range(5_000, 20_001, 200)])
        assert len(X.merge_duplicate_lanes([a, b])) == 2


class TestAttribute:
    def _handover(self, tail_ms):
        """Outgoing adult 0->200 s; incoming arrives at 150 s and stays to the end."""
        out = _seg(0, 200_000, x=0.4, app=CREAM)
        inc = _seg(150_000, 200_000 + tail_ms, x=0.6, app=BLACK)
        return [out, inc], 200_000 + tail_ms

    def test_one_adult_is_hers_with_high_confidence(self):
        att = X.attribute([_seg(0, 60_000)], 60_000)
        assert att.chosen is not None and att.confidence == "high"
        assert att.chosen.track_no == 1

    def test_the_adult_who_stays_is_attributed_over_the_one_who_leaves(self):
        segs, dur = self._handover(tail_ms=120_000)
        att = X.attribute(segs, dur, period_ms=(0, dur))
        assert att.chosen is not None
        assert att.chosen.first_ms == 150_000  # the incoming adult
        assert att.confidence == "high"
        assert "left while this one remained" in att.reason
        # the outgoing adult had MORE presence; that is exactly what must not decide it
        out = next(c for c in att.candidates if c.first_ms == 0)
        inc = next(c for c in att.candidates if c.first_ms == 150_000)
        assert out.in_period_ms > inc.in_period_ms and out.handed_over and not inc.handed_over

    def test_a_short_tail_names_the_right_adult_but_only_at_medium(self):
        """The 6-minute trim of the real handover: 24 s of the incoming teacher
        alone is enough to say who, not enough to grade her on."""
        segs, dur = self._handover(tail_ms=24_000)
        att = X.attribute(segs, dur, period_ms=(0, dur))
        assert att.chosen is not None and att.chosen.first_ms == 150_000
        assert att.confidence == "medium"
        assert "too little to grade" in att.reason

    def test_two_adults_present_to_the_end_with_similar_presence_is_undetermined(self):
        a = _seg(0, 100_000, x=0.3, app=CREAM)
        b = _seg(5_000, 100_000, x=0.7, app=BLACK)
        att = X.attribute([a, b], 100_000, period_ms=(0, 100_000))
        assert att.chosen is None and att.confidence == "low"
        assert "Not Observed" in att.reason

    def test_a_clear_presence_lead_decides_when_nobody_handed_over(self):
        a = _seg(0, 100_000, x=0.3, app=CREAM)
        b = _seg(60_000, 100_000, x=0.7, app=BLACK)  # both present at the end
        att = X.attribute([a, b], 100_000, period_ms=(0, 100_000))
        assert att.chosen is not None and att.chosen.first_ms == 0
        assert att.confidence == "high"

    def test_without_a_timetable_confidence_is_capped_at_medium(self):
        a = _seg(0, 100_000, x=0.3, app=CREAM)
        b = _seg(60_000, 100_000, x=0.7, app=BLACK)
        att = X.attribute([a, b], 100_000, period_ms=None)
        assert att.chosen is not None and att.confidence == "medium"
        assert att.period_known is False

    def test_presence_outside_the_period_does_not_count(self):
        """An adult present only before the bell has no claim on the period."""
        before = _seg(0, 50_000, x=0.3, app=CREAM)
        during = _seg(70_000, 200_000, x=0.7, app=BLACK)
        att = X.attribute([before, during], 200_000, period_ms=(60_000, 200_000))
        assert att.chosen is not None and att.chosen.first_ms == 70_000
        assert next(c for c in att.candidates if c.first_ms == 0).in_period_ms == 0

    def test_the_chosen_person_is_track_one_and_the_rest_follow(self):
        segs, dur = self._handover(tail_ms=120_000)
        att = X.attribute(segs, dur, period_ms=(0, dur))
        assert [p.track_no for p in sorted(att.persons, key=lambda p: p.track_no)] == [1, 2]
        assert att.chosen.track_no == 1
        assert att.as_report()["chosen_track_no"] == 1


WHITE = _colour("white-shirt")


def _swapped_crossing(gap_ms=0):
    """What the tracker hands over after an OCCLUDED crossing, as on the full
    handover recording at 2381 s: lane X carries the standing teacher (black)
    and then the walker (white); lane Y carries the walker and then the
    teacher. The walker passes within reach of the teacher at the swap."""
    t = 40_000
    x = Segment(
        [_det(ts, x=0.50, app=BLACK) for ts in range(0, t, 200)]
        + [_det(ts, x=0.50 - 0.002 * (ts - t) / 200, app=WHITE) for ts in range(t, 80_001, 200)]
    )
    y = Segment(
        [_det(ts, x=0.62 - 0.003 * ts / 200, app=WHITE) for ts in range(20_000, t - gap_ms, 200)]
        + [_det(ts, x=0.51, app=BLACK) for ts in range(t + gap_ms, 80_001, 200)]
    )
    return x, y, t


class TestSwaps:
    def test_an_occluded_crossing_the_tracker_swapped_is_undone(self):
        x, y, t = _swapped_crossing(gap_ms=1_000)
        segs, swaps = X.resolve_swaps([x, y])
        assert swaps == 1
        black = next(s for s in segs if s.detections[0].bbox["x"] == 0.50)
        white = next(s for s in segs if s is not black)
        # each lane is one person again: the descriptors on either side of the
        # swap now agree
        for s in (black, white):
            before = A.mean_descriptor([d.app for d in s.detections if d.video_ts_ms < t])
            after = A.mean_descriptor([d.app for d in s.detections if d.video_ts_ms >= t])
            assert A.distance(before, after) < 0.1
        assert black.first_ms == 0 and black.last_ms == 80_000

    def test_a_clean_crossing_is_left_to_the_motion_model(self):
        """Velocity carried the walker through; each lane is one colour
        throughout, so re-pairing would only make things worse."""
        walker = Segment([_det(ts, x=0.8 - 0.003 * ts / 200, app=WHITE) for ts in range(0, 40_001, 200)])
        still = Segment([_det(ts, x=0.5, app=BLACK) for ts in range(0, 40_001, 200)])
        before = [list(walker.detections), list(still.detections)]
        _, swaps = X.resolve_swaps([walker, still])
        assert swaps == 0
        assert [walker.detections, still.detections] == before

    def test_one_lane_is_never_examined(self):
        seg = _seg(0, 60_000, app=BLACK)
        _, swaps = X.resolve_swaps([seg])
        assert swaps == 0

    def test_a_student_recolouring_one_lane_is_not_a_swap(self):
        """A student in front of the teacher muddies HER window towards the
        other adult's colour. One cross term drops, the other does not, so the
        pairing as tracked still wins."""
        teacher = Segment(
            [_det(ts, x=0.50, app=BLACK) for ts in range(0, 40_000, 200)]
            + [_det(ts, x=0.50, app=WHITE) for ts in range(40_000, 60_001, 200)]  # muddied
        )
        other = Segment([_det(ts, x=0.56, app=WHITE) for ts in range(20_000, 60_001, 200)])
        _, swaps = X.resolve_swaps([teacher, other])
        assert swaps == 0

    def test_attribution_names_the_teacher_across_a_swapped_crossing(self):
        """End to end: the swapped lanes would have left the teacher's last
        forty seconds on the walker's person; resolved, she is one person from
        start to finish and the walker is the one who leaves."""
        x, y, t = _swapped_crossing(gap_ms=1_000)
        # the walker leaves at 80 s; the teacher stays to 200 s
        y.detections += [_det(ts, x=0.51, app=BLACK) for ts in range(80_200, 200_001, 200)]
        att = X.attribute([x, y], 200_000, period_ms=(0, 200_000))
        assert att.swaps == 1 and att.chosen is not None
        assert att.chosen.first_ms == 0 and att.chosen.last_ms == 200_000
        assert att.confidence == "high"


class TestLateVisitor:
    def test_a_colleague_who_sits_in_for_the_last_minutes_does_not_cap_confidence(self):
        """Full recording: a colleague sat at a desk for the last five minutes
        and was last detected 12 s before the end. The teacher did not 'hold
        the room alone' for a minute — but she was present many times longer
        than anyone who left, which is the same certainty."""
        out = _seg(0, 200_000, x=0.4, app=CREAM)
        teacher = _seg(150_000, 2_000_000, x=0.6, app=BLACK)
        visitor = _seg(1_700_000, 1_988_000, x=0.2, app=WHITE)
        att = X.attribute([out, teacher, visitor], 2_000_000, period_ms=(0, 2_000_000))
        assert att.chosen is not None and att.chosen.first_ms == 150_000
        assert att.confidence == "high"
        assert "longer than anyone who left" in att.reason

    def test_a_short_lead_over_the_one_who_left_stays_medium(self):
        out = _seg(0, 200_000, x=0.4, app=CREAM)
        teacher = _seg(150_000, 400_000, x=0.6, app=BLACK)  # 250 s vs 200 s: no lead
        visitor = _seg(300_000, 388_000, x=0.2, app=WHITE)
        att = X.attribute([out, teacher, visitor], 400_000, period_ms=(0, 400_000))
        assert att.chosen is not None and att.confidence == "medium"


class TestLeftAt:
    def test_departure_is_the_end_of_the_run_containing_the_bell(self):
        """Two light-dressed adults can link across a long gap; the previous
        teacher's departure must still be when SHE left, not the colleague's
        last sighting."""
        out = Segment(
            [_det(ts, x=0.4, app=CREAM) for ts in range(0, 300_001, 200)]
            + [_det(ts, x=0.3, app=CREAM) for ts in range(2_000_000, 2_100_001, 200)]
        )
        teacher = _seg(250_000, 2_500_000, x=0.6, app=BLACK)
        att = X.attribute([out, teacher], 2_500_000, period_ms=(2_000, 2_500_000))
        c = next(c for c in att.candidates if c.first_ms == 0)
        assert c.handed_over and c.last_ms == 2_100_000 and c.left_ms == 300_000
        assert next(c for c in att.candidates if c.first_ms == 250_000).left_ms is None

    def test_a_short_absence_does_not_count_as_leaving(self):
        dets = [_det(ts, x=0.4, app=CREAM) for ts in range(0, 100_001, 200)]
        dets += [_det(ts, x=0.4, app=CREAM) for ts in range(160_000, 300_001, 200)]  # out for a minute
        assert X._left_ms(dets, 0) == 300_000

    def test_not_there_at_the_bell_means_no_departure(self):
        dets = [_det(ts, x=0.4, app=CREAM) for ts in range(60_000, 100_001, 200)]
        assert X._left_ms(dets, 0) is None
