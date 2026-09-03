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
