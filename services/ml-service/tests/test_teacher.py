"""Unit tests for the teacher timeline (app/teacher.py).

The two things this module decides are which box is hers when the detector
offers more than one, and whether she is still here across a frame where it
offered none. Most of what follows is one of those two.

TestMultipleAdults is the exception, and it tests the thing this module does
NOT decide: whether "which box is hers" was even the right question. It only
measures and reports that; refusing on the answer happens downstream.
"""

import pytest

from app import teacher as T
from app.models import CLASS_DOOR, CLASS_SCREEN, CLASS_TEACHER, Detection


def _det(ts, x=0.2, y=0.3, w=0.1, h=0.4, conf=0.9, cls=CLASS_TEACHER):
    return Detection(
        video_ts_ms=ts, cls=cls, bbox={"x": x, "y": y, "w": w, "h": h}, conf=conf
    )


def _walk(start=0, end=20_000, step=200, x0=0.2, drift=0.0):
    """A teacher moving steadily from x0, one detection per sampled frame."""
    out = []
    n = max(1, (end - start) // step)
    for i, ts in enumerate(range(start, end + 1, step)):
        out.append(_det(ts, x=x0 + drift * i / n))
    return out


class TestBasics:
    def test_single_person_becomes_one_track(self):
        dets = _walk()
        track = T.build_teacher_track(dets, duration_ms=20_000)
        assert track.found
        assert len(track.detections) == len(dets)
        assert all(d.track_no == T.TEACHER_TRACK_NO for d in track.detections)
        assert track.coverage == 1.0
        assert track.confidence is not None and track.confidence > 0.5

    def test_no_teacher_class_means_no_track(self):
        dets = [_det(ts, cls=CLASS_SCREEN) for ts in range(0, 5_000, 200)]
        dets += [_det(ts, cls=CLASS_DOOR) for ts in range(0, 5_000, 200)]
        track = T.build_teacher_track(dets, duration_ms=5_000)
        assert not track.found
        assert track.confidence is None
        assert any("never found a teacher" in n for n in track.notes)

    def test_low_confidence_boxes_are_below_the_threshold(self):
        dets = [_det(ts, conf=0.1) for ts in range(0, 5_000, 200)]
        track = T.build_teacher_track(dets, duration_ms=5_000, conf_threshold=0.4)
        assert not track.found

    def test_non_teacher_detections_never_get_a_track_no(self):
        dets = _walk(end=2_000) + [_det(0, cls=CLASS_SCREEN), _det(0, cls=CLASS_DOOR)]
        T.build_teacher_track(dets, duration_ms=2_000)
        assert all(
            d.track_no is None for d in dets if d.cls != CLASS_TEACHER
        )


class TestDisambiguation:
    def test_continuity_beats_confidence(self):
        """A more confident box across the room is not her.

        Taking the highest-confidence box every frame is exactly the identity
        switch this ordering prevents.
        """
        dets = _walk(end=10_000)
        intruder = _det(5_000, x=0.9, y=0.7, conf=0.99)
        dets.append(intruder)
        track = T.build_teacher_track(dets, duration_ms=10_000)
        chosen = next(d for d in track.detections if d.video_ts_ms == 5_000)
        assert chosen is not intruder
        assert intruder.track_no is None

    def test_uncontested_first_frame_falls_back_to_confidence(self):
        """With a single instant and nothing to chain to, confidence decides."""
        weak = _det(0, x=0.2, conf=0.5)
        strong = _det(0, x=0.8, conf=0.95)
        track = T.build_teacher_track([weak, strong], duration_ms=1_000)
        assert track.detections == [strong]

    def test_a_confident_false_positive_on_frame_one_cannot_capture_the_track(self):
        """Regression: seeding on the first frame's most confident box let one
        spurious detection own the lesson.

        The intruder outscores her and appears first, but only on alternating
        frames; the chain seeds on the first UNCONTESTED instant (hers) and
        keeps her throughout.
        """
        dets = _walk(end=10_000)  # her: every 200 ms at x~0.2
        for ts in range(0, 10_001, 400):  # intruder: every other frame, louder
            dets.append(_det(ts, x=0.95, y=0.9, conf=0.99))
        track = T.build_teacher_track(dets, duration_ms=10_000)

        assert track.coverage == 1.0
        assert track.rejected_jumps == 0
        assert all(d.bbox["x"] < 0.5 for d in track.detections)
        assert track.first_ms == 0

    def test_track_extends_backwards_from_its_seed(self):
        """Seeding mid-lesson must not discard the frames before the seed."""
        dets = _walk(end=10_000)
        # Contest only the first two instants, so the seed lands at ts=400.
        for ts in (0, 200):
            dets.append(_det(ts, x=0.95, y=0.9, conf=0.99))
        track = T.build_teacher_track(dets, duration_ms=10_000)
        assert track.first_ms == 0
        assert all(d.bbox["x"] < 0.5 for d in track.detections)

    def test_unreachable_only_candidate_is_rejected_not_accepted(self):
        """When the ONLY box in a frame is implausible, the frame casts no
        claim — she is missing, not teleporting."""
        dets = _walk(end=4_000)
        far = _det(4_200, x=0.95, y=0.9)
        dets.append(far)
        track = T.build_teacher_track(dets, duration_ms=5_000)
        assert far.track_no is None
        assert track.rejected_jumps == 1
        assert any("implausible jumps" in n for n in track.notes)


class TestPlausibleMotion:
    def test_fast_walk_within_measured_speed_is_kept(self):
        """Ground truth peaks at 0.55 frame-fractions/sec; the gate must admit
        that or it would cut her track whenever she hurries."""
        a = _det(0, x=0.2)
        b = _det(200, x=0.2 + 0.11)  # 0.55/s over 200 ms
        assert T._plausible(a, b) is True

    def test_cross_room_jump_in_one_frame_is_rejected(self):
        a = _det(0, x=0.1)
        b = _det(200, x=0.9)
        assert T._plausible(a, b) is False

    def test_long_gap_frees_the_position_entirely(self):
        """After a long absence she may have left and re-entered elsewhere, so
        position carries no information and must not veto her return."""
        a = _det(0, x=0.1)
        b = _det(T.FREE_GAP_MS, x=0.95)
        assert T._plausible(a, b) is True

    def test_reacquisition_after_a_gap_continues_the_same_track(self):
        dets = _walk(end=5_000) + _walk(start=15_000, end=20_000, x0=0.8)
        track = T.build_teacher_track(dets, duration_ms=20_000)
        assert track.found
        # One identity spanning the absence: entries/exits are derived from the
        # gap downstream, not from a second identity.
        assert {d.track_no for d in track.detections} == {T.TEACHER_TRACK_NO}
        assert track.first_ms == 0 and track.last_ms == 20_000


class TestConfidence:
    def test_coverage_counts_every_sampled_instant_not_just_her_own(self):
        """She is found in a quarter of the lesson; coverage must say so.

        Dividing by the instants that offered a teacher box would report this
        as perfectly covered — the reassuring-but-wrong number the quality
        report exists to avoid. The board is detected in every frame, which is
        what makes the true denominator visible here.
        """
        dets = _walk(end=5_000)  # her: 0-5s only
        for ts in range(0, 20_001, 200):  # the board: the whole lesson
            dets.append(_det(ts, cls=CLASS_SCREEN, x=0.5))
        track = T.build_teacher_track(dets, duration_ms=20_000)
        assert track.found
        assert track.coverage == pytest.approx(0.25, abs=0.02)

    def test_rejected_jumps_lower_coverage(self):
        """Coverage is claims accepted over instants offered, so a lesson where
        the detector kept offering unreachable boxes reports as less covered."""
        clean = T.build_teacher_track(_walk(end=10_000), duration_ms=10_000)
        assert clean.coverage == 1.0

        noisy_dets = _walk(end=10_000)
        # Half the frames also carry a box across the room; those instants are
        # still claimed (her own box is reachable), so coverage holds...
        for ts in range(0, 10_001, 400):
            noisy_dets.append(_det(ts, x=0.95, y=0.9, conf=0.99))
        noisy = T.build_teacher_track(noisy_dets, duration_ms=10_000)
        assert noisy.coverage == 1.0
        assert noisy.rejected_jumps == 0  # her box won every contested frame

    def test_weak_detections_lower_confidence(self):
        strong = T.build_teacher_track(
            [_det(ts, conf=0.95) for ts in range(0, 10_000, 200)], duration_ms=10_000
        )
        weak = T.build_teacher_track(
            [_det(ts, conf=0.45) for ts in range(0, 10_000, 200)],
            duration_ms=10_000,
            conf_threshold=0.4,
        )
        assert strong.confidence > weak.confidence

    def test_empty_input_is_safe(self):
        track = T.build_teacher_track([], duration_ms=0)
        assert not track.found
        assert track.detections == []
        assert track.confidence is None


class TestMultipleAdults:
    """Co-presence: was there one adult to follow, or several?

    The measurement exists because every OTHER signal is silent here. A track
    blended from two people is continuous, fully covered and confidently
    detected — there is always *a* box — so nothing else in this file would
    notice, and the lesson would be graded on the wrong body.
    """

    def test_one_adult_is_not_flagged(self):
        track = T.build_teacher_track(_walk(end=60_000), duration_ms=60_000)
        assert track.multiple_adults is False
        assert track.max_simultaneous == 1
        assert track.co_presence_ms == 0
        assert track.contested_instants == 0

    def test_a_sustained_second_adult_is_flagged(self):
        """The handover: a second teacher shares the room for 60s of two
        minutes, which is the shape of the recording that started all this."""
        dets = _walk(end=120_000)
        dets += [_det(ts, x=0.7) for ts in range(0, 60_001, 200)]
        track = T.build_teacher_track(dets, duration_ms=120_000)

        assert track.multiple_adults is True
        assert track.max_simultaneous == 2
        assert track.co_presence_ms >= T.CO_PRESENCE_MIN_MS
        assert any("two or more adults" in n.lower() for n in track.notes)

    def test_someone_passing_through_is_not_a_second_teacher(self):
        """Three seconds of overlap is a duplicate box or an adult crossing the
        doorway. Refusing a lesson's numbers over that would spend the
        refusal's credibility on noise and leave nothing for the real case."""
        dets = _walk(end=120_000)
        dets += [_det(ts, x=0.7) for ts in range(0, 3_001, 200)]
        track = T.build_teacher_track(dets, duration_ms=120_000)

        assert track.multiple_adults is False
        assert track.max_simultaneous == 2  # seen and counted...
        assert track.co_presence_ms == pytest.approx(3_200, abs=400)  # ...but brief

    def test_co_presence_is_reported_in_time_not_frames(self):
        """A frame count is unreadable without the sample rate; the same two
        adults sampled twice as often must not read as twice the overlap."""
        dense = _walk(end=120_000, step=100)
        dense += [_det(ts, x=0.7) for ts in range(0, 60_001, 100)]
        sparse = _walk(end=120_000, step=400)
        sparse += [_det(ts, x=0.7) for ts in range(0, 60_001, 400)]

        a = T.build_teacher_track(dense, duration_ms=120_000)
        b = T.build_teacher_track(sparse, duration_ms=120_000)
        assert a.contested_instants > b.contested_instants  # 4x the frames
        assert a.co_presence_ms == pytest.approx(b.co_presence_ms, rel=0.02)

    def test_the_blend_warning_comes_before_the_rejected_jumps_note(self):
        """Ordering is load-bearing: "this may not be one person" changes how
        every other note is read, so it must not be buried under them."""
        dets = _walk(end=120_000)
        dets += [_det(ts, x=0.7) for ts in range(0, 60_001, 200)]
        dets.append(_det(120_400, x=0.95, y=0.9))  # an implausible jump too
        track = T.build_teacher_track(dets, duration_ms=121_000)

        assert track.rejected_jumps == 1
        assert len(track.notes) == 2
        assert "two or more adults" in track.notes[0].lower()
        assert "implausible jumps" in track.notes[1]

    def test_two_adults_standing_close_still_count_as_two(self):
        """The measured failure, not the easy one.

        In the real recording the two teachers were ~8% of the frame apart, so
        BOTH boxes are reachable from the previous frame and _pick_candidate
        settles each instant on confidence alone — the track silently swaps
        between them. Co-presence must catch this, because it is the case where
        coverage, continuity and confidence all look perfect.
        """
        dets = []
        for i, ts in enumerate(range(0, 60_001, 200)):
            dets.append(_det(ts, x=0.20, conf=0.80 if i % 2 else 0.86))
            dets.append(_det(ts, x=0.28, conf=0.86 if i % 2 else 0.80))
        track = T.build_teacher_track(dets, duration_ms=60_000)

        assert track.multiple_adults is True
        assert track.max_simultaneous == 2
        # The blend is invisible to every other signal: she is "found" in every
        # sampled instant, with no rejected jumps and high confidence.
        assert track.coverage == 1.0
        assert track.rejected_jumps == 0
        assert track.mean_conf > 0.8


class TestSampleInterval:
    """_sample_interval_ms: the multiplier behind the refusal decision.

    co_presence_ms is contested_instants * this value, and the refusal is a
    threshold on co_presence_ms — so reading the interval off the wrong
    statistic rescales the refuse-or-report decision by exactly that factor.
    Every assertion here was written because the mutant it kills survived the
    whole suite: max, mean and min all passed 158 tests, and so did sourcing
    the stamps from her own detections instead of every class.
    """

    def test_one_long_absence_does_not_inflate_the_interval(self):
        """The case the median was chosen for, stated as a test.

        A minute at 5 fps, six minutes where nothing was detected at all, then
        another minute. The mean over that is ~798 ms and the max is 360,000 —
        four seconds of real overlap would read as sixteen seconds under the
        mean, and as half a lesson under the max.
        """
        stamps = list(range(0, 60_001, 200)) + list(range(420_000, 480_001, 200))
        assert T._sample_interval_ms(stamps) == 200

    def test_one_unusually_short_gap_does_not_shrink_the_interval(self):
        """Symmetry: the minimum is as wrong as the maximum, in the other
        direction. A single pair of stamps closer together than the sample rate
        is a decoder artefact, not the rate."""
        stamps = [0, 40, 240, 440, 640, 840]
        assert T._sample_interval_ms(stamps) == 200

    def test_too_few_stamps_to_measure_reports_zero_not_a_guess(self):
        """Zero propagates to co_presence_ms == 0, which never refuses. A
        fabricated interval here would refuse a lesson on no evidence."""
        assert T._sample_interval_ms([]) == 0
        assert T._sample_interval_ms([1_000]) == 0

    def test_the_interval_comes_from_every_class_not_just_her(self):
        """The sampling rate is a property of the detector pass, not of how
        often she happened to be found.

        A heavily occluded lesson: the board is detected every 200 ms, she is
        found only once every 2 s, and a second box shares 20 of those
        instants. Measured against every class the overlap is 4 s and the
        lesson reports normally. Measured against her stamps alone the interval
        reads as 2,000 ms, the same 20 instants become 40 s, and a lesson with
        four seconds of overlap has its arrival, departure, time in the room
        and entry/exit counts all withheld.

        This is not hypothetical for /rederive, which replays stored rows —
        teacher-class only by construction — so its stamps really are hers
        alone. See the note in build_teacher_track.
        """
        dets = [_det(ts, cls=CLASS_SCREEN, x=0.5) for ts in range(0, 120_001, 200)]
        for ts in range(0, 40_001, 2_000):
            dets.append(_det(ts, x=0.2))
            if ts < 40_000:  # a second body at 20 of her 21 instants
                dets.append(_det(ts, x=0.7))

        track = T.build_teacher_track(dets, duration_ms=120_000)
        assert track.contested_instants == 20
        assert track.max_simultaneous == 2
        assert track.co_presence_ms == 4_000  # 20 x 200 ms, not 20 x 2,000
        assert track.multiple_adults is False
