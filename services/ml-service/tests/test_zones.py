"""Unit tests for class-specific zoning (app/zones.py)."""

import pytest

from app import zones as Z
from app.models import CLASS_DOOR, CLASS_SCREEN, CLASS_TEACHER, Detection


def _det(ts, cls, x=0.4, y=0.4, w=0.2, h=0.15, conf=0.9):
    return Detection(
        video_ts_ms=ts, cls=cls, bbox={"x": x, "y": y, "w": w, "h": h}, conf=conf
    )


def _frames(n=40, step=1000):
    return list(range(0, n * step, step))


class TestProposeZone:
    def test_stable_object_places_a_tight_zone(self):
        ts = _frames()
        dets = [_det(t, CLASS_SCREEN, x=0.42, y=0.37, w=0.2, h=0.2) for t in ts]
        out = Z.propose_zone(dets, "board", frames_seen=len(ts))
        assert out["method"] == "rfdetr"
        assert out["polygon"] is not None
        xs = [p[0] for p in out["polygon"]]
        ys = [p[1] for p in out["polygon"]]
        assert min(xs) == pytest.approx(0.42, abs=1e-3)
        assert max(xs) == pytest.approx(0.62, abs=1e-3)
        assert min(ys) == pytest.approx(0.37, abs=1e-3)
        assert out["confidence"] > 0.8

    def test_median_ignores_a_stray_box(self):
        """One frame's false positive elsewhere must not drag the zone.

        Measured motivation: the long lesson produced two screen boxes in 5 of
        1,118 frames.
        """
        ts = _frames()
        dets = [_det(t, CLASS_SCREEN, x=0.42) for t in ts]
        dets += [_det(t, CLASS_SCREEN, x=0.95) for t in ts[:3]]
        out = Z.propose_zone(dets, "board", frames_seen=len(ts))
        assert min(p[0] for p in out["polygon"]) == pytest.approx(0.42, abs=1e-3)

    def test_rarely_seen_object_proposes_nothing(self):
        """A handful of false positives in a room with no door must not mint a
        zone: below MIN_PRESENCE the proposal abstains."""
        ts = _frames()
        dets = [_det(t, CLASS_SCREEN, x=0.4) for t in ts]
        dets += [_det(t, CLASS_DOOR, x=0.1) for t in ts[:4]]
        out = Z.propose_zone(dets, "door", frames_seen=len(ts))
        assert out["polygon"] is None

    def test_low_confidence_detections_are_not_counted(self):
        ts = _frames()
        dets = [_det(t, CLASS_SCREEN, conf=0.2) for t in ts]
        out = Z.propose_zone(dets, "board", frames_seen=len(ts))
        assert out["polygon"] is None

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            Z.propose_zone([], "ceiling")

    def test_no_detections_is_safe(self):
        out = Z.propose_zone([], "board", frames_seen=0)
        assert out["polygon"] is None
        assert out["confidence"] == 0.0


class TestGateStatic:
    def _zone(self, kind, x0, y0, x1, y1):
        return {
            "kind": kind,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        }

    def test_out_of_zone_static_detection_is_dropped(self):
        zones = [self._zone("board", 0.4, 0.35, 0.62, 0.58)]
        inside = _det(0, CLASS_SCREEN, x=0.45, y=0.4, w=0.1, h=0.1)
        outside = _det(0, CLASS_SCREEN, x=0.9, y=0.9, w=0.05, h=0.05)
        kept = Z.gate_static([inside, outside], zones)
        assert kept == [inside]

    def test_teacher_is_never_gated(self):
        """She has the run of the room: standing at the board, at the door, or
        anywhere between. Gating her would be a bug, not an optimisation."""
        zones = [
            self._zone("board", 0.4, 0.35, 0.62, 0.58),
            self._zone("door", 0.26, 0.32, 0.34, 0.64),
        ]
        far_corner = _det(0, CLASS_TEACHER, x=0.92, y=0.88, w=0.06, h=0.1)
        kept = Z.gate_static([far_corner], zones)
        assert kept == [far_corner]

    def test_jitter_at_the_zone_edge_survives(self):
        zones = [self._zone("board", 0.4, 0.35, 0.6, 0.55)]
        # Centre just outside the polygon but inside the tolerance band.
        edge = _det(0, CLASS_SCREEN, x=0.60, y=0.44, w=0.03, h=0.03)
        assert Z.gate_static([edge], zones) == [edge]

    def test_unconfigured_room_gates_nothing(self):
        """With no zones there is nothing to gate against, and gating everything
        would stop the room ever proposing a zone in the first place."""
        dets = [_det(0, CLASS_SCREEN, x=0.9), _det(0, CLASS_DOOR, x=0.1)]
        assert Z.gate_static(dets, []) == dets

    def test_a_zone_for_one_kind_does_not_gate_the_other(self):
        zones = [self._zone("board", 0.4, 0.35, 0.6, 0.55)]
        door_anywhere = _det(0, CLASS_DOOR, x=0.05, y=0.5)
        assert Z.gate_static([door_anywhere], zones) == [door_anywhere]
