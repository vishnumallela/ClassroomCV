"""rdp_indices: the pure-python simplifier behind the permanent overlay tier.

Overlay polylines must stay faithful within epsilon while collapsing the
stationary majority of classroom tracks, so the properties under test are:
endpoints always survive, deviations > epsilon survive, deviations <= epsilon
are dropped, and degenerate inputs (short lists, closed loops) never crash.
"""

from app.geometry import rdp_indices


def test_short_inputs_returned_verbatim():
    assert rdp_indices([], 0.005) == []
    assert rdp_indices([(0.1, 0.1)], 0.005) == [0]
    assert rdp_indices([(0.1, 0.1), (0.9, 0.9)], 0.005) == [0, 1]


def test_collinear_points_collapse_to_endpoints():
    points = [(i / 10.0, i / 20.0) for i in range(11)]
    assert rdp_indices(points, 0.005) == [0, 10]


def test_spike_above_epsilon_is_kept():
    points = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.2), (0.75, 0.0), (1.0, 0.0)]
    kept = rdp_indices(points, 0.005)
    assert 2 in kept
    assert kept[0] == 0 and kept[-1] == 4


def test_jitter_below_epsilon_is_dropped():
    points = [(0.0, 0.0), (0.25, 0.001), (0.5, -0.001), (0.75, 0.002), (1.0, 0.0)]
    assert rdp_indices(points, 0.005) == [0, 4]


def test_closed_loop_keeps_far_point():
    # Identical endpoints degenerate the chord to a point; distance falls back
    # to point-to-point so the far vertex must survive.
    points = [(0.0, 0.0), (0.5, 0.5), (0.0, 0.0)]
    assert rdp_indices(points, 0.005) == [0, 1, 2]


def test_indices_ascending_and_result_within_epsilon():
    import math

    points = [
        (t / 100.0, 0.3 + 0.1 * math.sin(t / 6.0) + (0.02 if t == 50 else 0.0))
        for t in range(101)
    ]
    kept = rdp_indices(points, 0.005)
    assert kept == sorted(set(kept))
    assert kept[0] == 0 and kept[-1] == 100

    # every dropped point stays within epsilon of its simplified segment
    for a, b in zip(kept, kept[1:]):
        ax, ay = points[a]
        bx, by = points[b]
        dx, dy = bx - ax, by - ay
        chord = math.hypot(dx, dy)
        for i in range(a + 1, b):
            px, py = points[i]
            dist = abs(dx * (ay - py) - (ax - px) * dy) / chord
            assert dist <= 0.005
