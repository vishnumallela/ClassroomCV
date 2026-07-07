"""Unit tests for role scoring and the teacher-margin rule."""

from app.models import Detection
from app.roles import (
    IdentityFeatures,
    assign_roles,
    composite_score,
    compute_features,
)


def _f(track_no, standing, movement, presence, board=0.0):
    return IdentityFeatures(
        track_no=track_no,
        first_ms=0,
        last_ms=60_000,
        standing_ratio=standing,
        movement=movement,
        presence_ratio=presence,
        board_proximity=board,
        raw_track_ids=[track_no],
    )


def test_clear_teacher_vs_seated_student():
    teacher = _f(1, standing=0.9, movement=0.8, presence=0.9)
    student = _f(2, standing=0.1, movement=0.05, presence=0.9)
    roles = assign_roles([teacher, student], has_board=False)
    assert roles[1][0] == "teacher"
    assert roles[2][0] == "student"
    assert roles[1][1] is not None and 0.0 < roles[1][1] <= 1.0
    assert roles[2][1] is not None


def test_no_teacher_when_margin_too_small():
    a = _f(1, standing=0.5, movement=0.5, presence=0.9)
    b = _f(2, standing=0.5, movement=0.5, presence=0.88)
    roles = assign_roles([a, b], has_board=False)
    assert roles[1] == ("unknown", None)
    assert roles[2] == ("unknown", None)


def test_single_strong_identity_becomes_teacher():
    roles = assign_roles([_f(1, standing=0.8, movement=0.5, presence=0.9)])
    assert roles[1][0] == "teacher"


def test_single_weak_identity_stays_unknown():
    # implicit runner-up score is 0, so the margin equals the composite score
    roles = assign_roles([_f(1, standing=0.0, movement=0.0, presence=0.1)])
    assert roles[1] == ("unknown", None)


def test_empty_features_no_crash():
    assert assign_roles([]) == {}


def test_board_proximity_breaks_tie_when_board_zone_given():
    at_board = _f(1, standing=0.5, movement=0.5, presence=0.5, board=1.0)
    away = _f(2, standing=0.5, movement=0.5, presence=0.5, board=0.0)
    assert composite_score(at_board, True) > composite_score(away, True)
    roles = assign_roles([at_board, away], has_board=True)
    assert roles[1][0] == "teacher"
    assert roles[2][0] == "student"
    # without the board signal they tie -> unknown
    roles_no_board = assign_roles([at_board, away], has_board=False)
    assert roles_no_board[1][0] == "unknown"


def test_compute_features_movement_and_ratios():
    dets = [
        Detection(0, 4, {"x": 0.10, "y": 0.2, "w": 0.1, "h": 0.4}, 0.9, True, False),
        Detection(1_000, 4, {"x": 0.20, "y": 0.2, "w": 0.1, "h": 0.4}, 0.9, False, False),
    ]
    feats = compute_features({1: dets}, duration_ms=10_000)
    assert len(feats) == 1
    f = feats[0]
    assert f.track_no == 1
    assert f.standing_ratio == 0.5
    # movement is the SPATIAL RANGE of the bbox-center trajectory (not path
    # length / velocity): center moved from x=0.15 to x=0.25 -> x-range 0.1,
    # y-range 0. 0.1 / MOVEMENT_RANGE_NORM (0.4) = 0.25.
    assert abs(f.movement - 0.25) < 1e-9
    assert abs(f.presence_ratio - 0.1) < 1e-9
    assert (f.first_ms, f.last_ms) == (0, 1_000)


def test_all_seated_room_yields_no_teacher():
    """Every identity is seated (low standing, low movement): none clears the
    absolute teacher-likeness floor, so the whole room stays 'unknown'. This is
    the all-seated / uniform-crowd (bus-scene) degradation the role rule must
    guard — a teacher is only claimed when someone is genuinely teacher-like."""
    seated = [
        _f(1, standing=0.10, movement=0.05, presence=0.90),
        _f(2, standing=0.12, movement=0.04, presence=0.85),
        _f(3, standing=0.08, movement=0.06, presence=0.80),
    ]
    roles = assign_roles(seated, has_board=False)
    assert all(role == ("unknown", None) for role in roles.values())


def test_gated_identity_cannot_claim_teacher_slot():
    """An ineligible identity (failed a candidate gate — e.g. a brief edge
    sliver) may not become teacher even with the top raw score; a lower-scoring
    but eligible identity wins the slot and the gated one is labelled student."""
    sliver = _f(1, standing=1.0, movement=1.0, presence=1.0)
    sliver.eligible = False
    real = _f(2, standing=0.85, movement=0.7, presence=0.9)
    roles = assign_roles([sliver, real], has_board=False)
    assert roles[2][0] == "teacher"
    assert roles[1][0] == "student"  # gated, not teacher


def test_short_span_fragment_is_gated_ineligible_by_compute_features():
    """compute_features marks a brief identity (below the 60s teacher span
    gate) ineligible even when it looks standing + moving, so a passer-by
    sliver cannot claim the teacher slot; a long-lived identity stays eligible."""
    long_dets = [
        Detection(
            t, 1, {"x": 0.30 + 0.30 * (t / 120_000), "y": 0.30, "w": 0.10, "h": 0.30},
            0.9, True, False,
        )
        for t in range(0, 120_001, 1_000)
    ]
    short_dets = [
        Detection(t, 2, {"x": 0.50, "y": 0.30, "w": 0.10, "h": 0.30}, 0.9, True, False)
        for t in range(0, 10_001, 1_000)
    ]
    feats = compute_features({1: long_dets, 2: short_dets}, duration_ms=200_000)
    by_no = {f.track_no: f for f in feats}
    assert by_no[1].eligible is True  # 120s span >= 60s gate
    assert by_no[2].eligible is False  # 10s span < 60s gate


def test_movement_is_spatial_range_not_jitter_path_length():
    """A seated student jittering back and forth accumulates a large path
    length but a tiny spatial extent — movement must reflect the extent so
    jitter cannot make a seated student look like a roaming teacher."""
    dets = []
    for i in range(40):
        # oscillate within a 0.02-wide band -> path length grows, range ~0.02
        x = 0.50 + (0.02 if i % 2 else 0.0)
        dets.append(
            Detection(i * 500, 3, {"x": x, "y": 0.6, "w": 0.08, "h": 0.16}, 0.8, False, False)
        )
    feats = compute_features({1: dets}, duration_ms=30_000)
    # x-range 0.02 over 0.4 = 0.05, despite ~0.8 total path length
    assert feats[0].movement < 0.1
