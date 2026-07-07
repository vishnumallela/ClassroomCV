"""Role assignment: which merged identity is the teacher.

Per identity, composite score = weighted(standing_ratio, normalized movement,
presence_duration_ratio, board_proximity if a board zone is given).

Robustness on real classroom footage (calibrated against a 4.8-min 30-person
recording where the naive absolute-margin rule yielded all-unknown):

1. CANDIDATE GATES before scoring — junk identities score near the true
   teacher and destroy any margin:
   - short-lived fragments (16s slivers of the walking teacher / passers-by),
   - frame-edge slivers whose clipped tall bboxes always read as 'standing',
   - tiny boxes far below the room's median person size.
   Gated identities can still be labelled students; they just cannot claim
   or block the teacher slot.

2. NOISE-FREE FEATURES:
   - movement = spatial range of the bbox-center trajectory (max of x/y
     extents), not path length: seated students accumulate large fake path
     from per-frame bbox jitter, but their spatial extent stays ~0.02.
   - board_proximity counts only samples that look like standing AT the
     board: center inside the polygon's x-range AND bbox bottom edge below
     the polygon bottom AND standing. (The old expanded-bbox-center test gave
     seated mid-room students a perfect 1.0 because the expanded board bbox
     covers the frame center.)

3. RELATIVE (outlier-based) teacher rule instead of a fixed absolute margin:
   the best candidate wins only when it clears an absolute floor AND leads
   the runner-up by max(TEACHER_ABS_MARGIN, TEACHER_REL_MARGIN * best).
   When nothing is genuinely teacher-like (everyone similar — e.g. a bus
   scene where the whole crowd pans together), everyone stays 'unknown' and
   analytics degrade gracefully.

4. FRAGMENT ABSORPTION: the walking teacher fragments (spatial merge cannot
   bridge a 0.6-unit jump across a 30s absence), so short unassigned
   fragments that fit inside the teacher's absence windows and sit near the
   teacher's trajectory / the board are folded back into the teacher at the
   role level (see absorbable_fragments; applied by jobs.derive_result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.geometry import expand_bbox, polygon_bbox
from app.models import Detection

# Composite weights (renormalized when no board zone is available).
W_STANDING = 0.30
W_MOVEMENT = 0.25
W_PRESENCE = 0.25
W_BOARD = 0.20

# Movement: spatial extent of the bbox-center trajectory that saturates the
# feature at 1.0 (the real teacher covers ~0.89 of the frame width; seated
# students ~0.02).
MOVEMENT_RANGE_NORM = 0.4

# --- candidate gates (teacher eligibility) ---------------------------------
# Identities seen for less than this absolute span cannot be the teacher...
MIN_TEACHER_SPAN_MS = 60_000
# ...but never demand more than this fraction of the video (short clips).
MIN_SPAN_DURATION_FRACTION = 1.0 / 3.0
# Mean bbox center within this distance of a frame edge -> clipped sliver.
EDGE_MARGIN = 0.03
# Mean bbox area below this fraction of the median identity area -> too small.
MIN_RELATIVE_AREA = 0.3

# --- robust teacher rule ----------------------------------------------------
TEACHER_MIN_SCORE = 0.5  # best must at least look teacher-like in absolute terms
TEACHER_ABS_MARGIN = 0.08  # floor on the lead over the runner-up
# Lead must also be >= this fraction of the best score. Recalibrated from 0.2
# after CLIP re-ID started attaching the teacher's SEATED return segment
# (demo: standing_ratio 0.74 -> 0.66, composite 0.766 -> 0.710, lead 26.6% ->
# 17.6%): a correct re-ID must not un-classify the teacher. Chains of student
# fragments ride the union-inflated movement+presence terms to ~0.59, so the
# guard against declaring a teacher among lookalikes still holds at 15%.
TEACHER_REL_MARGIN = 0.15

# --- board proximity / absorption -------------------------------------------
# Expansion of the board polygon bbox used when testing whether an absorbed
# fragment is "near the board".
BOARD_PROXIMITY_EXPAND = 0.15
# Teacher absence gap that opens an absorption window (matches the presence
# split threshold in events.py).
ABSORB_GAP_MS = 5_000
# Fragment mean center must pass within this distance of the teacher's
# observed trajectory (or sit near the board) to be absorbed.
ABSORB_TRAJECTORY_DIST = 0.15
# Absorbed fragments must be at least this big relative to the teacher.
ABSORB_MIN_AREA_RATIO = 0.3


@dataclass
class IdentityFeatures:
    track_no: int
    first_ms: int
    last_ms: int
    standing_ratio: float
    movement: float  # 0..1
    presence_ratio: float  # 0..1
    board_proximity: float  # 0..1 (0 when no board zone)
    raw_track_ids: list[int] = field(default_factory=list)
    span_ms: int = 0
    mean_area: float = 0.0
    mean_cx: float = 0.5
    mean_cy: float = 0.5
    eligible: bool = True  # may this identity claim the teacher slot?


def _center(d: Detection) -> tuple[float, float]:
    return d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0


def _smooth_standing(dets: list[Detection]) -> list[bool]:
    """5-sample sliding majority vote over the chronological standing flags.

    Posture changes last far longer than one second, but _is_standing flickers
    on bbox-aspect noise and keypoint dropouts (worst when the teacher writes
    back-turned). The vote removes single-sample flips before the flags feed
    standing_ratio and the at-board gate.
    """
    flags = [d.standing for d in dets]
    if len(flags) < 5:
        return flags
    smoothed: list[bool] = []
    for i in range(len(flags)):
        lo = max(0, i - 2)
        window = flags[lo : i + 3]
        smoothed.append(sum(window) * 2 > len(window))
    return smoothed


def min_teacher_span_ms(duration_ms: int) -> int:
    """Span gate: 60s absolute, scaled down for clips shorter than 3 minutes."""
    if duration_ms and duration_ms > 0:
        return min(
            MIN_TEACHER_SPAN_MS, int(duration_ms * MIN_SPAN_DURATION_FRACTION)
        )
    return MIN_TEACHER_SPAN_MS


def compute_features(
    dets_by_track: dict[int, list[Detection]],
    duration_ms: int,
    board_polygon: Optional[list[list[float]]] = None,
    raw_ids_by_track: Optional[dict[int, list[int]]] = None,
) -> list[IdentityFeatures]:
    board_x0 = board_x1 = board_y1 = None
    if board_polygon is not None:
        bx0, _by0, bx1, by1 = polygon_bbox(board_polygon)
        board_x0, board_x1, board_y1 = bx0, bx1, by1

    features: list[IdentityFeatures] = []
    for track_no in sorted(dets_by_track):
        dets = sorted(dets_by_track[track_no], key=lambda d: d.video_ts_ms)
        if not dets:
            continue
        first_ms, last_ms = dets[0].video_ts_ms, dets[-1].video_ts_ms
        n = len(dets)

        standing = _smooth_standing(dets)
        standing_ratio = sum(standing) / n

        centers = [_center(d) for d in dets]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        movement = min(1.0, max(x_range, y_range) / MOVEMENT_RANGE_NORM)

        presence_ratio = (
            min(1.0, (last_ms - first_ms) / duration_ms) if duration_ms > 0 else 0.0
        )

        board_proximity = 0.0
        if board_x0 is not None:
            at_board = 0
            for d, is_standing, (cx, _cy) in zip(dets, standing, centers):
                if (
                    is_standing
                    and board_x0 <= cx <= board_x1
                    and d.bbox["y"] + d.bbox["h"] >= board_y1
                ):
                    at_board += 1
            board_proximity = at_board / n

        raw_ids = (raw_ids_by_track or {}).get(track_no, [track_no])
        features.append(
            IdentityFeatures(
                track_no=track_no,
                first_ms=first_ms,
                last_ms=last_ms,
                standing_ratio=standing_ratio,
                movement=movement,
                presence_ratio=presence_ratio,
                board_proximity=board_proximity,
                raw_track_ids=list(raw_ids),
                span_ms=last_ms - first_ms,
                mean_area=sum(max(0.0, d.bbox["w"] * d.bbox["h"]) for d in dets) / n,
                mean_cx=sum(xs) / n,
                mean_cy=sum(ys) / n,
            )
        )

    _apply_gates(features, duration_ms)
    return features


def _apply_gates(features: list[IdentityFeatures], duration_ms: int) -> None:
    """Mark identities that may not claim the teacher slot (eligible=False)."""
    if not features:
        return
    areas = sorted(f.mean_area for f in features)
    mid = len(areas) // 2
    median_area = (
        areas[mid] if len(areas) % 2 == 1 else (areas[mid - 1] + areas[mid]) / 2.0
    )
    min_span = min_teacher_span_ms(duration_ms)
    for f in features:
        near_edge = (
            f.mean_cx <= EDGE_MARGIN
            or f.mean_cx >= 1.0 - EDGE_MARGIN
            or f.mean_cy <= EDGE_MARGIN
            or f.mean_cy >= 1.0 - EDGE_MARGIN
        )
        too_small = median_area > 0 and f.mean_area < MIN_RELATIVE_AREA * median_area
        f.eligible = f.span_ms >= min_span and not near_edge and not too_small


def composite_score(f: IdentityFeatures, has_board: bool) -> float:
    score = (
        W_STANDING * f.standing_ratio
        + W_MOVEMENT * f.movement
        + W_PRESENCE * f.presence_ratio
    )
    total_w = W_STANDING + W_MOVEMENT + W_PRESENCE
    if has_board:
        score += W_BOARD * f.board_proximity
        total_w += W_BOARD
    return score / total_w


def assign_roles(
    features: list[IdentityFeatures],
    has_board: bool = False,
    min_score: float = TEACHER_MIN_SCORE,
    abs_margin: float = TEACHER_ABS_MARGIN,
    rel_margin: float = TEACHER_REL_MARGIN,
) -> dict[int, tuple[str, Optional[float]]]:
    """Map track_no -> (role, role_confidence).

    Only gate-eligible identities may claim the teacher slot. The best
    eligible candidate becomes teacher when:
      - its score clears the absolute floor `min_score` (teacher-like at all;
        with a single identity the implicit runner-up score is 0), AND
      - its lead over the eligible runner-up is an OUTLIER lead:
        best - second >= max(abs_margin, rel_margin * best).
    Otherwise every identity stays 'unknown' with null confidence and the
    analytics degrade gracefully downstream.
    """
    if not features:
        return {}

    scored = sorted(
        ((composite_score(f, has_board), f) for f in features),
        key=lambda x: x[0],
        reverse=True,
    )
    candidates = [(s, f) for s, f in scored if f.eligible]

    teacher: Optional[IdentityFeatures] = None
    teacher_conf: Optional[float] = None
    best_score = 0.0
    if candidates:
        best_score, best_f = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        margin_val = best_score - second_score
        required = max(abs_margin, rel_margin * best_score)
        if best_score >= min_score and margin_val >= required:
            teacher = best_f
            teacher_conf = round(min(1.0, 0.5 + margin_val), 4)

    if teacher is None:
        return {f.track_no: ("unknown", None) for _, f in scored}

    roles: dict[int, tuple[str, Optional[float]]] = {
        teacher.track_no: ("teacher", teacher_conf)
    }
    for score, f in scored:
        if f.track_no == teacher.track_no:
            continue
        conf = round(min(1.0, max(0.0, 0.5 + (best_score - score))), 4)
        roles[f.track_no] = ("student", conf)
    return roles


# --------------------------------------------------------------------------- #
# Teacher fragment absorption
# --------------------------------------------------------------------------- #


def _presence_windows(
    ts_sorted: list[int], duration_ms: int, gap_ms: int
) -> list[tuple[int, int, list[int]]]:
    """Teacher ABSENCE windows: (start, end, [edge indices into ts_sorted]).

    Windows are the gaps >= gap_ms between consecutive teacher samples, plus
    the leading [0, first_ts] and trailing [last_ts, duration] stretches. The
    edge indices point at the teacher samples bounding the window (used to
    fetch the teacher's position when it vanished/reappeared).
    """
    if not ts_sorted:
        return []
    windows: list[tuple[int, int, list[int]]] = []
    if ts_sorted[0] >= gap_ms:
        windows.append((0, ts_sorted[0], [0]))
    for i in range(1, len(ts_sorted)):
        if ts_sorted[i] - ts_sorted[i - 1] >= gap_ms:
            windows.append((ts_sorted[i - 1], ts_sorted[i], [i - 1, i]))
    if duration_ms > 0 and duration_ms - ts_sorted[-1] >= gap_ms:
        windows.append((ts_sorted[-1], duration_ms, [len(ts_sorted) - 1]))
    return windows


def absorbable_fragments(
    teacher_no: int,
    features: list[IdentityFeatures],
    dets_by_track: dict[int, list[Detection]],
    duration_ms: int,
    board_polygon: Optional[list[list[float]]] = None,
    door_polygons: Optional[list[list[list[float]]]] = None,
) -> list[int]:
    """track_nos of fragments to fold into the teacher identity.

    Two kinds of fragment qualify:

    BLIP: short (span below the teacher span gate) and entirely inside one
    teacher ABSENCE window, near the board or the teacher's trajectory. This
    recovers mid-absence tracker fragments.

    CONTINUATION: STARTS inside an absence window and keeps going after it,
    without temporally overlapping the teacher. This is the teacher walking
    back in: the tracker gives her a fresh id at the door, the merge cannot
    bridge the long spatial jump, and without this rule the returning teacher
    is labelled a student for the rest of the lesson. Its START position must
    be near a door, the board, or the teacher's trajectory; the whole-span
    mean is useless because she then walks the room.

    All fragments must not be frame-edge slivers, must be at least
    ABSORB_MIN_AREA_RATIO of the teacher's area, and may not overlap an
    already-absorbed fragment (two co-present fragments cannot both be her).
    """
    teacher_f = next((f for f in features if f.track_no == teacher_no), None)
    teacher_dets = dets_by_track.get(teacher_no)
    if teacher_f is None or not teacher_dets:
        return []

    teacher_dets = sorted(teacher_dets, key=lambda d: d.video_ts_ms)
    ts_sorted = [d.video_ts_ms for d in teacher_dets]
    centers = [_center(d) for d in teacher_dets]
    windows = _presence_windows(ts_sorted, duration_ms, ABSORB_GAP_MS)
    if not windows:
        return []

    board_box = None
    if board_polygon is not None:
        board_box = expand_bbox(polygon_bbox(board_polygon), BOARD_PROXIMITY_EXPAND)
    door_boxes = [
        expand_bbox(polygon_bbox(p), BOARD_PROXIMITY_EXPAND)
        for p in (door_polygons or [])
    ]

    min_span = min_teacher_span_ms(duration_ms)

    def in_box(box, cx: float, cy: float) -> bool:
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def near_board(cx: float, cy: float) -> bool:
        return board_box is not None and in_box(board_box, cx, cy)

    def near_door(cx: float, cy: float) -> bool:
        return any(in_box(b, cx, cy) for b in door_boxes)

    def near_trajectory(cx: float, cy: float) -> bool:
        limit_sq = ABSORB_TRAJECTORY_DIST * ABSORB_TRAJECTORY_DIST
        return any(
            (tx - cx) ** 2 + (ty - cy) ** 2 <= limit_sq for tx, ty in centers
        )

    def teacher_overlap_samples(first_ms: int, last_ms: int) -> int:
        return sum(1 for ts in ts_sorted if first_ms <= ts <= last_ms)

    qualifying: list[IdentityFeatures] = []
    for f in features:
        if f.track_no == teacher_no:
            continue
        near_edge = (
            f.mean_cx <= EDGE_MARGIN
            or f.mean_cx >= 1.0 - EDGE_MARGIN
            or f.mean_cy <= EDGE_MARGIN
            or f.mean_cy >= 1.0 - EDGE_MARGIN
        )
        if near_edge:
            continue
        if teacher_f.mean_area > 0 and (
            f.mean_area < ABSORB_MIN_AREA_RATIO * teacher_f.mean_area
        ):
            continue

        contained = any(ws <= f.first_ms and f.last_ms <= we for ws, we, _ in windows)
        starts_in_window = any(ws <= f.first_ms <= we for ws, we, _ in windows)

        if contained and f.span_ms < min_span:
            if near_board(f.mean_cx, f.mean_cy) or near_trajectory(
                f.mean_cx, f.mean_cy
            ):
                qualifying.append(f)
            continue

        if starts_in_window:
            # Continuation: she cannot be in two places, so any real temporal
            # overlap with the teacher identity disqualifies the fragment.
            if teacher_overlap_samples(f.first_ms, f.last_ms) > 5:
                continue
            frag_dets = sorted(
                dets_by_track.get(f.track_no, []), key=lambda d: d.video_ts_ms
            )
            if not frag_dets:
                continue
            head = [_center(d) for d in frag_dets[:3]]
            hx = sum(c[0] for c in head) / len(head)
            hy = sum(c[1] for c in head) / len(head)
            if near_door(hx, hy) or near_board(hx, hy) or near_trajectory(hx, hy):
                qualifying.append(f)

    # Two co-present fragments cannot both be the teacher: keep the earliest
    # non-overlapping set (ties resolved by proximity to the teacher path via
    # list order being start-sorted; overlap tolerance mirrors merge's 1s).
    qualifying.sort(key=lambda f: (f.first_ms, f.last_ms))
    absorbed: list[IdentityFeatures] = []
    for f in qualifying:
        overlaps = any(
            min(f.last_ms, g.last_ms) - max(f.first_ms, g.first_ms) > 1_000
            for g in absorbed
        )
        if not overlaps:
            absorbed.append(f)
    return [f.track_no for f in absorbed]
