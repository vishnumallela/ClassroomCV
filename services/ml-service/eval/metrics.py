"""Metrics that say whether the teacher was actually followed.

The old harness gated on aggregate KPIs (present_ms, board_ms, entries) with
tolerances. Those numbers can be right for the wrong reasons — a pipeline that
follows the teacher for half the lesson and a look-alike pupil for the other
half reports a perfectly plausible present_ms — and they say nothing at all
about the failures that actually get reported from real classrooms: her id
switching when she comes back into frame, and nothing being labelled for the
first minutes of the lesson.

These are identity metrics over per-frame ground truth, so each of those
failures has its own number:

    coverage        of the frames where she is visible, how many did we
                    label as her (recall)
    purity          of the frames we called her, how many were her (precision)
    id_switches     how many times the answer flipped from her to someone
                    else and back — the "id keeps changing" complaint
    cold_start_ms   how long into the lesson before she was first labelled —
                    the "first few minutes nothing happens" complaint
    reentry_recall  after she leaves frame and comes back, how often we pick
                    her up again
    gap_recovery_ms how long the average re-acquisition takes

All are pure functions of (ground truth boxes by timestamp, predicted boxes by
timestamp), so they work identically for real annotated footage and for
synthetic scenarios.
"""

from __future__ import annotations

from typing import Iterable, Optional

Box = tuple[float, float, float, float]  # x, y, w, h (normalized, top-left)

# A predicted box counts as the same person as the ground-truth box above this
# overlap. 0.3 is the usual MOT anchor tolerance: it accepts the tracker's
# looser box on a half-occluded body while still rejecting the neighbour.
MATCH_IOU = 0.3
# A hole in ground truth at least this long means she genuinely left the shot
# (or the annotator lost her); shorter holes are sampling noise.
ABSENCE_MS = 5_000
# After a re-entry, the window in which a pipeline still counts as having
# re-acquired her.
REACQUIRE_WINDOW_MS = 15_000


def iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _absences(ts_sorted: list[int], gap_ms: int = ABSENCE_MS) -> list[tuple[int, int]]:
    """(gone_at, back_at) pairs where ground truth has a real hole."""
    return [
        (a, b)
        for a, b in zip(ts_sorted, ts_sorted[1:])
        if b - a >= gap_ms
    ]


def evaluate_teacher(
    truth: dict[int, Box],
    predicted: dict[int, Box],
    match_iou: float = MATCH_IOU,
) -> dict:
    """Identity metrics for one video's teacher timeline.

    `truth` and `predicted` map a sampled timestamp to the teacher's box.
    Timestamps present in `predicted` but not in `truth` are not counted
    against precision: ground truth is only an assertion about the frames it
    covers, and on real footage the annotator abstains rather than guessing.
    """
    if not truth:
        return {
            "coverage": 0.0,
            "purity": 0.0,
            "id_switches": 0,
            "cold_start_ms": None,
            "reentry_recall": None,
            "gap_recovery_ms": None,
            "truth_frames": 0,
            "predicted_frames": len(predicted),
        }

    ts_sorted = sorted(truth)
    hits = 0
    scored = 0  # truth frames where we said something
    correct_seq: list[tuple[int, bool]] = []
    for ts in ts_sorted:
        p = predicted.get(ts)
        if p is None:
            correct_seq.append((ts, False))
            continue
        scored += 1
        ok = iou(truth[ts], p) >= match_iou
        hits += 1 if ok else 0
        correct_seq.append((ts, ok))

    coverage = hits / len(ts_sorted)
    purity = hits / scored if scored else 0.0

    # An id switch is the answer going from her to NOT her while she is still
    # on screen — a pipeline that simply loses her (says nothing) has a
    # coverage problem, not a switching problem, and the two want different
    # fixes.
    switches = 0
    for (ts_a, a), (ts_b, b) in zip(correct_seq, correct_seq[1:]):
        if a and not b and predicted.get(ts_b) is not None:
            switches += 1

    first_ok = next((ts for ts, ok in correct_seq if ok), None)
    cold_start = None if first_ok is None else first_ok - ts_sorted[0]

    reentries = _absences(ts_sorted)
    recovered: list[int] = []
    for _gone, back in reentries:
        window = [
            ts for ts, ok in correct_seq if back <= ts <= back + REACQUIRE_WINDOW_MS and ok
        ]
        if window:
            recovered.append(window[0] - back)
    reentry_recall = (len(recovered) / len(reentries)) if reentries else None
    gap_recovery = (sum(recovered) / len(recovered)) if recovered else None

    return {
        "coverage": round(coverage, 4),
        "purity": round(purity, 4),
        "id_switches": switches,
        "cold_start_ms": cold_start,
        "reentry_recall": None if reentry_recall is None else round(reentry_recall, 4),
        "gap_recovery_ms": None if gap_recovery is None else int(gap_recovery),
        "reentries": len(reentries),
        "truth_frames": len(ts_sorted),
        "predicted_frames": len(predicted),
    }


def teacher_boxes(tracks: Iterable[dict], detections: Iterable) -> dict[int, Box]:
    """Predicted teacher box per timestamp, from an AnalysisResult + detections."""
    teacher_no = next(
        (t["track_no"] for t in tracks if t.get("role") == "teacher"), None
    )
    if teacher_no is None:
        return {}
    out: dict[int, Box] = {}
    for d in detections:
        if d.track_no == teacher_no:
            b = d.bbox
            out[d.video_ts_ms] = (b["x"], b["y"], b["w"], b["h"])
    return out


def summarize(name: str, m: dict) -> str:
    cold = "never" if m["cold_start_ms"] is None else f"{m['cold_start_ms'] / 1000:.0f}s"
    reentry = "-" if m.get("reentry_recall") is None else f"{m['reentry_recall'] * 100:.0f}%"
    return (
        f"{name:<28} coverage={m['coverage'] * 100:5.1f}%  purity={m['purity'] * 100:5.1f}%  "
        f"switches={m['id_switches']:<3} cold_start={cold:<6} reentry={reentry}"
    )


def gate(m: dict, spec: dict) -> list[tuple[str, bool, str]]:
    """Check a metrics dict against a gate spec; returns (name, ok, detail) rows.

    Gate spec keys are metric names mapped to {"min"} and/or {"max"} bounds, so
    a scenario can demand "coverage at least 0.9 and no more than 1 switch"
    without the harness knowing anything about that scenario.
    """
    rows: list[tuple[str, bool, str]] = []
    for key, bound in spec.items():
        actual = m.get(key)
        if actual is None:
            rows.append((key, "min" not in bound, f"actual=None {bound}"))
            continue
        ok = True
        if "min" in bound:
            ok = ok and actual >= bound["min"]
        if "max" in bound:
            ok = ok and actual <= bound["max"]
        rows.append((key, ok, f"actual={actual} {bound}"))
    return rows
