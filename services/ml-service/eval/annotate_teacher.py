"""Build dense teacher ground truth for a captured fixture, then let a human check it.

Measuring identity quality needs per-frame truth ("which box is the teacher at
t?"), and hand-labelling 3,460 frames of a 12-minute lesson is not going to
happen. This annotator uses a cue that is trivially true for ONE video and that
the pipeline is never allowed to use — in the Khaitan footage the teacher is the
only person not wearing a coloured school polo — and turns it into a per-frame
label, which a person then spot-checks on a contact sheet.

The split matters: the oracle may use anything video-specific (a shirt colour, a
seat, a face) precisely because it is not shipping. The pipeline must find her
from age, motion and appearance alone, and this file is what proves whether it
did.

Usage (from services/ml-service):
    uv run python eval/annotate_teacher.py <fixture-name> [--review out.jpg]

Writes eval/gt/<fixture-name>.teacher.json:
    {"anchors": {ts_ms: [x, y, w, h]}, "rule": ..., "reviewed": [...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app import detector  # noqa: E402
from eval import fixture as fixture_mod  # noqa: E402

GT_DIR = EVAL_DIR / "gt"

# "Not wearing a coloured uniform" — which in practice means one of two things,
# because school uniforms are bright and saturated and teachers' clothes are
# not. Numbers are in OpenCV HSV units (S 0-255, V 0-255) and are deliberately
# loose: the decision is made by RANKING people within a frame, never by the
# threshold. Which cue applies is a property of the video and is recorded in
# the ground-truth file.
WHITE_S_MAX = 70
WHITE_V_MIN = 110
# ...and the dark counterpart, for a teacher in a dark or muted outfit among
# bright polos.
DARK_S_MAX = 120
DARK_V_MAX = 95
# The teacher is an adult standing among seated children: require a candidate
# to be at least this tall relative to the median box in the frame, which
# removes blown-out doorway slivers and desk reflections.
MIN_RELATIVE_HEIGHT = 0.9
# A frame only gets a label when the best candidate is clearly the best.
MIN_WHITE_FRACTION = 0.45
MIN_LEAD = 0.10
# Temporal cleanup: labels are IoU-linked into runs, and runs shorter than this
# are dropped as flicker.
MIN_RUN_MS = 2_000
MAX_LINK_DIST = 0.12


def _plausible(bbox: dict, median_h: float, exclude_x_below: float) -> bool:
    """Reject what the colour cue alone would happily label as the teacher.

    Two things in this room are pale and are not her: the desktops (which the
    detector occasionally boxes as a person) and the one pupil who wears a
    white shirt, who never leaves his corner seat. Both are excluded
    geometrically, and the exclusion is recorded in the ground-truth file so a
    reader can see exactly what the labels are and are not.
    """
    if bbox["h"] < MIN_RELATIVE_HEIGHT * median_h:
        return False
    return bbox["x"] + bbox["w"] / 2.0 >= exclude_x_below


def _torso_offuniform(frame: np.ndarray, bbox: dict, cue: str) -> float:
    """Fraction of the chest band that does not look like a bright uniform."""
    fh, fw = frame.shape[:2]
    # A tight band across the chest: wider or taller and a seated pupil's box
    # starts sampling the white desktop behind them, which reads as a white
    # shirt.
    x0 = max(0, int((bbox["x"] + 0.25 * bbox["w"]) * fw))
    x1 = min(fw, int((bbox["x"] + 0.75 * bbox["w"]) * fw))
    y0 = max(0, int((bbox["y"] + 0.28 * bbox["h"]) * fh))
    y1 = min(fh, int((bbox["y"] + 0.52 * bbox["h"]) * fh))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    if cue == "dark":
        hit = (hsv[:, :, 1] < DARK_S_MAX) & (hsv[:, :, 2] < DARK_V_MAX)
    else:
        hit = (hsv[:, :, 1] < WHITE_S_MAX) & (hsv[:, :, 2] > WHITE_V_MIN)
    return float(hit.mean())


def _link_runs(labels: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Drop labels that do not belong to a temporally coherent run.

    A single frame where a blown-out doorway outscored her is noise; a stretch
    of seconds where she stands in white is the truth. Runs are broken by a
    jump in position, which is what a mislabel looks like.
    """
    if not labels:
        return []
    runs: list[list[tuple[int, dict]]] = [[labels[0]]]
    for ts, box in labels[1:]:
        pts, pbox = runs[-1][-1]
        d = np.hypot(
            (box["x"] + box["w"] / 2) - (pbox["x"] + pbox["w"] / 2),
            (box["y"] + box["h"] / 2) - (pbox["y"] + pbox["h"] / 2),
        )
        if ts - pts <= 3_000 and d <= MAX_LINK_DIST:
            runs[-1].append((ts, box))
        else:
            runs.append([(ts, box)])
    out: list[tuple[int, dict]] = []
    for run in runs:
        if run[-1][0] - run[0][0] >= MIN_RUN_MS:
            out.extend(run)
    return out


def annotate(
    name: str,
    review_path: str | None,
    exclude_x_below: float = 0.0,
    cue: str = "pale",
    min_fraction: float = MIN_WHITE_FRACTION,
) -> int:
    fx = fixture_mod.load(name)
    if fx is None:
        print(f"no fixture named {name!r}")
        return 1
    if not fx.source or not Path(fx.source).exists():
        print(f"fixture {name!r} has no readable source video ({fx.source})")
        return 1

    by_ts: dict[int, list] = {}
    for d in fx.detections:
        by_ts.setdefault(d.video_ts_ms, []).append(d)

    labels: list[tuple[int, dict]] = []
    info = detector.FrameSourceInfo()
    frames = detector.iter_frames(fx.source, 5.0, info=info)
    matched = 0
    try:
        for ts_ms, frame in frames:
            dets = by_ts.get(ts_ms)
            if not dets:
                continue
            matched += 1
            median_h = float(np.median([d.bbox["h"] for d in dets]))
            scored: list[tuple[float, dict]] = []
            for d in dets:
                # Standing only: she is on her feet whenever she is teaching,
                # and the rule keeps a seated pupil whose box swallows a pale
                # desk from ever winning the frame. Ground truth abstains on
                # the frames where she crouches rather than guessing.
                if not d.standing or not _plausible(d.bbox, median_h, exclude_x_below):
                    continue
                scored.append((_torso_offuniform(frame, d.bbox, cue), d.bbox))
            if not scored:
                continue
            scored.sort(key=lambda s: -s[0])
            best = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best[0] >= min_fraction and best[0] - runner >= MIN_LEAD:
                labels.append((ts_ms, best[1]))
    finally:
        frames.close()

    kept = _link_runs(labels)
    GT_DIR.mkdir(exist_ok=True)
    out = GT_DIR / f"{name}.teacher.json"
    out.write_text(
        json.dumps(
            {
                "fixture": name,
                "cue": cue,
                "rule": (
                    f"torso {cue}-pixel fraction, ranked within each frame; "
                    f"candidates shorter than {MIN_RELATIVE_HEIGHT} x the frame "
                    f"median height, not standing, or centred left of "
                    f"x={exclude_x_below} are excluded; runs shorter than "
                    f"{MIN_RUN_MS} ms dropped"
                ),
                "frames_scored": matched,
                "anchors": {
                    str(ts): [
                        round(b["x"], 5),
                        round(b["y"], 5),
                        round(b["w"], 5),
                        round(b["h"], 5),
                    ]
                    for ts, b in kept
                },
            },
            indent=1,
        )
    )
    span = (kept[-1][0] - kept[0][0]) / 1000.0 if kept else 0.0
    print(
        f"{len(kept)} anchors over {span:.0f}s of {matched} scored frames "
        f"({len(labels) - len(kept)} flicker labels dropped) -> {out.name}"
    )

    if review_path and kept:
        _contact_sheet(fx.source, kept, review_path)
        print(f"contact sheet for review: {review_path}")
    return 0


def _contact_sheet(video: str, kept: list[tuple[int, dict]], path: str, cols: int = 8, rows: int = 4) -> None:
    """Tile evenly-spaced labelled crops so a human can check the whole video at once."""
    cap = cv2.VideoCapture(video)
    picks = [kept[i * (len(kept) - 1) // (cols * rows - 1)] for i in range(cols * rows)]
    cell = 180
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    try:
        for i, (ts, b) in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ok, frame = cap.read()
            if not ok:
                continue
            fh, fw = frame.shape[:2]
            x0 = max(0, int((b["x"] - 0.02) * fw))
            x1 = min(fw, int((b["x"] + b["w"] + 0.02) * fw))
            y0 = max(0, int((b["y"] - 0.02) * fh))
            y1 = min(fh, int((b["y"] + b["h"] + 0.02) * fh))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            scale = cell / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            r, c = divmod(i, cols)
            sheet[r * cell : r * cell + crop.shape[0], c * cell : c * cell + crop.shape[1]] = crop
            cv2.putText(sheet, f"{ts // 1000}s", (c * cell + 4, r * cell + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    finally:
        cap.release()
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--review", default=None, help="write a contact sheet here")
    ap.add_argument("--cue", choices=("pale", "dark"), default="pale")
    ap.add_argument("--min-fraction", type=float, default=MIN_WHITE_FRACTION)
    ap.add_argument(
        "--exclude-x-below",
        type=float,
        default=0.0,
        help="ignore candidates whose centre x is left of this (a fixed pale "
        "distractor: a white-shirted pupil in a corner seat)",
    )
    args = ap.parse_args()
    raise SystemExit(
        annotate(args.name, args.review, args.exclude_x_below, args.cue, args.min_fraction)
    )
