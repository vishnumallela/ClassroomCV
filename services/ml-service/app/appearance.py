"""What a tracked body looks like, compactly enough to store per detection.

Phase 3 of docs/teacher-attribution-plan.md needs two things motion cannot
give: whether two segments separated by a gap are the same person, and whether
a segment quietly changed person at an occluded crossing. Both are questions
about appearance, and both must be answerable on /rederive — from stored rows,
with no video in hand. So the descriptor is computed once, at detection time,
and persisted in detection_events.meta beside the class id (jsonb: additive,
no migration).

It is a colour histogram of the torso band, not an embedding. Three reasons:
no new dependency; it is the linker, not the decider (the timetable decides,
appearance only joins fragments and catches swaps); and the case that made
this necessary — a cream-striped kurta versus a black one — is exactly what a
colour histogram is good at. Two similar saris will defeat it, and the plan
says so; that is the day an embedding earns its place.

Torso band, not the whole box: the head is hair and skin (the same for most
adults here), the bottom is floor and furniture, and the model's box extent
varies frame to frame (head-only, torso, full body). The band from 15% to 75%
of the box's height is the clothing whichever extent was drawn.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Histogram layout: hue (8 bins over OpenCV's 0-179), saturation (4), value (4).
# Hue is weighted by saturation so grey, black and white contribute almost no
# hue — their "hue" is noise — and show up in the S and V parts instead.
H_BINS, S_BINS, V_BINS = 8, 4, 4
DIMS = H_BINS + S_BINS + V_BINS
# The clothing band, as fractions of box height and width. The central 60% of
# the width, because a teacher in this room is rarely alone in her box: the
# students standing beside her wear green and yellow, and on the real handover
# clip trimming the sides raised the swap signal by half and lowered the noise
# floor. Measured, not guessed — see docs/teacher-attribution-plan.md, Phase 3.
BAND_TOP, BAND_BOTTOM = 0.15, 0.75
BAND_LEFT, BAND_RIGHT = 0.2, 0.8
# Crops are resized to this before the histogram so a far, small box and a
# near, large one contribute the same number of pixels.
PATCH = 24
# Boxes smaller than this in either pixel dimension carry no colour worth
# reading; they get no descriptor rather than a misleading one.
MIN_PX = 8
ROUND = 4


def describe(frame_bgr: np.ndarray, bbox: dict) -> Optional[list[float]]:
    """Descriptor for one box on one frame, or None when the crop is too small.

    `bbox` is normalized {x, y, w, h}; `frame_bgr` is the full frame as cv2
    reads it. Returns DIMS floats in [0, 1]: three L1-normalized histograms
    concatenated, so histogram intersection per part is a distance.
    """
    fh, fw = frame_bgr.shape[:2]
    x0 = int(round((bbox["x"] + bbox["w"] * BAND_LEFT) * fw))
    x1 = int(round((bbox["x"] + bbox["w"] * BAND_RIGHT) * fw))
    top = bbox["y"] + bbox["h"] * BAND_TOP
    bottom = bbox["y"] + bbox["h"] * BAND_BOTTOM
    y0 = int(round(top * fh))
    y1 = int(round(bottom * fh))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fw, x1), min(fh, y1)
    if x1 - x0 < MIN_PX or y1 - y0 < MIN_PX:
        return None
    crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    h_hist, _ = np.histogram(h, bins=H_BINS, range=(0.0, 180.0), weights=s)
    s_hist, _ = np.histogram(s, bins=S_BINS, range=(0.0, 1.0))
    v_hist, _ = np.histogram(v, bins=V_BINS, range=(0.0, 1.0))

    parts = []
    for hist in (h_hist, s_hist, v_hist):
        total = float(hist.sum())
        if total > 0:
            parts.extend((hist / total).tolist())
        else:
            # No chroma at all (a black or white garment): "no hue" is a
            # uniform hue, so two hue-less things agree with each other and
            # only partially with anything actually coloured. All-zero here
            # would make a black kurta disagree with ITSELF.
            parts.extend([1.0 / len(hist)] * len(hist))
    return [round(float(p), ROUND) for p in parts]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - mean histogram intersection over the three parts: 0 = identical."""
    if len(a) != DIMS or len(b) != DIMS:
        return 1.0
    a_ = np.asarray(a, dtype=np.float32)
    b_ = np.asarray(b, dtype=np.float32)
    bounds = ((0, H_BINS), (H_BINS, H_BINS + S_BINS), (H_BINS + S_BINS, DIMS))
    inter = [float(np.minimum(a_[lo:hi], b_[lo:hi]).sum()) for lo, hi in bounds]
    return 1.0 - sum(inter) / len(inter)


def mean_descriptor(descs: Sequence[Sequence[float]]) -> Optional[list[float]]:
    """Element-wise mean of several descriptors (each part stays L1-normalized)."""
    rows = [d for d in descs if d is not None and len(d) == DIMS]
    if not rows:
        return None
    return [round(float(v), ROUND) for v in np.mean(np.asarray(rows, dtype=np.float32), axis=0)]
