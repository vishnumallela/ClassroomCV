"""Class-specific zones for the two static objects: the board and the door.

The board and the door do not move, and the detector finds them in almost every
frame of a lesson (measured: screen present in 100% and 96.5% of sampled frames
on two lessons, door in 77-81%). That makes zoning a data problem rather than a
search problem, and it replaces a 773-line open-vocabulary + SAM 2 proposal
chain with a median.

Two jobs, and they are opposites:

PROPOSE  Turn a lesson's own screen/door detections into a zone polygon, so a
         room configures itself on first upload and the user only corrects it.
         The median position across the whole video, not one frame's box: the
         detector occasionally puts a second screen box somewhere else (5
         frames in 1,118 on the long lesson), and a median ignores that where a
         single sample would enshrine it.

GATE     Drop screen/door detections that fall OUTSIDE the configured zone. A
         wall poster that reads as a screen for six frames cannot then move the
         board, and per-frame board geometry stays anchored to the real board.

The teacher is deliberately NOT gated. She has the run of the room — she stands
at the board, walks to the door, crosses to the back — so a zone around her
would be a bug, not an optimisation. Zoning exists to pin down the furniture.

Zone kinds keep the product's own vocabulary ("board"), which is what the
database, the API contract and the zone editor all speak; the model's label for
the same object is "Screen".
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.geometry import polygon_bbox
from app.models import CLASS_DOOR, CLASS_SCREEN, Detection

logger = logging.getLogger(__name__)

# Zone kind <-> detector class. "board" is the product's word for the thing the
# checkpoint calls "Screen".
ZONE_CLASS = {"board": CLASS_SCREEN, "door": CLASS_DOOR}

# A proposal needs to be seen in at least this fraction of the frames that
# carried ANY detection, so a handful of false positives in a lesson where the
# object is genuinely absent cannot mint a zone. The door is missing from ~20%
# of frames even when present, so this sits well below that.
MIN_PRESENCE = 0.30
# ...and this many frames minimum, so a 10-frame clip cannot propose a zone off
# three detections.
MIN_SAMPLES = 8
# How far outside its zone a static detection may sit and still be accepted, as
# a fraction of the frame. Covers honest box jitter around the true object
# without admitting a poster on the far wall.
GATE_TOLERANCE = 0.05


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _bbox_to_polygon(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Axis-aligned rectangle as the 4-point polygon the zone contract expects."""
    return [
        [round(x0, 4), round(y0, 4)],
        [round(x1, 4), round(y0, 4)],
        [round(x1, 4), round(y1, 4)],
        [round(x0, 4), round(y1, 4)],
    ]


def propose_zone(
    detections: list[Detection], kind: str, frames_seen: Optional[int] = None
) -> dict:
    """Propose a zone polygon for 'board' or 'door' from a lesson's detections.

    Returns the same contract the old detector chain did:
    {polygon, confidence, method, frame_ts_ms}, with polygon None when the
    object was not seen consistently enough to place it.
    """
    cls = ZONE_CLASS.get(kind)
    if cls is None:
        raise ValueError(f"unknown zone kind: {kind!r}")
    threshold = get_settings().zone_conf
    hits = [d for d in detections if d.cls == cls and d.conf >= threshold]

    total_frames = frames_seen or len({d.video_ts_ms for d in detections}) or 1
    seen_frames = len({d.video_ts_ms for d in hits})
    presence = seen_frames / total_frames

    if len(hits) < MIN_SAMPLES or presence < MIN_PRESENCE:
        logger.info(
            "no %s zone proposed: %d detections over %d/%d frames (%.0f%% presence)",
            kind, len(hits), seen_frames, total_frames, presence * 100,
        )
        return {
            "polygon": None,
            "confidence": round(presence, 3),
            "method": "rfdetr",
            "frame_ts_ms": 0,
        }

    # Median of each edge independently. The object is static, so every edge has
    # a true value and the median is the robust estimate of it; averaging would
    # let one stray box drag the zone.
    x0 = _median([d.bbox["x"] for d in hits])
    y0 = _median([d.bbox["y"] for d in hits])
    x1 = _median([d.bbox["x"] + d.bbox["w"] for d in hits])
    y1 = _median([d.bbox["y"] + d.bbox["h"] for d in hits])
    # Confidence blends how reliably the object was seen with how sure the model
    # was, so a zone placed off a handful of weak boxes reports as weak.
    mean_conf = sum(d.conf for d in hits) / len(hits)
    confidence = round(min(1.0, 0.5 * min(1.0, presence / 0.8) + 0.5 * mean_conf), 3)

    logger.info(
        "%s zone from %d detections (%.0f%% presence, conf %.2f): [%.3f %.3f %.3f %.3f]",
        kind, len(hits), presence * 100, confidence, x0, y0, x1, y1,
    )
    return {
        "polygon": _bbox_to_polygon(x0, y0, x1, y1),
        "confidence": confidence,
        "method": "rfdetr",
        "frame_ts_ms": min(d.video_ts_ms for d in hits),
    }


def gate_static(detections: list[Detection], zones: list[dict]) -> list[Detection]:
    """Drop board/door detections that fall outside their configured zone.

    A zone that is not configured gates nothing — an unconfigured room must
    still see its furniture, or it could never propose a zone in the first
    place. The teacher and the action classes pass through untouched.
    """
    limits: dict[int, tuple[float, float, float, float]] = {}
    for z in zones:
        cls = ZONE_CLASS.get(z.get("kind", ""))
        polygon = z.get("polygon")
        if cls is None or not polygon:
            continue
        x0, y0, x1, y1 = polygon_bbox(polygon)
        limits[cls] = (
            x0 - GATE_TOLERANCE,
            y0 - GATE_TOLERANCE,
            x1 + GATE_TOLERANCE,
            y1 + GATE_TOLERANCE,
        )
    if not limits:
        return detections

    kept: list[Detection] = []
    dropped = 0
    for d in detections:
        bounds = limits.get(d.cls)
        if bounds is not None:
            cx = d.bbox["x"] + d.bbox["w"] / 2.0
            cy = d.bbox["y"] + d.bbox["h"] / 2.0
            if not (bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]):
                dropped += 1
                continue
        kept.append(d)
    if dropped:
        logger.info("zone gate dropped %d out-of-zone static detection(s)", dropped)
    return kept
