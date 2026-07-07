"""Board (blackboard / whiteboard) zone detection for classroom videos.

Strategy chain (feature contract):
1. YOLO-World open-vocabulary proposals ("blackboard", "whiteboard",
   "chalkboard", "projection screen"), each refined into a mask by SAM 2 and
   geometrically scored. YOLOWorld.set_classes needs the optional `clip`
   package (ultralytics' CLIP fork, git-only); when it is not installed this
   strategy is skipped silently and the chain reports 'sam2_geometric'.
   IMPORTANT: clip availability is checked with importlib BEFORE touching
   ultralytics.nn.text_model, because importing that module auto-pip-installs
   at runtime (not offline-safe).
2. Always-available fallback: SAM 2 ('sam2.1_s.pt' preferred, 'sam2.1_b.pt'
   fallback; weights predownloaded into services/ml-service/) prompted with a
   grid of points over the upper ~65% of the frame.

Every candidate mask gets a geometric score in 0..1:
  wide aspect + centroid in the upper ~65% of the frame + area 3-45% of the
  frame + rectangularity + color uniformity / low saturation (the color term
  is a multiplicative factor so busy textures are punished hard), plus a
  penalty for full-width bands (sky / wall / road strips are not boards).

Three frames are sampled at 5% / 15% / 30% of duration; the best-scoring
candidate across frames wins and frame_ts_ms is that frame's timestamp.
Best score < MIN_SCORE (0.25) -> polygon None.

Device: prefer MPS, fall back to CPU on failure (cached per model, mirroring
app.detector's _fallback_cpu pattern). SAM 2.1-small was verified working on
MPS with torch 2.12.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.detector import _validate_video_path, get_device

logger = logging.getLogger(__name__)

# Chain / scoring knobs
MIN_SCORE = 0.25  # below this the response polygon is null
MAX_POLYGON_POINTS = 12
MIN_POLYGON_POINTS = 4
FRAME_FRACTIONS = (0.05, 0.15, 0.30)
FALLBACK_NATIVE_FPS = 30.0

# Models (resolved against the service dir so cwd does not matter)
_SERVICE_DIR = Path(__file__).resolve().parent.parent
SAM_MODEL_NAMES = ("sam2.1_s.pt", "sam2.1_b.pt")
WORLD_MODEL_NAME = "yolov8s-worldv2.pt"
BOARD_PROMPTS = ["blackboard", "whiteboard", "chalkboard", "projection screen"]
DOOR_PROMPTS = ["door", "doorway"]
# One YOLO-World model serves both targets; class indices < N_BOARD_CLASSES are
# board prompts, the rest are door prompts.
WORLD_PROMPTS = BOARD_PROMPTS + DOOR_PROMPTS
N_BOARD_CLASSES = len(BOARD_PROMPTS)
WORLD_MIN_CONF = 0.05  # open-vocab proposals are low-confidence by nature
WORLD_MAX_PROPOSALS = 5

# SAM point-prompt grid over the candidate (upper) region of the frame
GRID_X = (0.15, 0.30, 0.50, 0.70, 0.85)
GRID_Y = (0.15, 0.30, 0.45, 0.60)

# Door grid: doors span floor to ceiling and usually sit at the sides, so this
# covers more horizontal positions and the mid-to-lower band.
DOOR_GRID_X = (0.08, 0.22, 0.38, 0.50, 0.62, 0.78, 0.92)
DOOR_GRID_Y = (0.28, 0.45, 0.62, 0.78)
DOOR_MIN_SCORE = 0.22

_sam = None
_world = None
_world_failed = False
_sam_fallback_cpu = False
_world_fallback_cpu = False


# --------------------------------------------------------------------------- #
# Geometric scoring (pure, unit-testable without models)
# --------------------------------------------------------------------------- #


def _ramp(v: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """Trapezoid membership: 0 at <=lo0, ramps to 1 on [lo1, hi1], 0 at >=hi0."""
    if v <= lo0 or v >= hi0:
        return 0.0
    if v < lo1:
        return (v - lo0) / (lo1 - lo0)
    if v <= hi1:
        return 1.0
    return (hi0 - v) / (hi0 - hi1)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _largest_contour(mask: np.ndarray):
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    return contour


def mask_stats(mask: np.ndarray) -> Optional[dict]:
    """Geometric properties of the largest connected blob in a binary mask.

    Returns None for empty/degenerate masks. All fractions are relative to
    the mask (frame) size: area_frac, cx, cy, span_frac; aspect is the
    pixel-space width/height of the axis-aligned bounding box;
    rectangularity is blob area / min-area-rect area (1.0 = perfect
    rectangle, rotation tolerant).
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return None
    contour = _largest_contour(mask)
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    if area < 16:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    bx, by, bw, bh = cv2.boundingRect(contour)
    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    rect_area = rw * rh
    return {
        "area_frac": area / (w * h),
        "cx": moments["m10"] / moments["m00"] / w,
        "cy": moments["m01"] / moments["m00"] / h,
        "aspect": bw / bh if bh > 0 else 0.0,
        "rectangularity": area / rect_area if rect_area > 0 else 0.0,
        "span_frac": bw / w,
        "touches_left": bx <= 2,
        "touches_right": bx + bw >= w - 2,
    }


def _color_score(mask: np.ndarray, frame: np.ndarray) -> float:
    """Color uniformity + low saturation of the masked region, 0..1.

    Boards are flat-colored (green/black/white) surfaces: per-channel HSV
    std should be small and saturation moderate-to-low. Busy textures
    (vehicles, crowds, shelves) score near 0.
    """
    sel = mask.astype(bool)
    if frame is None or frame.shape[:2] != mask.shape[:2] or sel.sum() < 32:
        return 0.5  # neutral when color is unavailable
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    region = hsv[sel].astype(np.float64)
    stds = region.std(axis=0)  # (H, S, V) stds
    uniformity = _clamp01(1.0 - float(stds.mean()) / 48.0)
    sat = float(region[:, 1].mean()) / 255.0
    low_sat = _clamp01(1.0 - max(0.0, sat - 0.35) / 0.45)
    return 0.65 * uniformity + 0.35 * low_sat


def score_mask(mask: np.ndarray, frame: Optional[np.ndarray] = None) -> float:
    """Board-likeness score in 0..1 for a binary mask (optionally + frame).

    Geometric base (weighted): wide aspect, centroid in the upper ~65% of
    the frame, area 3-45% of the frame, rectangularity. Multiplied by a
    color factor (uniform / low-saturation regions keep their score, busy
    textures lose more than half) and a full-width-band penalty.
    """
    stats = mask_stats(mask)
    if stats is None:
        return 0.0

    aspect_s = _ramp(stats["aspect"], 1.1, 1.9, 6.0, 12.0)
    position_s = _ramp(stats["cy"], 0.05, 0.12, 0.48, 0.68)
    area_s = _ramp(stats["area_frac"], 0.03, 0.06, 0.35, 0.45)
    rect_s = _clamp01((stats["rectangularity"] - 0.70) / 0.25)

    geom = 0.30 * aspect_s + 0.25 * position_s + 0.20 * area_s + 0.25 * rect_s

    # Hard gates (multiplicative — tuned so the bus street scene stays below
    # the pipeline's 0.5 auto-accept while a real board keeps scoring high):
    # a board is defined by being WIDE and by a sane 3-45% footprint; tall
    # panels / slivers / wall-sized blobs must not ride the other components.
    if stats["aspect"] < 1.2:
        geom *= 0.3
    if stats["area_frac"] < 0.03 or stats["area_frac"] > 0.50:
        geom *= 0.5

    # Full-width horizontal bands (sky, wall strip, road) are not boards.
    if stats["span_frac"] > 0.9 and stats["touches_left"] and stats["touches_right"]:
        geom *= 0.5

    if frame is not None:
        geom *= 0.45 + 0.55 * _color_score(mask, frame)

    return float(_clamp01(geom))


def mask_to_polygon(mask: np.ndarray) -> Optional[list[list[float]]]:
    """Largest contour -> approxPolyDP simplified to 4..12 normalized points.

    Epsilon grows until the simplification fits MAX_POLYGON_POINTS; if it
    collapses below MIN_POLYGON_POINTS the min-area rectangle (4 corners) is
    used instead. Coordinates are normalized 0-1 and clamped to the frame.
    """
    contour = _largest_contour(mask)
    if contour is None:
        return None
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return None

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = contour
    for _ in range(12):
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) <= MAX_POLYGON_POINTS:
            break
        epsilon *= 1.5
    if len(approx) > MAX_POLYGON_POINTS or len(approx) < MIN_POLYGON_POINTS:
        approx = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2)

    polygon = [
        [
            round(_clamp01(float(x) / w), 4),
            round(_clamp01(float(y) / h), 4),
        ]
        for x, y in approx.reshape(-1, 2)
    ]
    return polygon


# --------------------------------------------------------------------------- #
# Model layer (lazy; monkeypatched away in unit tests)
# --------------------------------------------------------------------------- #


def _weight_path(name: str) -> str:
    """Resolve a weight file against the service dir (predownloaded) or name."""
    local = _SERVICE_DIR / name
    return str(local) if local.is_file() else name


def _preferred_device() -> str:
    return get_device()  # settings device with MPS availability check


def _get_sam():
    global _sam
    if _sam is None:
        from ultralytics import SAM

        last_exc: Optional[Exception] = None
        for name in SAM_MODEL_NAMES:
            try:
                _sam = SAM(_weight_path(name))
                logger.info("[board-detect] SAM model loaded: %s", name)
                break
            except Exception as exc:  # pragma: no cover - weight availability
                last_exc = exc
        if _sam is None:  # pragma: no cover
            raise RuntimeError(f"no SAM 2 weights available: {last_exc}")
    return _sam


def _get_world():
    """YOLO-World model with board prompts, or None when unusable.

    Returns None (and caches the failure) when the `clip` package is not
    installed — set_classes() requires it, and letting ultralytics discover
    that itself would trigger a runtime pip install, which must never happen
    in an offline service.
    """
    global _world, _world_failed
    if _world is not None:
        return _world
    if _world_failed:
        return None
    if importlib.util.find_spec("clip") is None:
        _world_failed = True
        logger.info(
            "[board-detect] 'clip' package not installed; "
            "YOLO-World proposals disabled (sam2_geometric only)"
        )
        return None
    try:
        from ultralytics import YOLOWorld

        model = YOLOWorld(_weight_path(WORLD_MODEL_NAME))
        model.set_classes(WORLD_PROMPTS)
        _world = model
    except Exception:
        logger.warning(
            "[board-detect] YOLO-World unavailable; using sam2_geometric only",
            exc_info=True,
        )
        _world_failed = True
        return None
    return _world


def _sam_segment(
    frame: np.ndarray,
    *,
    bboxes: Optional[list[list[float]]] = None,
    points: Optional[list[list[float]]] = None,
) -> list[np.ndarray]:
    """Run SAM 2 with box or point prompts -> list of boolean HxW masks."""
    global _sam_fallback_cpu
    model = _get_sam()
    kwargs: dict = {}
    if bboxes:
        kwargs["bboxes"] = bboxes
    if points:
        kwargs["points"] = points
        kwargs["labels"] = [1] * len(points)
    if not kwargs:
        return []

    device = "cpu" if _sam_fallback_cpu else _preferred_device()
    try:
        results = model(frame, device=device, verbose=False, **kwargs)
    except Exception as exc:
        if device == "cpu":
            raise
        logger.warning(
            "[board-detect] SAM on %s failed (%s); falling back to cpu", device, exc
        )
        _sam_fallback_cpu = True
        results = model(frame, device="cpu", verbose=False, **kwargs)

    masks = results[0].masks
    if masks is None:
        return []
    return [m.astype(bool) for m in masks.data.cpu().numpy()]


def _yolo_world_proposals(frame: np.ndarray) -> list[tuple[list[float], float, int]]:
    """Open-vocab proposals as ([x0, y0, x1, y1] pixels, conf, class_index)."""
    global _world_fallback_cpu
    model = _get_world()
    if model is None:
        return []
    device = "cpu" if _world_fallback_cpu else _preferred_device()
    try:
        results = model.predict(
            frame, conf=WORLD_MIN_CONF, device=device, verbose=False
        )
    except Exception as exc:
        if device == "cpu":
            logger.warning("[board-detect] YOLO-World inference failed: %s", exc)
            return []
        logger.warning(
            "[board-detect] YOLO-World on %s failed (%s); falling back to cpu",
            device,
            exc,
        )
        _world_fallback_cpu = True
        try:
            results = model.predict(
                frame, conf=WORLD_MIN_CONF, device="cpu", verbose=False
            )
        except Exception as exc2:
            logger.warning("[board-detect] YOLO-World inference failed: %s", exc2)
            return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    order = np.argsort(-confs)[:WORLD_MAX_PROPOSALS]
    return [([float(v) for v in xyxy[i]], float(confs[i]), int(clss[i])) for i in order]


# --------------------------------------------------------------------------- #
# Frame sampling + per-frame chain
# --------------------------------------------------------------------------- #


def _sample_frames(video_path: str) -> list[tuple[int, np.ndarray]]:
    """Frames at 5% / 15% / 30% of duration as (ts_ms, frame) pairs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = FALLBACK_NATIVE_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out: list[tuple[int, np.ndarray]] = []
        seen: set[int] = set()
        if frame_count > 0:
            for fraction in FRAME_FRACTIONS:
                idx = min(frame_count - 1, int(round(fraction * frame_count)))
                if idx in seen:
                    continue
                seen.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.append((int(round(idx / fps * 1000.0)), frame))
        if not out:  # metadata-less container: at least use the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append((0, frame))
        return out
    finally:
        cap.release()


def _dedupe_masks(
    masks: list[np.ndarray], iou_threshold: float = 0.9
) -> list[np.ndarray]:
    """Drop near-duplicate masks (grid points inside one object collapse)."""
    kept: list[np.ndarray] = []
    kept_small: list[np.ndarray] = []
    for mask in masks:
        small = mask[::4, ::4]
        duplicate = False
        for other in kept_small:
            union = np.logical_or(small, other).sum()
            if union and np.logical_and(small, other).sum() / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
            kept_small.append(small)
    return kept


def _detect_on_frame(
    frame: np.ndarray,
) -> Optional[tuple[float, list[list[float]], str]]:
    """Run the strategy chain on one frame -> (score, polygon, method) | None."""
    h, w = frame.shape[:2]
    candidates: list[tuple[float, list[list[float]], str]] = []

    # Strategy 1: YOLO-World board proposals refined by SAM 2 (optional).
    proposals = [p for p in _yolo_world_proposals(frame) if p[2] < N_BOARD_CLASSES]
    if proposals:
        masks = _sam_segment(frame, bboxes=[box for box, _, _ in proposals])
        for mask in masks:
            polygon = mask_to_polygon(mask)
            if polygon:
                candidates.append((score_mask(mask, frame), polygon, "yolo_world_sam2"))

    # Strategy 2 (always): SAM 2 with a point grid over the upper frame.
    points = [[gx * w, gy * h] for gy in GRID_Y for gx in GRID_X]
    for mask in _dedupe_masks(_sam_segment(frame, points=points)):
        polygon = mask_to_polygon(mask)
        if polygon:
            candidates.append((score_mask(mask, frame), polygon, "sam2_geometric"))

    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])


def detect_board(video_id: str, video_path: str) -> dict:
    """Full chain over sampled frames -> contract response dict.

    {"polygon": [[x,y],...]|None, "confidence": 0..1, "method": str,
     "frame_ts_ms": int}. polygon is None when nothing scores >= MIN_SCORE.
    Per-frame failures are logged and skipped — an all-failed run degrades to
    a null polygon (the pipeline treats absence as non-fatal).
    """
    video_path = _validate_video_path(video_path)
    frames = _sample_frames(video_path)

    best: Optional[tuple[float, list[list[float]], str]] = None
    best_ts = frames[0][0] if frames else 0
    for ts_ms, frame in frames:
        try:
            candidate = _detect_on_frame(frame)
        except Exception:
            logger.warning(
                "[board-detect] video %s: frame ts=%sms failed",
                video_id,
                ts_ms,
                exc_info=True,
            )
            continue
        if candidate is not None and (best is None or candidate[0] > best[0]):
            best = candidate
            best_ts = ts_ms

    if best is None:
        logger.info("[board-detect] video %s: no candidates found", video_id)
        return {
            "polygon": None,
            "confidence": 0.0,
            "method": "sam2_geometric",
            "frame_ts_ms": int(best_ts),
        }

    score, polygon, method = best
    confidence = round(float(_clamp01(score)), 4)
    logger.info(
        "[board-detect] video %s: method=%s confidence=%.3f ts=%sms polygon=%s",
        video_id,
        method,
        confidence,
        best_ts,
        "yes" if score >= MIN_SCORE else "below-threshold",
    )
    return {
        "polygon": polygon if score >= MIN_SCORE else None,
        "confidence": confidence,
        "method": method,
        "frame_ts_ms": int(best_ts),
    }


# --------------------------------------------------------------------------- #
# Door detection (same model layer, door-shaped scoring)
# --------------------------------------------------------------------------- #


def score_door(mask: np.ndarray, frame: Optional[np.ndarray] = None) -> float:
    """Door-likeness score in 0..1 for a binary mask (optionally + frame).

    A door is a TALL, narrow, rectangular panel reaching toward the floor, the
    opposite geometry to a board. Wide blobs, slivers, and full-width bands are
    penalised; color uniformity is a softer factor than for boards because
    doors often carry a handle, a window, or a poster.
    """
    stats = mask_stats(mask)
    if stats is None:
        return 0.0

    aspect = stats["aspect"]  # width / height
    aspect_s = _ramp(aspect, 0.12, 0.28, 0.70, 1.05)
    position_s = _ramp(stats["cy"], 0.18, 0.32, 0.82, 0.96)
    area_s = _ramp(stats["area_frac"], 0.012, 0.025, 0.18, 0.32)
    rect_s = _clamp01((stats["rectangularity"] - 0.60) / 0.30)

    geom = 0.34 * aspect_s + 0.22 * position_s + 0.18 * area_s + 0.26 * rect_s

    if aspect > 1.1:  # wider than tall is not a door
        geom *= 0.3
    if stats["span_frac"] > 0.45:  # a door does not span most of the width
        geom *= 0.5
    if stats["area_frac"] < 0.01 or stats["area_frac"] > 0.40:
        geom *= 0.5

    if frame is not None:
        geom *= 0.55 + 0.45 * _color_score(mask, frame)

    return float(_clamp01(geom))


def _detect_door_on_frame(
    frame: np.ndarray,
) -> Optional[tuple[float, list[list[float]], str]]:
    """Door strategy chain for one frame -> (score, polygon, method) | None."""
    h, w = frame.shape[:2]
    candidates: list[tuple[float, list[list[float]], str]] = []

    proposals = [p for p in _yolo_world_proposals(frame) if p[2] >= N_BOARD_CLASSES]
    if proposals:
        masks = _sam_segment(frame, bboxes=[box for box, _, _ in proposals])
        for mask in masks:
            polygon = mask_to_polygon(mask)
            if polygon:
                candidates.append((score_door(mask, frame), polygon, "yolo_world_sam2"))

    points = [[gx * w, gy * h] for gy in DOOR_GRID_Y for gx in DOOR_GRID_X]
    for mask in _dedupe_masks(_sam_segment(frame, points=points)):
        polygon = mask_to_polygon(mask)
        if polygon:
            candidates.append((score_door(mask, frame), polygon, "sam2_geometric"))

    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])


def detect_door(video_id: str, video_path: str) -> dict:
    """Full door chain over sampled frames -> the same response contract.

    {"polygon": ...|None, "confidence": 0..1, "method": str, "frame_ts_ms": int}.
    polygon is None when nothing scores >= DOOR_MIN_SCORE.
    """
    video_path = _validate_video_path(video_path)
    frames = _sample_frames(video_path)

    best: Optional[tuple[float, list[list[float]], str]] = None
    best_ts = frames[0][0] if frames else 0
    for ts_ms, frame in frames:
        try:
            candidate = _detect_door_on_frame(frame)
        except Exception:
            logger.warning(
                "[door-detect] video %s: frame ts=%sms failed",
                video_id,
                ts_ms,
                exc_info=True,
            )
            continue
        if candidate is not None and (best is None or candidate[0] > best[0]):
            best = candidate
            best_ts = ts_ms

    if best is None:
        return {
            "polygon": None,
            "confidence": 0.0,
            "method": "sam2_geometric",
            "frame_ts_ms": int(best_ts),
        }

    score, polygon, method = best
    confidence = round(float(_clamp01(score)), 4)
    logger.info(
        "[door-detect] video %s: method=%s confidence=%.3f ts=%sms",
        video_id,
        method,
        confidence,
        best_ts,
    )
    return {
        "polygon": polygon if score >= DOOR_MIN_SCORE else None,
        "confidence": confidence,
        "method": method,
        "frame_ts_ms": int(best_ts),
    }
