"""YOLO pose detection + BoT-SORT tracking over sampled video frames.

Reads the video with cv2.VideoCapture, samples frames at
stride = max(1, round(native_fps / sample_fps)) and runs
model.track(..., persist=True, tracker='botsort.yaml', classes=[0]).

Per kept frame, per person we emit a Detection with:
- bbox {x, y, w, h} normalized, top-left based
- standing: bbox aspect h/w > 1.6 OR hip-above-knee keypoint geometry
- back_to_camera: nose/eyes keypoints low-confidence while shoulders visible
and collect up to 10 torso HSV histogram samples per raw track (>= 1 s apart)
for the identity merge stage.

Robustness: MPS failures fall back to CPU with a warning; effective sample
fps is capped at 5; the capture is always released.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.models import Detection, VideoMeta

logger = logging.getLogger(__name__)

MAX_SAMPLE_FPS = 5.0
FALLBACK_NATIVE_FPS = 30.0
MAX_HIST_SAMPLES_PER_TRACK = 10
HIST_SAMPLE_SPACING_MS = 1_000
KPT_CONF_LOW = 0.3
KPT_CONF_VISIBLE = 0.5
STANDING_ASPECT = 1.6

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14

_model = None
_fallback_cpu = False


def _lapjv_shim(cost, extend_cost=False, cost_limit=None, return_cost=True):
    """lap.lapjv-compatible solver built on ultralytics' NumPy linear_sum_assignment.

    The real 'lap' package is a project dependency and is used whenever it is
    importable; this shim is only a documented fallback (registered by
    _ensure_lap_shim when "import lap" raises ImportError). It emulates lapjv
    (same semantics as ultralytics' own use_lap=False branch): solve the
    assignment, then treat pairs with cost > cost_limit as unassigned.
    Returns (total_cost, x, y) where x[i] = assigned column or -1,
    y[j] = assigned row or -1.
    """
    from ultralytics.utils.ops import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    x = np.full(n, -1, dtype=int)
    y = np.full(m, -1, dtype=int)
    if n == 0 or m == 0:
        return 0.0, x, y

    limit = None
    if cost_limit is not None and np.isfinite(cost_limit):
        limit = float(cost_limit)
    finite = cost[np.isfinite(cost)]
    big = (float(finite.max()) if finite.size else 1.0)
    big = (max(big, limit or 0.0) + 1.0) * 10.0 + 1e6

    size = max(n, m)
    padded = np.full((size, size), big, dtype=np.float64)
    padded[:n, :m] = np.where(np.isfinite(cost), cost, big)

    rows, cols = linear_sum_assignment(padded)
    total = 0.0
    for r, c in zip(rows, cols):
        if r < n and c < m and padded[r, c] < big and (limit is None or cost[r, c] <= limit):
            x[r] = c
            y[c] = r
            total += float(cost[r, c])
    return total, x, y


def _ensure_lap_shim() -> None:
    """Register a minimal 'lap' module ONLY if the real package is unavailable.

    The real 'lap' package is installed as a primary dependency; the NumPy
    shim below is a fallback kept for environments where it cannot be built.
    """
    try:
        import lap  # noqa: F401  # real package present (primary path)

        return
    except ImportError:
        pass
    import sys
    import types

    shim = types.ModuleType("lap")
    shim.__version__ = "0.0.0-ultralytics-numpy-shim"
    shim.lapjv = _lapjv_shim
    sys.modules["lap"] = shim
    logger.warning(
        "'lap' package not installed; using NumPy lapjv shim for BoT-SORT matching"
    )


def model_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    """Effective inference device ('mps' or 'cpu')."""
    if _fallback_cpu:
        return "cpu"
    configured = get_settings().device
    if configured == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch import issues
            pass
        return "cpu"
    return configured


def _get_model():
    global _model
    if _model is None:
        _ensure_lap_shim()  # must precede any ultralytics.trackers import
        from ultralytics import YOLO

        _model = YOLO(get_settings().model_name)
    return _model


def _reset_tracker(model) -> None:
    """Reset BoT-SORT state so raw ids do not bleed across videos."""
    try:
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
    except Exception:  # pragma: no cover - defensive
        logger.warning("failed to reset tracker state", exc_info=True)


def _track_frame(model, frame: np.ndarray, device: str):
    global _fallback_cpu
    effective = "cpu" if _fallback_cpu else device
    try:
        return model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            device=effective,
            verbose=False,
        )
    except Exception as exc:
        if effective == "cpu":
            raise
        logger.warning("device %s failed (%s); falling back to cpu", effective, exc)
        _fallback_cpu = True
        return model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            device="cpu",
            verbose=False,
        )


def _validate_video_path(video_path: str) -> str:
    """Constrain video_path to a real local video file before cv2 sees it.

    cv2.VideoCapture's FFMPEG backend happily opens http://, rtsp:// and any
    filesystem path, and /analyze receives video_path from the network — an
    unvalidated value is an SSRF / arbitrary-file-read primitive. Require an
    absolute path (URLs are relative as Paths) to an existing regular file
    and, when DATA_DIR is configured in the environment, one that resolves
    inside it. Returns the resolved path.
    """
    raw = str(video_path)
    p = Path(raw)
    if not p.is_absolute():
        raise ValueError(f"video_path must be an absolute file path, got: {raw!r}")
    p = p.resolve()
    if not p.is_file():
        raise ValueError(f"video_path is not a regular file: {raw!r}")
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        base = Path(data_dir).resolve()
        if not p.is_relative_to(base):
            raise ValueError(
                f"video_path must be inside DATA_DIR ({base}): {raw!r}"
            )
    return str(p)


def _is_standing(
    w: float,
    h: float,
    kxy: Optional[np.ndarray],
    kconf: Optional[np.ndarray],
    frame_aspect: float = 1.0,
) -> bool:
    """Standing when the pixel-space bbox aspect h/w exceeds 1.6, else the
    hip-above-knee keypoint fallback.

    w and h are normalized by frame width/height (boxes.xywhn), so
    h_norm/w_norm = (h_px/w_px) * (frame_w/frame_h). Divide by frame_aspect
    (= frame_w/frame_h) to recover the pixel ratio the SPEC heuristic is
    defined on — otherwise every seated person on a 16:9 frame counts as
    standing (effective threshold 0.9) and true standing is missed on
    portrait frames.
    """
    if frame_aspect <= 0:
        frame_aspect = 1.0
    if w > 0 and (h / w) / frame_aspect > STANDING_ASPECT:
        return True
    if kxy is None or kconf is None or len(kconf) < 15:
        return False
    hip_ys = [float(kxy[i][1]) for i in (L_HIP, R_HIP) if kconf[i] > KPT_CONF_LOW]
    knee_ys = [float(kxy[i][1]) for i in (L_KNEE, R_KNEE) if kconf[i] > KPT_CONF_LOW]
    if not hip_ys or not knee_ys:
        return False
    hip_y = sum(hip_ys) / len(hip_ys)
    knee_y = sum(knee_ys) / len(knee_ys)
    return (knee_y - hip_y) > 0.25 * h


def _back_to_camera(kconf: Optional[np.ndarray]) -> bool:
    if kconf is None or len(kconf) < 7:
        return False
    face = max(float(kconf[NOSE]), float(kconf[L_EYE]), float(kconf[R_EYE]))
    shoulders = min(float(kconf[L_SHOULDER]), float(kconf[R_SHOULDER]))
    return face < KPT_CONF_LOW and shoulders > KPT_CONF_VISIBLE


def _torso_hist(
    frame: np.ndarray,
    bbox: dict,
    kxy: Optional[np.ndarray],
    kconf: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """HSV (H,S) histogram of the torso crop, L1-normalized, flattened."""
    fh, fw = frame.shape[:2]
    torso_pts = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if (
        kxy is not None
        and kconf is not None
        and len(kconf) >= 13
        and all(kconf[i] > KPT_CONF_LOW for i in torso_pts)
    ):
        xs = [float(kxy[i][0]) for i in torso_pts]
        ys = [float(kxy[i][1]) for i in torso_pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        x0 = bbox["x"] + 0.2 * bbox["w"]
        x1 = bbox["x"] + 0.8 * bbox["w"]
        y0 = bbox["y"] + 0.15 * bbox["h"]
        y1 = bbox["y"] + 0.6 * bbox["h"]

    px0 = max(0, min(fw - 1, int(x0 * fw)))
    px1 = max(0, min(fw, int(x1 * fw)))
    py0 = max(0, min(fh - 1, int(y0 * fh)))
    py1 = max(0, min(fh, int(y1 * fh)))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None

    crop = frame[py0:py1, px0:px1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]).ravel()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


def _clip_bbox(cx: float, cy: float, w: float, h: float) -> dict:
    """Clamp a normalized center-format box to the frame as an interval.

    Tracker (Kalman-filtered) boxes are not re-clipped by ultralytics and can
    extend past the frame. Clamping x/y alone shifts the stored center and
    can leave x+w > 1; clamp both edges instead so 0 <= x <= x+w <= 1 (same
    for y) and the stored center is the center of the visible region.
    """
    x0 = max(0.0, min(1.0, cx - w / 2.0))
    x1 = max(0.0, min(1.0, cx + w / 2.0))
    y0 = max(0.0, min(1.0, cy - h / 2.0))
    y1 = max(0.0, min(1.0, cy + h / 2.0))
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(x1 - x0, 5),
        "h": round(y1 - y0, 5),
    }


def _extract_frame(
    results,
    frame: np.ndarray,
    ts_ms: int,
    detections: list[Detection],
    hists: dict[int, list[np.ndarray]],
    last_hist_ms: dict[int, int],
) -> None:
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return
    ids = boxes.id
    if ids is None:  # tracker produced no ids for this frame
        return
    ids = ids.int().cpu().tolist()
    xywhn = boxes.xywhn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    kpts_xy = kpts_conf = None
    kpts = getattr(r, "keypoints", None)
    if kpts is not None and len(kpts) == len(boxes):
        try:
            kpts_xy = kpts.xyn.cpu().numpy()
            kc = kpts.conf
            kpts_conf = kc.cpu().numpy() if kc is not None else None
        except Exception:  # pragma: no cover - defensive
            kpts_xy = kpts_conf = None

    fh, fw = frame.shape[:2]
    frame_aspect = (fw / fh) if fh > 0 else 1.0

    for i, raw_id in enumerate(ids):
        cx, cy, w, h = (float(v) for v in xywhn[i])
        bbox = _clip_bbox(cx, cy, w, h)
        kxy = kpts_xy[i] if kpts_xy is not None else None
        kcf = kpts_conf[i] if kpts_conf is not None else None
        detections.append(
            Detection(
                video_ts_ms=ts_ms,
                raw_track_id=int(raw_id),
                bbox=bbox,
                conf=float(confs[i]),
                # raw (unclipped) w/h on purpose: the aspect of the full box
                # is more faithful for the standing heuristic.
                standing=_is_standing(w, h, kxy, kcf, frame_aspect),
                back_to_camera=_back_to_camera(kcf),
            )
        )

        samples = hists.setdefault(int(raw_id), [])
        if len(samples) < MAX_HIST_SAMPLES_PER_TRACK and (
            int(raw_id) not in last_hist_ms
            or ts_ms - last_hist_ms[int(raw_id)] >= HIST_SAMPLE_SPACING_MS
        ):
            hist = _torso_hist(frame, bbox, kxy, kcf)
            if hist is not None:
                samples.append(hist)
                last_hist_ms[int(raw_id)] = ts_ms


def _effective_frame_count(metadata_count: int, frames_read: int) -> int:
    """Prefer the number of frames actually decoded over container metadata.

    After the full sequential pass frames_read is exact ground truth, while
    CAP_PROP_FRAME_COUNT trusts the container (a truncated MP4 whose moov
    atom still claims the full length inflates duration_ms, diluting
    occupancy buckets and presence ratios). Metadata is only used when no
    frame was decoded at all.
    """
    if frames_read > 0 or metadata_count <= 0:
        return frames_read
    return metadata_count


def detect_video(
    video_path: str,
    sample_fps: float = 5.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> tuple[VideoMeta, list[Detection], dict[int, list[np.ndarray]]]:
    """Run detection+tracking over the video.

    video_path must be an absolute path to an existing file (inside DATA_DIR
    when that env var is set) — see _validate_video_path.
    Returns (video_meta, detections, torso_hist_samples_by_raw_track_id).
    progress_cb receives the 0..1 fraction of sampled frames processed.
    """
    video_path = _validate_video_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video file: {video_path}")

    detections: list[Detection] = []
    hists: dict[int, list[np.ndarray]] = {}
    last_hist_ms: dict[int, int] = {}
    last_ts_ms = 0

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0 or math.isnan(native_fps):
            native_fps = FALLBACK_NATIVE_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        effective_fps = max(0.5, min(float(sample_fps or MAX_SAMPLE_FPS), MAX_SAMPLE_FPS))
        stride = max(1, round(native_fps / effective_fps))
        frames_to_process = (frame_count // stride + 1) if frame_count > 0 else None

        model = _get_model()
        _reset_tracker(model)
        device = get_device()

        frame_idx = 0
        processed = 0
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    ts_ms = int(round(frame_idx / native_fps * 1000.0))
                    last_ts_ms = ts_ms
                    results = _track_frame(model, frame, device)
                    _extract_frame(
                        results, frame, ts_ms, detections, hists, last_hist_ms
                    )
                    processed += 1
                    if progress_cb and frames_to_process:
                        progress_cb(min(1.0, processed / frames_to_process))
            frame_idx += 1

        frame_count = _effective_frame_count(frame_count, frame_idx)
        duration_ms = (
            int(round(frame_count / native_fps * 1000.0)) if frame_count > 0 else last_ts_ms
        )
        meta = VideoMeta(
            duration_ms=duration_ms,
            fps=round(float(native_fps), 3),
            width=width,
            height=height,
        )
    finally:
        cap.release()

    if progress_cb:
        progress_cb(1.0)
    return meta, detections, hists
