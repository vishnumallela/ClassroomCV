"""YOLO pose detection + BoT-SORT tracking over sampled video frames.

Reads the video with cv2.VideoCapture, samples frames at
stride = max(1, round(native_fps / sample_fps)) and runs
model.track(..., persist=True, tracker='botsort.yaml', classes=[0]).

iter_frames is the frame-source seam (Kafka readiness, plan section 7 K1):
it owns path validation plus the grab/retrieve/stride loop and yields
(video_ts_ms, frame); detect_video consumes it, so a future KafkaSource only
has to reproduce the same (ts_ms, frame) contract.

Per kept frame, per person we emit a Detection with:
- bbox {x, y, w, h} normalized, top-left based
- standing: bbox aspect h/w > 1.6 OR hip-above-knee keypoint geometry
- back_to_camera: nose/eyes keypoints low-confidence while shoulders visible
and collect up to 10 torso HSV histogram samples plus up to 10 upper-body
crops per raw track (>= 1 s apart) for the identity merge stage. The crops
are batch-embedded with CLIP ViT-B/32 AFTER the frame loop (plan M5) so the
per-frame sampling cadence stays untouched.

Robustness: MPS failures fall back to CPU with a warning; effective sample
fps is capped at 5; the capture is always released.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.models import Detection, VideoMeta

logger = logging.getLogger(__name__)

MAX_SAMPLE_FPS = 5.0
FALLBACK_NATIVE_FPS = 30.0
MAX_HIST_SAMPLES_PER_TRACK = 10
HIST_SAMPLE_SPACING_MS = 1_000
# CLIP re-ID crops (plan M5): upper 60% keeps head+shoulders+torso (the parts
# that separate same-uniform people) and drops legs/desk clutter; 224 matches
# CLIP's input resolution, so bigger crops only waste memory while they wait
# for the post-loop batch embed.
CLIP_CROP_UPPER_FRAC = 0.6
CLIP_CROP_MAX_SIDE = 224
CLIP_BATCH_SIZE = 64
CLIP_MODEL_NAME = "ViT-B/32"
KPT_CONF_LOW = 0.3
KPT_CONF_VISIBLE = 0.5
STANDING_ASPECT = 1.6
# The hip/knee standing fallback needs spatially meaningful keypoints: demand
# higher keypoint confidence than the general 0.3 gate AND a box at least
# ~90 px tall on a 1440p frame. Below that the geometry is noise and the
# aspect-only result is more trustworthy.
STANDING_KPT_CONF = 0.4
STANDING_MIN_BOX_H = 90 / 1440

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14

_model = None
_fallback_cpu = False
_clip_bundle = None  # (model, preprocess, device), lazy like _model


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
    settings = get_settings()
    kwargs = dict(
        persist=True,
        tracker=settings.tracker_cfg,
        classes=[0],
        imgsz=settings.imgsz,
        conf=settings.det_conf,
        max_det=settings.max_det,
        verbose=False,
    )
    try:
        # fp16 only on the GPU path; the cpu fallback stays fp32.
        return model.track(frame, device=effective, half=effective == "mps", **kwargs)
    except Exception as exc:
        if effective == "cpu":
            raise
        logger.warning("device %s failed (%s); falling back to cpu", effective, exc)
        _fallback_cpu = True
        return model.track(frame, device="cpu", half=False, **kwargs)


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
    if h < STANDING_MIN_BOX_H:
        return False
    hip_ys = [float(kxy[i][1]) for i in (L_HIP, R_HIP) if kconf[i] > STANDING_KPT_CONF]
    knee_ys = [float(kxy[i][1]) for i in (L_KNEE, R_KNEE) if kconf[i] > STANDING_KPT_CONF]
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


def _get_clip():
    """Lazily load + cache CLIP ViT-B/32 on the detection device.

    Loaded on first use only (the frame loop never touches it), so videos
    processed before any re-ID work pay no startup cost, and tests that fake
    detect_video never trigger the checkpoint load.
    """
    global _clip_bundle
    if _clip_bundle is None:
        import clip

        device = get_device()
        model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        model.eval()
        _clip_bundle = (model, preprocess, device)
    return _clip_bundle


def _upper_crop(frame: np.ndarray, bbox: dict) -> Optional[np.ndarray]:
    """BGR crop of the upper 60% of the bbox, downscaled to <= 224 px."""
    fh, fw = frame.shape[:2]
    px0 = max(0, min(fw - 1, int(bbox["x"] * fw)))
    px1 = max(0, min(fw, int((bbox["x"] + bbox["w"]) * fw)))
    py0 = max(0, min(fh - 1, int(bbox["y"] * fh)))
    py1 = max(0, min(fh, int((bbox["y"] + bbox["h"] * CLIP_CROP_UPPER_FRAC) * fh)))
    # Below ~8 px the crop is compression mush that embeds as noise.
    if px1 - px0 < 8 or py1 - py0 < 8:
        return None
    crop = frame[py0:py1, px0:px1]
    scale = CLIP_CROP_MAX_SIDE / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _embed_tracks(crops: dict[int, list[np.ndarray]]) -> dict[int, list[float]]:
    """L2-normalized median CLIP embedding (512 floats) per raw track.

    One batched post-pass over all sampled crops (~1560 for the demo video at
    batch 64), so the 5 fps detection loop stays untouched. Failures degrade
    to {} instead of raising: losing re-ID evidence must never discard a
    completed multi-minute YOLO pass — the merge falls back to hist+spatial.
    """
    flat: list[tuple[int, np.ndarray]] = [
        (raw_id, crop) for raw_id, samples in crops.items() for crop in samples
    ]
    if not flat:
        return {}
    try:
        import torch
        from PIL import Image

        model, preprocess, device = _get_clip()
        tensors = [
            # cv2 frames are BGR; CLIP's preprocess expects an RGB PIL image.
            preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            for _raw_id, crop in flat
        ]
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(tensors), CLIP_BATCH_SIZE):
                batch = torch.stack(tensors[i : i + CLIP_BATCH_SIZE]).to(device)
                feats.append(model.encode_image(batch).float().cpu().numpy())
        all_feats = np.concatenate(feats, axis=0)
    except Exception:
        logger.warning("CLIP track embedding failed; merge will run without embeds", exc_info=True)
        return {}

    by_id: dict[int, list[np.ndarray]] = {}
    for (raw_id, _crop), feat in zip(flat, all_feats):
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            # Normalize per sample so the median averages directions, not
            # magnitudes (CLIP feature norms vary with crop content).
            by_id.setdefault(raw_id, []).append(feat / norm)
    out: dict[int, list[float]] = {}
    for raw_id, samples in by_id.items():
        med = np.median(np.stack(samples), axis=0)
        norm = float(np.linalg.norm(med))
        if norm > 0:
            # Median of unit vectors is not unit; re-normalize so downstream
            # dot products are true cosines.
            out[raw_id] = [float(v) for v in med / norm]
    return out


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
    crops: dict[int, list[np.ndarray]],
    last_crop_ms: dict[int, int],
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

        # CLIP crop sampling mirrors the hist cadence but tracks its own
        # last-sample time: a failed torso hist (tiny box) must not stall or
        # accelerate crop collection, and vice versa.
        crop_samples = crops.setdefault(int(raw_id), [])
        if len(crop_samples) < MAX_HIST_SAMPLES_PER_TRACK and (
            int(raw_id) not in last_crop_ms
            or ts_ms - last_crop_ms[int(raw_id)] >= HIST_SAMPLE_SPACING_MS
        ):
            crop = _upper_crop(frame, bbox)
            if crop is not None:
                crop_samples.append(crop)
                last_crop_ms[int(raw_id)] = ts_ms


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


@dataclass
class FrameSourceInfo:
    """Source properties iter_frames fills in for the caller's meta/progress
    math: capture properties before the first yield, frames_read once the
    source is exhausted or closed."""

    native_fps: float = FALLBACK_NATIVE_FPS
    width: int = 0
    height: int = 0
    metadata_frame_count: int = 0
    frames_to_process: Optional[int] = None
    frames_read: int = 0


def iter_frames(
    video_path: str,
    sample_fps: float,
    info: Optional[FrameSourceInfo] = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """File-backed frame source (the Kafka seam, plan section 7 K1).

    Validates video_path via _validate_video_path, then yields
    (video_ts_ms, BGR frame) for every stride-th decodable frame, with
    stride = max(1, round(native_fps / effective_sample_fps)) and
    ts_ms = round(frame_idx / native_fps * 1000): the exact math the
    detect_video loop always ran. Any future source (Kafka/RTSP) only has to
    reproduce this (ts_ms, frame) contract. The capture is released when the
    generator is exhausted, closed, or unwound by an exception.
    """
    video_path = _validate_video_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video file: {video_path}")

    frame_idx = 0
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0 or math.isnan(native_fps):
            native_fps = FALLBACK_NATIVE_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        effective_fps = max(0.5, min(float(sample_fps or MAX_SAMPLE_FPS), MAX_SAMPLE_FPS))
        stride = max(1, round(native_fps / effective_fps))

        if info is not None:
            info.native_fps = float(native_fps)
            info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            info.metadata_frame_count = frame_count
            info.frames_to_process = (
                (frame_count // stride + 1) if frame_count > 0 else None
            )

        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield int(round(frame_idx / native_fps * 1000.0)), frame
            frame_idx += 1
    finally:
        if info is not None:
            info.frames_read = frame_idx
        cap.release()


def detect_video(
    video_path: str,
    sample_fps: float = 5.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> tuple[VideoMeta, list[Detection], dict[int, list[np.ndarray]], dict[int, list[float]]]:
    """Run detection+tracking over the video.

    video_path must be an absolute path to an existing file (inside DATA_DIR
    when that env var is set); see _validate_video_path.
    Returns (video_meta, detections, torso_hist_samples_by_raw_track_id,
    clip_embed_by_raw_track_id) where each embed is the L2-normalized median
    CLIP ViT-B/32 vector of the track's sampled upper-body crops.
    progress_cb receives the 0..1 fraction of sampled frames processed.
    """
    detections: list[Detection] = []
    hists: dict[int, list[np.ndarray]] = {}
    last_hist_ms: dict[int, int] = {}
    crops: dict[int, list[np.ndarray]] = {}
    last_crop_ms: dict[int, int] = {}
    last_ts_ms = 0

    model = _get_model()
    _reset_tracker(model)
    device = get_device()

    info = FrameSourceInfo()
    frames = iter_frames(video_path, sample_fps, info=info)
    processed = 0
    try:
        for ts_ms, frame in frames:
            last_ts_ms = ts_ms
            results = _track_frame(model, frame, device)
            _extract_frame(
                results, frame, ts_ms, detections, hists, last_hist_ms, crops, last_crop_ms
            )
            processed += 1
            if progress_cb and info.frames_to_process:
                progress_cb(min(1.0, processed / info.frames_to_process))
    finally:
        frames.close()

    embeds = _embed_tracks(crops)

    frame_count = _effective_frame_count(info.metadata_frame_count, info.frames_read)
    duration_ms = (
        int(round(frame_count / info.native_fps * 1000.0)) if frame_count > 0 else last_ts_ms
    )
    meta = VideoMeta(
        duration_ms=duration_ms,
        fps=round(float(info.native_fps), 3),
        width=info.width,
        height=info.height,
    )

    if progress_cb:
        progress_cb(1.0)
    return meta, detections, hists, embeds
