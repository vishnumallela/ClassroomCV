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

GPU serving (RunPod): device 'auto' resolves cuda > mps > cpu. On cuda the
model is served as a TensorRT engine when one exists next to the weight (and
TENSORRT_EXPORT=true builds it once at first load). REQUIRE_DEVICE=cuda makes
a mis-provisioned pod fail loud instead of silently billing 20x the wall-clock
on CPU. Engines never fall back to CPU (TensorRT is CUDA-only); .pt weights
keep the dev-friendly MPS/CPU fallback.

Robustness: effective sample fps is capped at 5; the capture is always
released; media downloads resume from the last byte written.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
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
# Appearance crops for re-identification. The upper 60% (head+shoulders+torso)
# is what CLIP wanted; a purpose-built person re-ID encoder is trained on FULL
# bodies, so REID_CROP_UPPER_FRAC = 1.0 when one is in use.
#
# Why the encoder changed: raw CLIP is at CHANCE for person re-identification
# (published zero-shot mAP on MSMT17: CLIP-B/32 = 0.10, and scaling to L/14
# only reaches 0.14), which is exactly what we measured on our own footage —
# different people sat at cosine 0.82-0.89, leaving no dynamic range to
# threshold. The ultralytics-native re-ID encoder puts different people at
# 0.12-0.23 and two views of one person at ~0.64 on the same crops: roughly
# three times the margin, no new licence, and no new dependency beyond the
# ONNX runtime.
CLIP_CROP_UPPER_FRAC = 0.6
REID_CROP_UPPER_FRAC = 1.0
# OFF BY DEFAULT, pending a measurement that separates two confounded changes.
# Set REID_MODEL=yolo26s-reid.onnx to enable (weights auto-download, 28 MB, no
# new licence obligation beyond ultralytics itself).
#
# Measured with the encoder enabled, against per-frame ground truth on two
# lessons:
#     Khaitan (tuned on):  coverage 91.4 -> 78.9, purity 98.0 -> 81.6
#     Demo    (held out):  coverage 59.9 -> 64.3, purity 80.6 -> 96.9
# It clearly helps the room it was not tuned against and clearly hurts the one
# it was. Neither configuration dominates, and the run that produced those
# numbers ALSO changed the torso histogram, so the two effects are confounded.
# The honest next experiment is to carry re-ID, CLIP and colour as three
# separate modalities and let the fusion weigh them, rather than swapping one
# for another. Until that is measured, the default is the configuration whose
# numbers we actually verified.
REID_MODEL_DEFAULT = ""
CLIP_CROP_MAX_SIDE = 224
CLIP_BATCH_SIZE = 64
CLIP_MODEL_NAME = "ViT-B/32"
# Occlusion gate for appearance sampling (see _appearance_sample_ok): a crop
# where a third of the body is behind someone else already contains more of the
# occluder than of the subject.
OCCLUSION_SAMPLE_MAX = 0.35
# ...relaxed only while a track still has almost no appearance evidence, so a
# permanently half-hidden back-row pupil is still represented (badly) rather
# than not at all.
OCCLUSION_SAMPLE_FALLBACK = 0.65
OCCLUSION_FALLBACK_UNTIL = 3
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
_model_is_engine = False  # set at load; engines are CUDA-only (no CPU fallback)
_fallback_cpu = False
_clip_bundle = None  # (model, preprocess, device), lazy like _model
_reid_encoder = None
_reid_failed = False


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


def clip_loaded() -> bool:
    """True when the CLIP checkpoint is already resident in this process.

    Callers use this to keep optional appearance work free: a path that has
    not embedded anything must not pull a 350 MB checkpoint onto the box just
    to add one more signal.
    """
    return _clip_bundle is not None


# Best YOLO26 pose weight per resolved device when Settings.model_name is
# 'auto'. YOLO26 is NMS-free and reports up to +7.2 pose AP over YOLO11; a GPU
# affords the x variant, while mps/cpu stay on m for fast dev iteration. Any of
# these is overridable via MODEL_NAME (a .pt weight or a TensorRT .engine).
_DEVICE_MODEL_DEFAULT = {
    "cuda": "yolo26x-pose.pt",
    "mps": "yolo26m-pose.pt",
    "cpu": "yolo26m-pose.pt",
}


def get_device() -> str:
    """Effective inference device: 'cuda', 'mps', or 'cpu'.

    Resolves Settings.device. 'auto' prefers cuda, then mps, then cpu; an
    explicit device is honoured but degrades to cpu when unavailable so a
    misconfigured box does not crash. Once a runtime failure trips
    _fallback_cpu, cpu is pinned for the rest of the process.
    """
    if _fallback_cpu:
        return "cpu"
    configured = (get_settings().device or "auto").strip().lower()
    cuda_ok = mps_ok = False
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover - torch import issues
        pass
    if configured in ("", "auto"):
        return "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    if configured == "cuda":
        return "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    if configured == "mps":
        return "mps" if mps_ok else "cpu"
    return configured  # explicit 'cpu' or a specific 'cuda:N' device string


def _assert_required_device(resolved: str) -> None:
    """Fail loud when REQUIRE_DEVICE is set and the box resolved elsewhere.

    A RunPod worker that silently degrades to CPU still bills wall-clock — at
    ~20x the runtime that is the whole GPU budget gone on one job. Production
    sets REQUIRE_DEVICE=cuda so a mis-provisioned pod dies at load instead.
    """
    required = (get_settings().require_device or "").strip().lower()
    if required and resolved.split(":", 1)[0] != required:
        raise RuntimeError(
            f"REQUIRE_DEVICE={required!r} but the resolved inference device is "
            f"{resolved!r}; refusing to run on the wrong device"
        )


def resolve_model_name() -> str:
    """The YOLO weight to load: Settings.model_name, or the best device default
    when it is 'auto'/empty. Auto-resolved weights live under WEIGHTS_DIR when
    set (the RunPod volume — the container layer is recreated on every pod
    start, so a CWD cache would re-download and re-export each time). On cuda
    an already-exported TensorRT engine next to the weight is preferred (built
    by TENSORRT_EXPORT or scripts/export_tensorrt.py); an explicit MODEL_NAME
    is always honoured verbatim."""
    settings = get_settings()
    configured = (settings.model_name or "").strip()
    if configured and configured.lower() != "auto":
        return configured
    base = get_device().split(":", 1)[0]  # 'cuda:0' -> 'cuda'
    name = _DEVICE_MODEL_DEFAULT.get(base, "yolo26m-pose.pt")
    weights_dir = (settings.weights_dir or "").strip()
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        name = str(Path(weights_dir) / name)
    if base == "cuda":
        engine = Path(name).with_suffix(".engine")
        if engine.is_file():
            return str(engine)
    return name


def _export_engine(weight: str, device: str) -> Optional[str]:
    """One-time TensorRT export of a .pt weight; returns the engine path.

    half=True is the ~free 2x; dynamic=True lets IMGSZ move between 1280 and
    1536 without a rebuild. Engines are specific to the GPU model and TensorRT
    version, so this runs on the pod itself (first load, several minutes) and
    the result is cached on disk for every later start. Failure degrades to
    the .pt weight — a batch job must not die because an export did.
    """
    from ultralytics import YOLO

    try:
        logger.info(
            "TensorRT export of %s starting (one-time on this GPU, several minutes)",
            weight,
        )
        exported = YOLO(weight).export(
            format="engine",
            half=True,
            dynamic=True,
            batch=1,
            imgsz=get_settings().imgsz,
            device=device,
        )
        logger.info("TensorRT export complete: %s", exported)
        return str(exported)
    except Exception:
        logger.warning(
            "TensorRT export failed; serving the PyTorch weight instead", exc_info=True
        )
        return None


def _precision_kwargs(device: str) -> dict:
    """The fp16 request for predict/track, or {} when it must not be asked for.

    MUST be passed to the FIRST predict/track call of the process. ultralytics
    builds its AutoBackend exactly once -- predictor.setup_model() runs only
    `if not self.model`, and it computes `fp16=self.args.quantize == 16` at that
    moment. Every later call merges args but never rebuilds the backend, so a
    warmup that omits this pins the weights to fp32 for the whole run and the
    fp16 asked for on the real frames is silently ignored.

    'quantize' rather than the older 'half': half is a deprecated alias in the
    pinned ultralytics (it is not even in DEFAULT_CFG_DICT any more) and warns
    on every call. cpu stays fp32 -- fp16 on cpu is pathologically slow, not
    faster. Engine precision is baked at export, so the flag is meaningless
    there.
    """
    if _model_is_engine or device.split(":", 1)[0] == "cpu":
        return {}
    return {"quantize": 16}


def _record_precision(model, device: str) -> None:
    """Log the precision the backend ACTUALLY loaded, read back from the weights.

    Deliberately not read from predictor.args.quantize: that reports the value
    we asked for (16) even when the backend was already built fp32, so it shows
    a green light on exactly the broken configuration this guards against. The
    parameter dtype is the ground truth. Best-effort -- a diagnostic must never
    be the thing that fails an analysis.
    """
    if _model_is_engine:
        return
    import torch  # lazy, like every other torch use here (module import is hot)

    try:
        backend = model.predictor.model
        inner = getattr(backend, "backend", backend)
        module = getattr(inner, "model", None)
        dtype = next(module.parameters()).dtype if module is not None else None
    except Exception:
        logger.warning("could not read back inference precision", exc_info=True)
        return
    if dtype is None:
        return
    want_fp16 = bool(_precision_kwargs(device))
    logger.info("inference precision on %s: %s", device, dtype)
    if want_fp16 and dtype is not torch.float16:
        # Not fatal: the run is correct, just about 2x slower than it should be
        # and 2x the VRAM. Loud because it is otherwise invisible -- it was
        # shipped this way and nothing in the product surfaced it.
        logger.error(
            "DEGRADED: asked for fp16 on %s but the backend loaded %s; "
            "inference will run at roughly half speed",
            device,
            dtype,
        )


def _warmup(model, device: str) -> None:
    """Run one dummy inference so engine deserialization / cuDNN autotune has
    happened before the first billed frame. Best-effort: a warmup failure is
    logged and the real frames decide.

    This call also DECIDES the backend precision for the entire process (see
    _precision_kwargs), so it must carry the same precision the real frames
    will ask for.
    """
    try:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        model.predict(
            dummy,
            imgsz=get_settings().imgsz,
            device=device,
            verbose=False,
            **_precision_kwargs(device),
        )
        logger.info("warmup inference complete on %s", device)
    except Exception:
        logger.warning("warmup inference failed", exc_info=True)


def _get_model():
    global _model, _model_is_engine
    if _model is None:
        _ensure_lap_shim()  # must precede any ultralytics.trackers import
        from ultralytics import YOLO

        device = get_device()
        _assert_required_device(device)
        name = resolve_model_name()
        if (
            get_settings().tensorrt_export
            and device.split(":", 1)[0] == "cuda"
            and name.endswith(".pt")
        ):
            engine = Path(name).with_suffix(".engine")
            if engine.is_file():
                name = str(engine)
            else:
                name = _export_engine(name, device) or name
        _model_is_engine = name.endswith(".engine")
        logger.info(
            "loading pose model %s on device %s%s",
            name,
            device,
            " (TensorRT engine)" if _model_is_engine else "",
        )
        # Engines carry no task metadata, so tell ultralytics this is pose.
        _model = YOLO(name, task="pose") if _model_is_engine else YOLO(name)
        # Every non-cpu device, not just cuda: the warmup is what fixes the
        # backend precision, so skipping it on mps meant the dev Macs and the
        # billed GPU pod ran DIFFERENT precisions (mps got fp16 because its
        # first call was the real track(); cuda got fp32 from this warmup).
        # Running it everywhere makes the pod's path testable on a laptop.
        if device.split(":", 1)[0] != "cpu":
            _warmup(_model, device)
            _record_precision(_model, device)
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
    # Same helper the warmup used, so the request can never drift from the
    # precision the backend was actually built with.
    kwargs.update(_precision_kwargs(effective))
    try:
        return model.track(frame, device=effective, **kwargs)
    except Exception as exc:
        if effective == "cpu":
            raise
        if _model_is_engine or (settings.require_device or "").strip():
            # A TensorRT engine cannot run on CPU, and a REQUIRE_DEVICE box
            # must fail loud rather than silently burn 20x the wall-clock.
            raise
        logger.warning("device %s failed (%s); falling back to cpu", effective, exc)
        _fallback_cpu = True
        kwargs.pop("quantize", None)  # cpu stays fp32
        return model.track(frame, device="cpu", **kwargs)


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


def _media_url_host_allowed(url: str) -> bool:
    """True when the URL's host[:port] is in Settings.media_url_allowlist.

    The allowlist is the SSRF gate for object-store fetches: cv2/ffmpeg will
    open ANY http/rtsp URL, and video_path arrives over the network, so only an
    explicitly configured object-store host may be fetched.
    """
    allow = {
        h.strip()
        for h in (get_settings().media_url_allowlist or "").split(",")
        if h.strip()
    }
    if not allow:
        return False
    return urllib.parse.urlparse(url).netloc in allow


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop against media_url_allowlist.

    urllib follows 3xx redirects by default, and the allowlist is only checked
    on the INITIAL url. Without this, an allowlisted origin could 301/302 the
    fetch to an internal address (169.254.169.254, an intranet host) that the
    allowlist never saw -- the classic SSRF-via-redirect bypass. Here every hop
    is re-validated, so a redirect to a non-allowlisted host is refused.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        if not _media_url_host_allowed(newurl):
            raise ValueError(
                f"media URL redirected to a non-allowlisted host: {newurl!r}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Opener whose only redirect handler re-validates the allowlist on each hop.
_MEDIA_URL_OPENER = urllib.request.build_opener(_AllowlistRedirectHandler)
# Cap the download so a huge (or malicious) allowlisted object cannot exhaust
# disk. Defaults to 8 GiB (2x the API's 4 GiB upload cap, for headroom);
# override with MEDIA_MAX_DOWNLOAD_BYTES.
_MEDIA_MAX_DOWNLOAD_BYTES = int(os.environ.get("MEDIA_MAX_DOWNLOAD_BYTES", 8 * 1024**3))
_DOWNLOAD_CHUNK = 1024 * 1024
# A multi-GB object crossing the WAN to a RunPod worker stalls sooner or later;
# resuming with a Range header from the byte already on disk keeps the work
# done instead of restarting a whole camera-day transfer from zero.
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_RETRIES = 5


def _content_total(resp) -> Optional[int]:
    """Total object size from Content-Range (a 206) or Content-Length (a 200).

    None when the response carries neither; the caller then cannot detect a
    truncated stream and accepts what it got.
    """
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    cr = headers.get("Content-Range")
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    cl = headers.get("Content-Length")
    if cl and str(cl).isdigit():
        return int(cl)
    return None


def _download_to_temp(url: str) -> str:
    """Stream an allowlisted media URL to a temp file (chunked, no full buffer).

    Redirects are re-validated against the allowlist on every hop, and the total
    byte count is capped, so neither an SSRF-via-redirect nor an oversized object
    can slip through. A stalled transfer resumes from the byte already written
    (Range request); a server that ignores the Range restarts cleanly. Placed
    inside DATA_DIR when configured so the downloaded copy also satisfies
    _validate_video_path's DATA_DIR containment when the pipeline re-validates.
    """
    data_dir = os.environ.get("DATA_DIR")
    tmp_dir = data_dir if data_dir and os.path.isdir(data_dir) else tempfile.gettempdir()
    fd, tmp = tempfile.mkstemp(prefix="mediacache_", suffix=".mp4", dir=tmp_dir)
    os.close(fd)
    written = 0
    total: Optional[int] = None
    try:
        for attempt in range(_DOWNLOAD_RETRIES):
            # noqa: S310 (host allowlisted, redirects re-validated per hop)
            req = urllib.request.Request(url)
            if written:
                req.add_header("Range", f"bytes={written}-")
            try:
                with _MEDIA_URL_OPENER.open(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
                    mode = "ab"
                    if written and getattr(resp, "status", 200) != 206:
                        # Server ignored the Range: start over rather than
                        # appending a second copy of the whole object.
                        written, mode = 0, "wb"
                    if total is None:
                        total = _content_total(resp)
                    with open(tmp, mode) as out:
                        while True:
                            chunk = resp.read(_DOWNLOAD_CHUNK)
                            if not chunk:
                                break
                            if written + len(chunk) > _MEDIA_MAX_DOWNLOAD_BYTES:
                                raise ValueError(
                                    "media download exceeds "
                                    f"{_MEDIA_MAX_DOWNLOAD_BYTES} byte cap: {url!r}"
                                )
                            out.write(chunk)
                            # Advance only AFTER the bytes are on disk: the
                            # resume offset must never run ahead of the file,
                            # or a failed write leaves a silent hole.
                            written += len(chunk)
                if total is None or written >= total:
                    return tmp
                # Short read without an exception is still a truncated file.
                raise TimeoutError(f"stream ended at {written} of {total} bytes")
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                if attempt == _DOWNLOAD_RETRIES - 1:
                    raise
                logger.warning(
                    "media download stalled at %d/%s bytes (%s); resuming (attempt %d/%d)",
                    written,
                    total if total is not None else "?",
                    exc,
                    attempt + 2,
                    _DOWNLOAD_RETRIES,
                )
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return tmp


def resolve_video_source(video_path: str) -> tuple[str, bool]:
    """Resolve a request's video_path to a LOCAL file: returns (path, is_temp).

    An http(s) URL whose host is allowlisted is downloaded to a temp file and
    returned with is_temp=True (the caller unlinks it when done). This lets a
    remote GPU worker fetch the video straight from MinIO/S3 by presigned URL
    instead of the API node writing it to a shared filesystem, while cv2 still
    decodes a LOCAL file (reliable over a multi-minute sequential read, unlike
    streaming the whole video frame-by-frame over HTTP). Any other value goes
    through the local-path SSRF guard (_validate_video_path); a non-allowlisted
    URL is rejected there via ValueError.
    """
    if video_path.startswith(("http://", "https://")):
        if not _media_url_host_allowed(video_path):
            raise ValueError(
                f"media URL host is not in media_url_allowlist: {video_path!r}"
            )
        return _download_to_temp(video_path), True
    return _validate_video_path(video_path), False


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


def _occlusions(
    raw: list[tuple[float, float, float, float]]
) -> list[float]:
    """Per-box occlusion in 0..1 for one frame's boxes (center-format, normalized).

    Two things hide a person in a classroom, and both corrupt every downstream
    reading of that box (appearance crop, height, posture):

    COVERED   a person BETWEEN them and the camera. On a fixed overhead camera
              the ground-plane rule is reliable: whoever's feet are lower in
              the frame is nearer, so only boxes with a lower bottom edge can
              occlude. Contribution is summed (a back-row pupil is commonly
              hidden by two people at once) and clamped at 1.
    TRUNCATED the frame edge cuts the box off — the teacher half out of shot at
              the door reads as a short, cropped person.

    Combined as independent losses of visibility: 1 - (1-covered)(1-truncated).
    O(n^2) over the ~30 boxes of one frame is a few microseconds.
    """
    n = len(raw)
    out = [0.0] * n
    boxes = []
    for cx, cy, w, h in raw:
        boxes.append((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0))
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        area = max(1e-9, (x1 - x0) * (y1 - y0))
        covered = 0.0
        for j, (a0, b0, a1, b1) in enumerate(boxes):
            if i == j or b1 <= y1:  # only nearer-to-camera boxes occlude
                continue
            iw = min(x1, a1) - max(x0, a0)
            ih = min(y1, b1) - max(y0, b0)
            if iw > 0 and ih > 0:
                covered += iw * ih / area
        covered = min(1.0, covered)
        visible_w = max(0.0, min(1.0, x1) - max(0.0, x0))
        visible_h = max(0.0, min(1.0, y1) - max(0.0, y0))
        truncated = max(0.0, 1.0 - (visible_w * visible_h) / area)
        out[i] = round(1.0 - (1.0 - covered) * (1.0 - truncated), 4)
    return out


def _body_ratios(
    kxy: Optional[np.ndarray],
    kconf: Optional[np.ndarray],
    frame_aspect: float,
) -> Optional[dict]:
    """Scale-free body proportions, or None when the keypoints cannot support them.

    Adults and children differ in PROPORTION, not just size: a child's head is
    roughly 1/6 of their stature and an adult's about 1/8, and adult legs are
    longer relative to the torso. Ratios of keypoint distances are invariant to
    how far the person is from the camera, which is exactly what raw bbox
    height is not — the front-row pupil is the tallest box in the room.

    - head: nose-to-shoulder-line distance over torso length (shoulders to
      hips). Larger for children.
    - leg: hip-to-ankle (falling back to twice hip-to-knee when the feet are
      under a desk) over torso length. Larger for adults.
    - vis: fraction of the 17 keypoints the model is confident about, so
      consumers can discount a measurement taken through an occlusion.

    xyn is normalized by frame width/height independently, so x is rescaled by
    the frame aspect before any distance is taken; otherwise a 16:9 frame
    stretches every horizontal component by 1.78.
    """
    if kxy is None or kconf is None or len(kconf) < 17:
        return None
    conf = np.asarray(kconf, dtype=np.float64)
    pts = np.asarray(kxy, dtype=np.float64).copy()
    if frame_aspect > 0:
        pts[:, 0] *= frame_aspect

    def ok(*idx: int) -> bool:
        return all(conf[i] > STANDING_KPT_CONF for i in idx)

    def mid(a: int, b: int) -> np.ndarray:
        return (pts[a] + pts[b]) / 2.0

    if not ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return None
    shoulder = mid(L_SHOULDER, R_SHOULDER)
    hip = mid(L_HIP, R_HIP)
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 1e-4:
        return None

    out: dict = {"vis": round(float((conf > STANDING_KPT_CONF).mean()), 3)}
    if conf[NOSE] > STANDING_KPT_CONF:
        out["head"] = round(float(np.linalg.norm(pts[NOSE] - shoulder)) / torso, 4)
    ankles = [i for i in (15, 16) if conf[i] > STANDING_KPT_CONF]
    knees = [i for i in (L_KNEE, R_KNEE) if conf[i] > STANDING_KPT_CONF]
    if ankles:
        foot = pts[ankles].mean(axis=0)
        out["leg"] = round(float(np.linalg.norm(foot - hip)) / torso, 4)
    elif knees:
        # Feet under a desk: the knee is half the leg, doubled to estimate it.
        knee = pts[knees].mean(axis=0)
        out["leg"] = round(2.0 * float(np.linalg.norm(knee - hip)) / torso, 4)
    return out if len(out) > 1 else None


def _torso_hist(
    frame: np.ndarray,
    bbox: dict,
    kxy: Optional[np.ndarray],
    kconf: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Coarse HSV histogram of the torso crop, L1-normalized, flattened.

    All three channels, not hue and saturation only. Hue is meaningless where
    there is no light: a teacher in a dark kurta among bright school polos
    produced a diffuse, noisy H-S histogram that looked like everybody and
    nobody, and the "who is not wearing the uniform" signal — the strongest
    appearance cue in a uniformed classroom — collapsed to zero on that
    lesson. Brightness is what makes dark clothing describable.

    MEASURED, and kept at 30 hue x 32 saturation. Adding a brightness axis and
    coarsening hue to 12x6x6 was tried, to describe a teacher in a dark kurta
    whose hue is meaningless — and it was much worse on both lessons
    (coverage 91.4 -> 77.6 and 59.9 -> 10.0), because hue resolution is exactly
    what separates one uniform colour from another, and brightness mostly
    encodes where in the room somebody is standing. The dark-clothing case
    needs a better encoder, not a coarser histogram.
    """
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


@dataclass
class _SampleState:
    """Reservoir state for one track's appearance samples."""

    spacing_ms: int
    last_ms: int
    last_occlusion: float


def _due(
    state: dict[int, _SampleState], raw_id: int, ts_ms: int, occlusion: float
) -> bool:
    """Is this frame worth cropping for track `raw_id`?

    True either because the current interval has elapsed, or because this view
    is cleaner than the one already held for the current interval.
    """
    st = state.get(raw_id)
    if st is None:
        return True
    return ts_ms - st.last_ms >= st.spacing_ms or occlusion < st.last_occlusion


def _offer(
    store: dict[int, list],
    state: dict[int, _SampleState],
    raw_id: int,
    ts_ms: int,
    occlusion: float,
    value,
) -> None:
    """Add a sample under a reservoir that stays spread over the WHOLE track.

    The sampler this replaced took the first ten crops one second apart and
    then stopped for good. Measured on a real lesson, that left each track's
    appearance evidence covering a median of EIGHT PERCENT of its lifetime —
    the teacher's 367-second track carried nine seconds of gallery. Everything
    downstream inherited that: matching a tracklet from minute nine against a
    prototype built entirely from minute one, and a change-of-clothing
    detector that could only ever look inside the first ten seconds of an id,
    which is precisely where a mid-track handoff never happens.

    Halve-and-double keeps the same ten samples and the same memory, in one
    streaming pass: when the buffer fills, drop every other sample and double
    the interval. Within an interval the LEAST-OCCLUDED view wins, because
    crop quality is what sets the re-identification ceiling.

    Measured against per-frame ground truth on two lessons: teacher coverage
    91.4 -> 98.0% and 59.9 -> 81.7%, re-acquisition after leaving frame 67% and
    80% -> 100% on both.
    """
    samples = store.setdefault(raw_id, [])
    st = state.get(raw_id)
    if st is not None and samples and ts_ms - st.last_ms < st.spacing_ms:
        # Same interval, cleaner view: replace rather than spend a slot.
        if occlusion < st.last_occlusion:
            samples[-1] = value
            st.last_occlusion = occlusion
        return

    samples.append(value)
    if st is None:
        state[raw_id] = _SampleState(
            spacing_ms=HIST_SAMPLE_SPACING_MS, last_ms=ts_ms, last_occlusion=occlusion
        )
        return
    st.last_ms = ts_ms
    st.last_occlusion = occlusion
    if len(samples) > MAX_HIST_SAMPLES_PER_TRACK:
        # Decimate to every other sample and stretch the interval to match, so
        # the retained set stays evenly spread however long the track runs.
        store[raw_id] = samples[::2]
        st.spacing_ms *= 2


def _appearance_sample_ok(occlusion: float, samples_so_far: int) -> bool:
    """May this frame contribute appearance evidence for this track?

    Clean views only, except while a track is still nearly evidence-free: then
    a heavily occluded view is still better than none (it can only be vetoed
    against, never used to claim a match).
    """
    if occlusion <= OCCLUSION_SAMPLE_MAX:
        return True
    return (
        samples_so_far < OCCLUSION_FALLBACK_UNTIL
        and occlusion <= OCCLUSION_SAMPLE_FALLBACK
    )


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


def _get_reid():
    """Lazily load the person re-ID encoder, or None when unavailable.

    Failure is remembered and degrades to CLIP rather than raising: losing the
    better embedding must never cost a completed multi-minute detection pass.
    """
    global _reid_encoder, _reid_failed
    if _reid_encoder is None and not _reid_failed:
        name = (os.environ.get("REID_MODEL") or REID_MODEL_DEFAULT).strip()
        if not name:
            _reid_failed = True
            return None
        try:
            from ultralytics.trackers.utils.reid import ReID

            weights_dir = (get_settings().weights_dir or "").strip()
            path = str(Path(weights_dir) / name) if weights_dir else name
            _reid_encoder = ReID(path, device=get_device())
            logger.info("re-ID encoder loaded: %s", path)
        except Exception:
            logger.warning(
                "person re-ID encoder unavailable; falling back to CLIP "
                "(which is near-chance for re-identification)",
                exc_info=True,
            )
            _reid_failed = True
    return _reid_encoder


def _upper_crop(frame: np.ndarray, bbox: dict) -> Optional[np.ndarray]:
    """BGR crop of the body for appearance embedding, downscaled to <= 224 px."""
    frac = REID_CROP_UPPER_FRAC if _get_reid() is not None else CLIP_CROP_UPPER_FRAC
    fh, fw = frame.shape[:2]
    px0 = max(0, min(fw - 1, int(bbox["x"] * fw)))
    px1 = max(0, min(fw, int((bbox["x"] + bbox["w"]) * fw)))
    py0 = max(0, min(fh - 1, int(bbox["y"] * fh)))
    py1 = max(0, min(fh, int((bbox["y"] + bbox["h"] * frac) * fh)))
    # Below ~8 px the crop is compression mush that embeds as noise.
    if px1 - px0 < 8 or py1 - py0 < 8:
        return None
    crop = frame[py0:py1, px0:px1]
    scale = CLIP_CROP_MAX_SIDE / max(crop.shape[:2])
    if scale < 1.0:
        return cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    # A slice is a VIEW: it keeps the whole decoded frame (6 MB at 1080p, 11 MB
    # at 1440p) alive for as long as the crop is held, and crops are held until
    # the post-loop CLIP embed. Most people in a classroom are smaller than the
    # 224 px cap and skip the resize above, so a long lesson would pin thousands
    # of full frames while the crops themselves measure a few MB. Copy the bytes.
    return crop.copy()


def _embed_tracks(
    crops: dict[int, list[tuple[int, np.ndarray]]]
) -> dict[int, list[tuple[int, list[float]]]]:
    """Timestamped L2-normalized CLIP embedding GALLERY per raw track.

    A gallery, not a single median, because a median collapses exactly the
    information re-ID needs on this footage: one crop catches her from behind,
    one against a blown-out doorway, one clean. Averaging them produces a
    vector that matches nothing well, while keeping the samples lets a matcher
    ask "did her BEST view match this track's BEST view" — the question that
    survives a person walking out of frame and back in under different light.
    Consumers that want one vector per track (the merge, DB persistence) take
    the median themselves.

    One batched post-pass over all sampled crops, so the 5 fps detection loop
    stays untouched. Preprocessing happens INSIDE the batch loop, not up front:
    a 1-hour video can accumulate tens of thousands of crops, and materializing
    every 3x224x224 CLIP tensor at once (~600 KB each) would need >10 GB and OOM
    the worker. Streaming keeps peak tensor memory at one batch (~40 MB) no
    matter how long the video is; the output is identical to a single pass.
    Failures degrade to {} instead of raising: losing re-ID evidence must never
    discard a completed multi-minute YOLO pass — the merge falls back to
    hist+spatial.
    """
    flat: list[tuple[int, int, np.ndarray]] = [
        (raw_id, ts, crop)
        for raw_id, samples in crops.items()
        for ts, crop in samples
    ]
    if not flat:
        return {}

    encoder = _get_reid()
    if encoder is not None:
        try:
            out: dict[int, list[tuple[int, list[float]]]] = {}
            for i in range(0, len(flat), CLIP_BATCH_SIZE):
                chunk = flat[i : i + CLIP_BATCH_SIZE]
                for raw_id, ts, crop in chunk:
                    h, w = crop.shape[:2]
                    # The encoder crops from a frame given a box, so hand it the
                    # crop itself as a full-frame box.
                    box = np.array([[w / 2.0, h / 2.0, w, h]], dtype=np.float32)
                    feat = np.asarray(encoder(crop, box)[0], dtype=np.float64).ravel()
                    norm = float(np.linalg.norm(feat))
                    if norm > 0:
                        out.setdefault(raw_id, []).append(
                            (ts, [float(v) for v in feat / norm])
                        )
            return out
        except Exception:
            logger.warning("re-ID embedding failed; falling back to CLIP", exc_info=True)

    try:
        import torch
        from PIL import Image

        model, preprocess, device = _get_clip()
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(flat), CLIP_BATCH_SIZE):
                chunk = flat[i : i + CLIP_BATCH_SIZE]
                tensors = [
                    # cv2 frames are BGR; CLIP's preprocess expects an RGB PIL image.
                    preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    for _raw_id, _ts, crop in chunk
                ]
                batch = torch.stack(tensors).to(device)
                feats.append(model.encode_image(batch).float().cpu().numpy())
        all_feats = np.concatenate(feats, axis=0)
    except Exception:
        logger.warning("CLIP track embedding failed; merge will run without embeds", exc_info=True)
        return {}

    out: dict[int, list[tuple[int, list[float]]]] = {}
    for (raw_id, ts, _crop), feat in zip(flat, all_feats):
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            # Normalize per sample so every later dot product is a true cosine
            # (CLIP feature norms vary with crop content).
            out.setdefault(raw_id, []).append((ts, [float(v) for v in feat / norm]))
    return out


def gallery_vectors(samples) -> list[list[float]]:
    """Plain vectors from one track's appearance evidence, whatever shape it is in.

    Three shapes are legitimately in circulation and callers that only want
    appearance (the merge, DB persistence) should not have to tell them apart:
    a single vector (what the database stores per track), a list of vectors,
    and the list of (timestamp, vector) samples /analyze produces.
    """
    items = list(samples or [])
    if not items:
        return []
    if np.isscalar(items[0]):  # one bare vector
        return [[float(v) for v in items]]
    out: list[list[float]] = []
    for s in items:
        if (
            isinstance(s, (tuple, list))
            and len(s) == 2
            and np.isscalar(s[0])
            and not np.isscalar(s[1])
        ):
            out.append([float(v) for v in s[1]])
        else:
            out.append([float(v) for v in s])
    return out


def median_embed(samples) -> Optional[list[float]]:
    """One representative unit vector for a gallery (median, re-normalized)."""
    vecs = gallery_vectors(samples)
    if not vecs:
        return None
    med = np.median(np.stack([np.asarray(v, dtype=np.float64) for v in vecs]), axis=0)
    norm = float(np.linalg.norm(med))
    if norm <= 0:
        return None
    return [float(v) for v in med / norm]


# Zero-shot age reading. Prompt pairs rather than single prompts: CLIP's
# absolute image-text cosine is dominated by the scene ("classroom"), so what
# carries signal is the DIFFERENCE between two prompts that hold the scene
# fixed and vary only the age of the person.
_ADULT_PROMPTS = (
    "a photo of an adult teacher standing in a classroom",
    "a photo of a grown woman in a classroom",
    "a photo of a grown man in a classroom",
)
_CHILD_PROMPTS = (
    "a photo of a school child sitting in a classroom",
    "a photo of a young pupil in school uniform",
    "a photo of a small boy or girl in a classroom",
)
_text_cache: Optional[tuple[np.ndarray, np.ndarray]] = None


def _adult_child_text_features():
    """(adult, child) mean unit text embeddings, cached for the process."""
    global _text_cache
    if _text_cache is None:
        import clip
        import torch

        model, _preprocess, device = _get_clip()
        with torch.no_grad():
            def encode(prompts):
                toks = clip.tokenize(list(prompts)).to(device)
                feats = model.encode_text(toks).float().cpu().numpy()
                feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
                mean = feats.mean(axis=0)
                return mean / max(float(np.linalg.norm(mean)), 1e-9)

            _text_cache = (encode(_ADULT_PROMPTS), encode(_CHILD_PROMPTS))
    return _text_cache


def zero_shot_adult(galleries: dict) -> dict[int, float]:
    """Per-track 0..1 adult-vs-child reading of the CLIP crops already embedded.

    Uses the track's BEST (most adult-leaning) views rather than its average:
    a walking adult is half-occluded in most frames, and the frames where she
    is clearly visible are the ones that carry age information. Returns {} when
    CLIP is unavailable — this is an optional signal, never a dependency.
    """
    if not galleries:
        return {}
    try:
        adult_t, child_t = _adult_child_text_features()
    except Exception:
        logger.warning("zero-shot adult scoring unavailable", exc_info=True)
        return {}

    out: dict[int, float] = {}
    for raw_id, samples in galleries.items():
        vecs = gallery_vectors(samples)
        if not vecs:
            continue
        mat = np.asarray(vecs, dtype=np.float64)
        margins = mat @ adult_t - mat @ child_t
        # Top third of views, minimum one: the cleanest evidence available.
        keep = max(1, len(margins) // 3)
        best = float(np.mean(np.sort(margins)[-keep:]))
        # CLIP prompt margins land in roughly [-0.06, 0.06]; map to 0..1.
        out[raw_id] = float(min(1.0, max(0.0, (best + 0.03) / 0.06)))
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
    hist_state: dict[int, "_SampleState"],
    crops: dict[int, list[tuple[int, np.ndarray]]],
    crop_state: dict[int, "_SampleState"],
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
    occl = _occlusions([tuple(float(v) for v in xywhn[i]) for i in range(len(ids))])

    for i, raw_id in enumerate(ids):
        cx, cy, w, h = (float(v) for v in xywhn[i])
        bbox = _clip_bbox(cx, cy, w, h)
        kxy = kpts_xy[i] if kpts_xy is not None else None
        kcf = kpts_conf[i] if kpts_conf is not None else None
        occlusion = occl[i]
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
                occlusion=occlusion,
                body=_body_ratios(kxy, kcf, frame_aspect),
            )
        )

        # Appearance sampling is OCCLUSION-GATED. A crop taken while someone
        # stands in front of this person embeds the OTHER person: in a packed
        # classroom that quietly poisons re-ID for exactly the people who move
        # (the teacher walking between rows is occluded most of the time). Only
        # a clean view earns a sample; a track that never gets one falls back to
        # its least-bad views rather than being left with no appearance at all.
        if not _appearance_sample_ok(occlusion, len(hists.get(int(raw_id), []))):
            continue

        if _due(hist_state, int(raw_id), ts_ms, occlusion):
            hist = _torso_hist(frame, bbox, kxy, kcf)
            if hist is not None:
                _offer(hists, hist_state, int(raw_id), ts_ms, occlusion, hist)

        # Crop sampling mirrors the hist cadence but keeps its own state: a
        # failed torso hist (tiny box) must not stall or accelerate crop
        # collection, and vice versa. Crops are stamped with WHEN they were
        # taken, because a raw id the tracker hands from one person to another
        # is split downstream and each half must keep only its own crops.
        if _due(crop_state, int(raw_id), ts_ms, occlusion):
            crop = _upper_crop(frame, bbox)
            if crop is not None:
                _offer(crops, crop_state, int(raw_id), ts_ms, occlusion, (ts_ms, crop))


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
) -> tuple[
    VideoMeta,
    list[Detection],
    dict[int, list[np.ndarray]],
    dict[int, list[tuple[int, list[float]]]],
]:
    """Run detection+tracking over the video.

    video_path must be an absolute path to an existing file (inside DATA_DIR
    when that env var is set); see _validate_video_path.
    Returns (video_meta, detections, torso_hist_samples_by_raw_track_id,
    clip_embed_gallery_by_raw_track_id) where each gallery is the list of
    L2-normalized CLIP ViT-B/32 vectors of that track's sampled upper-body
    crops (see _embed_tracks on why the samples are kept, not averaged).
    progress_cb receives the 0..1 fraction of sampled frames processed.
    """
    detections: list[Detection] = []
    hists: dict[int, list[np.ndarray]] = {}
    hist_state: dict[int, _SampleState] = {}
    crops: dict[int, list[tuple[int, np.ndarray]]] = {}
    crop_state: dict[int, _SampleState] = {}
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
                results, frame, ts_ms, detections, hists, hist_state, crops, crop_state
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
