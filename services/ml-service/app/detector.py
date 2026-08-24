"""RF-DETR detection over sampled video frames — the pipeline's only detector.

Reads the video with cv2.VideoCapture, samples frames at
stride = max(1, round(native_fps / sample_fps)), and runs one fine-tuned
RF-DETR over each batch. The model was trained on this product's own five
classes:

    0 door   1 screen   2 teacher   3 pointing   4 writing

which is the whole reason this module is a tenth of its former size. The
pipeline it replaced ran a general person detector, then had to work out WHICH
of thirty detected bodies was the teacher — by pose, stature, body
proportions, clothing colour, CLIP appearance, a tracklet DP and a vision-model
vote. The detector now answers that question directly, so all of it is gone.

iter_frames is the frame-source seam (a future KafkaSource only has to
reproduce the same (ts_ms, frame) contract); detect_video consumes it.

Per kept frame, per detection we emit a Detection with:
- cls: one of the five class ids above
- bbox {x, y, w, h} normalized, top-left based
- conf: the model's score

GPU serving: device 'auto' resolves cuda > mps > cpu, and REQUIRE_DEVICE=cuda
makes a mis-provisioned pod fail loud instead of silently billing ~20x the
wall-clock on CPU. Frames are batched (rfdetr_batch) because RF-DETR's predict
takes a list, which is the main throughput lever on a GPU.

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
from app.models import (
    CLASS_DOOR,
    CLASS_NAMES,
    CLASS_POINTING,
    CLASS_SCREEN,
    CLASS_TEACHER,
    CLASS_WRITING,
    Detection,
    VideoMeta,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_DOOR",
    "CLASS_NAMES",
    "CLASS_POINTING",
    "CLASS_SCREEN",
    "CLASS_TEACHER",
    "CLASS_WRITING",
    "detect_video",
    "get_device",
    "iter_frames",
    "model_loaded",
    "resolve_model_name",
    "resolve_video_source",
]

MAX_SAMPLE_FPS = 5.0
FALLBACK_NATIVE_FPS = 30.0

# What the training run recorded, in label order. Checked at load so a
# retrained checkpoint with reordered classes fails loudly instead of quietly
# reporting the door as the teacher — the class ids in app/models.py are the
# entire contract between the model and every KPI this product ships.
EXPECTED_CLASS_NAMES = ["Door", "Screen", "Teacher", "pointing", "writing"]

# Floor for what leaves the detector at all. Per-class thresholds
# (settings.teacher_conf / zone_conf / action_conf) are applied downstream, so
# one detection pass can serve every consumer without re-running the model.
DETECT_FLOOR = 0.15

_model = None
# The TensorRT backend, when one loaded. None = serving PyTorch.
_trt = None


def model_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    """Effective inference device: 'cuda', 'mps', or 'cpu'.

    Resolves Settings.device. 'auto' prefers cuda, then mps, then cpu; an
    explicit device is honoured but degrades to cpu when the hardware is
    absent, so a misconfigured dev box does not crash. Production sets
    REQUIRE_DEVICE to turn that degradation back into a loud failure.
    """
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
    """Path of the RF-DETR checkpoint to serve (Settings.rfdetr_weights)."""
    return (get_settings().rfdetr_weights or "").strip()


def _check_class_order(weights: str) -> None:
    """Verify the checkpoint's own class order matches what this module assumes.

    The class ids are the entire contract between the model and the product:
    every KPI hangs off 'which id is the teacher'. The order is recorded in the
    checkpoint's training args, so it is cheap to assert and expensive to get
    wrong silently.
    """
    try:
        import torch

        ck = torch.load(weights, map_location="cpu", weights_only=False)
        args = ck.get("args")
        names = list((vars(args) if not isinstance(args, dict) else args).get("class_names") or [])
    except Exception:
        logger.warning("could not read class_names from %s; assuming the default order", weights)
        return
    if names and names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            f"checkpoint {weights} declares classes {names}, but this service maps "
            f"{EXPECTED_CLASS_NAMES}. Fix app/detector.py's CLASS_* ids before serving it."
        )


def _get_model():
    global _model
    if _model is None:
        weights = resolve_model_name()
        if not weights:
            raise RuntimeError(
                "RFDETR_WEIGHTS is not set: the ML service has no detector to run"
            )
        if not Path(weights).is_file():
            raise RuntimeError(f"RF-DETR checkpoint not found: {weights}")
        device = get_device()
        _assert_required_device(device)
        _check_class_order(weights)

        from rfdetr import RFDETRMedium

        settings = get_settings()
        logger.info("loading RF-DETR %s on device %s", weights, device)
        _model = RFDETRMedium(
            pretrain_weights=weights,
            num_classes=len(CLASS_NAMES),
            resolution=settings.rfdetr_resolution,
            device=device,
        )
        # TensorRT FIRST, and only optimize the PyTorch model if it does not
        # take over. Order matters both ways: export() wants the clean module
        # rather than a JIT-traced one, and tracing a model that is about to be
        # replaced by an engine is pure startup cost.
        global _trt
        if settings.rfdetr_tensorrt:
            from app import tensorrt_backend

            _trt = tensorrt_backend.try_load(
                _model,
                weights,
                device,
                settings.rfdetr_resolution,
                max(1, int(settings.rfdetr_batch)),
            )
        if _trt is None:
            _optimize(_model, device)
        logger.info("RF-DETR serving backend: %s", serving_backend())
    return _model


def _record_precision(model, requested) -> None:
    """Read the precision the model ACTUALLY carries back off its weights.

    Restored, deliberately, from the pipeline this replaced. Its guard existed
    because asking for fp16 and getting fp16 are different things, and the
    request is what every convenient accessor reports back — so a check built
    on the request shows a green light on precisely the broken configuration it
    was meant to catch. The parameter dtype is the ground truth.

    That is not a hypothetical here. The bug this file just fixed (rfdetr's
    `inference()` defaulting to float32) is invisible from the outside: the run
    is correct, merely half-speed at double the VRAM, and the unit tests around
    _optimize can only prove which arguments were PASSED. This is the check
    that proves what was loaded, and it is the one that will fire on the pod if
    a future rfdetr silently ignores the dtype.

    Best-effort: a diagnostic must never be the thing that fails an analysis.
    """
    import torch

    try:
        ctx = getattr(model, "model", None)
        module = getattr(ctx, "inference_model", None) or getattr(ctx, "model", None)
        actual = next(module.parameters()).dtype if module is not None else None
    except Exception:
        logger.warning("could not read back inference precision", exc_info=True)
        return
    if actual is None:
        return
    logger.info("inference precision (read back from weights): %s", actual)
    if actual is not requested:
        # Not fatal: the results are right, the throughput is not. Loud because
        # it is otherwise completely invisible — the previous pipeline shipped
        # this way and nothing in the product surfaced it.
        logger.error(
            "DEGRADED: asked for %s but the model loaded %s; inference will run "
            "at roughly half speed and twice the VRAM",
            requested,
            actual,
        )


def _optimize(model, device: str) -> None:
    """JIT-trace the model at the precision and batch size it will actually run.

    Both arguments are load-bearing and both default to the WRONG thing for a
    GPU deployment:

    dtype defaults to float32. On an L4 that is about half the throughput and
    twice the VRAM for identical outputs. This project has already paid for
    this exact mistake once: the previous pipeline's celebrated "~5x TensorRT
    speedup" turned out, when measured, to be 1.05-1.25x — the real win was a
    warmup call that had been silently pinning the backend to fp32.

    batch_size defaults to 1, and it is what the trace is specialised for. We
    feed batches of rfdetr_batch, so tracing at 1 either forces a retrace or
    runs a graph built for the wrong shape — the throughput lever cancelling
    itself out.

    fp16 is CUDA-only here: it is not faster on cpu, and MPS support for
    half-precision transformer ops is uneven enough that dev boxes are better
    off in fp32.
    """
    import torch

    settings = get_settings()
    on_cuda = device.split(":", 1)[0] == "cuda"
    dtype = torch.float16 if on_cuda else torch.float32
    batch = max(1, int(settings.rfdetr_batch)) if on_cuda else 1
    try:
        model.inference(compile=True, batch_size=batch, dtype=dtype)
        logger.info(
            "RF-DETR traced for inference: dtype=%s batch=%d device=%s", dtype, batch, device
        )
        _record_precision(model, dtype)
    except Exception:
        # Purely a throughput step; a failure here costs speed, not
        # correctness, and must never sink a multi-minute analysis.
        logger.warning(
            "RF-DETR inference optimization unavailable; serving the eager model",
            exc_info=True,
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


def _clip_bbox(x0: float, y0: float, x1: float, y1: float) -> dict:
    """Clamp a normalized corner-format box to the frame as an interval.

    Clamping x/y alone would shift the stored origin and can leave x+w > 1;
    clamp both edges instead so 0 <= x <= x+w <= 1 (same for y).
    """
    ax0 = max(0.0, min(1.0, x0))
    ax1 = max(0.0, min(1.0, x1))
    ay0 = max(0.0, min(1.0, y0))
    ay1 = max(0.0, min(1.0, y1))
    return {
        "x": round(ax0, 5),
        "y": round(ay0, 5),
        "w": round(max(0.0, ax1 - ax0), 5),
        "h": round(max(0.0, ay1 - ay0), 5),
    }


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
    """File-backed frame source (the Kafka seam).

    Validates video_path via _validate_video_path, then yields
    (video_ts_ms, BGR frame) for every stride-th decodable frame, with
    stride = max(1, round(native_fps / effective_sample_fps)) and
    ts_ms = round(frame_idx / native_fps * 1000). Any future source
    (Kafka/RTSP) only has to reproduce this (ts_ms, frame) contract. The
    capture is released when the generator is exhausted, closed, or unwound by
    an exception.
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


def serving_backend() -> str:
    """Which backend is actually serving: 'tensorrt' or 'pytorch'.

    Read back rather than inferred from config, because "TensorRT is enabled"
    and "TensorRT is running" are different claims and only the second one is
    worth anything. /health reports this.
    """
    return "tensorrt" if _trt is not None else "pytorch"


def _predict_batch(model, frames: list[np.ndarray]) -> list:
    """RF-DETR over a batch of BGR frames; returns one Detections per frame.

    predict() takes a list, so one call covers the whole batch — the main
    throughput lever on a GPU. cv2 decodes BGR and the model expects RGB.

    A failure here propagates. The pipeline this replaced caught it and retried
    on CPU, which was worth doing when the alternative was losing a completed
    multi-minute pass; here it would mean finishing the job at roughly twenty
    times the wall-clock while reporting success, which is the outcome
    REQUIRE_DEVICE exists to prevent. A device that cannot run the model is a
    misconfiguration, and it should say so on the first batch.
    """
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    if _trt is not None:
        return _trt.predict(rgb, threshold=DETECT_FLOOR)
    out = model.predict(rgb, threshold=DETECT_FLOOR)
    return out if isinstance(out, list) else [out]


def _to_detections(det, ts_ms: int, width: int, height: int) -> list[Detection]:
    """One frame's supervision Detections -> our normalized Detection rows."""
    out: list[Detection] = []
    if det is None or len(det) == 0:
        return out
    for cid, conf, box in zip(det.class_id, det.confidence, det.xyxy):
        cls = int(cid)
        if cls not in CLASS_NAMES:
            continue  # a class this build does not know about
        x0, y0, x1, y1 = (float(v) for v in box)
        out.append(
            Detection(
                video_ts_ms=ts_ms,
                cls=cls,
                bbox=_clip_bbox(x0 / width, y0 / height, x1 / width, y1 / height),
                conf=float(conf),
            )
        )
    return out


def detect_video(
    video_path: str,
    sample_fps: float = 5.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> tuple[VideoMeta, list[Detection]]:
    """Run RF-DETR over the sampled frames of a video.

    video_path must be an absolute path to an existing file (inside DATA_DIR
    when that env var is set); see _validate_video_path. Returns
    (video_meta, detections) with every detection above DETECT_FLOOR, of every
    class — per-class thresholds are applied downstream so one pass serves
    every consumer. progress_cb receives the 0..1 fraction of frames processed.
    """
    detections: list[Detection] = []
    last_ts_ms = 0

    model = _get_model()
    settings = get_settings()
    batch_size = max(1, int(settings.rfdetr_batch))
    if get_device().split(":", 1)[0] != "cuda":
        batch_size = 1  # batching only pays on a GPU

    info = FrameSourceInfo()
    frames = iter_frames(video_path, sample_fps, info=info)
    processed = 0
    pending: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        nonlocal processed
        if not pending:
            return
        results = _predict_batch(model, [f for _ts, f in pending])
        for (ts, _f), det in zip(pending, results):
            detections.extend(_to_detections(det, ts, info.width, info.height))
        processed += len(pending)
        pending.clear()
        if progress_cb and info.frames_to_process:
            progress_cb(min(1.0, processed / info.frames_to_process))

    try:
        for ts_ms, frame in frames:
            last_ts_ms = ts_ms
            pending.append((ts_ms, frame))
            if len(pending) >= batch_size:
                flush()
        flush()
    finally:
        frames.close()

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
    logger.info(
        "detected %d boxes over %d frames of %s",
        len(detections),
        processed,
        Path(video_path).name,
    )
    return meta, detections
