"""Regression tests for detector helpers (pure functions, no GPU/model).

Covers:
- _clip_bbox interval clamping (stored bboxes must satisfy 0<=x<=x+w<=1);
- _to_detections class filtering and normalization;
- _effective_frame_count (frames actually decoded win over container metadata);
- _validate_video_path (SSRF / arbitrary-path guard in front of cv2);
- resolve_video_source's allowlist, redirect and size guards.
"""

from pathlib import Path


import numpy as np
import pytest

from app.detector import (
    _clip_bbox,
    _effective_frame_count,
    _to_detections,
    _validate_video_path,
)
from app.models import CLASS_SCREEN, CLASS_TEACHER


# --------------------------------------------------------------------------- #
# _clip_bbox — corner-format clamping
# --------------------------------------------------------------------------- #


class TestClipBbox:
    def test_inside_frame_is_unchanged(self):
        bbox = _clip_bbox(0.2, 0.3, 0.5, 0.9)
        assert bbox == {"x": 0.2, "y": 0.3, "w": 0.3, "h": 0.6}

    def test_left_overflow_clamps_both_edges(self):
        # A box starting off the left edge keeps only its visible part, and the
        # stored origin must not go negative.
        bbox = _clip_bbox(-0.1, 0.2, 0.4, 0.5)
        assert bbox["x"] == pytest.approx(0.0)
        assert bbox["w"] == pytest.approx(0.4)

    def test_vertical_overflow_is_clamped(self):
        bbox = _clip_bbox(0.5, 0.9, 0.7, 1.4)
        assert bbox["y"] == pytest.approx(0.9)
        assert bbox["h"] == pytest.approx(0.1)
        assert bbox["y"] + bbox["h"] <= 1.0 + 1e-9

    def test_fully_offscreen_box_has_zero_extent(self):
        bbox = _clip_bbox(1.2, 1.3, 1.5, 1.6)
        assert bbox["w"] == 0.0
        assert bbox["h"] == 0.0


# --------------------------------------------------------------------------- #
# _to_detections
# --------------------------------------------------------------------------- #


class _FakeDetections:
    """Stand-in for supervision.Detections: the three arrays we read, plus len."""

    def __init__(self, class_ids, confs, boxes):
        self.class_id = np.array(class_ids)
        self.confidence = np.array(confs)
        self.xyxy = np.array(boxes, dtype=float)

    def __len__(self):
        return len(self.class_id)


def _fake_det(class_ids, confs, boxes):
    return _FakeDetections(class_ids, confs, boxes)


class TestToDetections:
    def test_pixels_are_normalized_by_frame_size(self):
        det = _fake_det([CLASS_TEACHER], [0.9], [[256, 144, 512, 720]])
        out = _to_detections(det, ts_ms=1000, width=2560, height=1440)
        assert len(out) == 1
        d = out[0]
        assert d.cls == CLASS_TEACHER
        assert d.video_ts_ms == 1000
        assert d.conf == pytest.approx(0.9)
        assert d.bbox["x"] == pytest.approx(0.1)
        assert d.bbox["y"] == pytest.approx(0.1)
        assert d.bbox["w"] == pytest.approx(0.1)
        assert d.bbox["h"] == pytest.approx(0.4)

    def test_unknown_class_ids_are_dropped(self):
        """A checkpoint emitting a class this build does not map must not
        produce a detection with a meaningless class id."""
        det = _fake_det([CLASS_SCREEN, 99], [0.8, 0.8], [[0, 0, 10, 10], [0, 0, 10, 10]])
        out = _to_detections(det, ts_ms=0, width=100, height=100)
        assert [d.cls for d in out] == [CLASS_SCREEN]

    def test_empty_frame_yields_nothing(self):
        assert _to_detections(None, 0, 100, 100) == []


# --------------------------------------------------------------------------- #
# _effective_frame_count
# --------------------------------------------------------------------------- #


class TestEffectiveFrameCount:
    def test_truncated_file_ignores_inflated_metadata(self):
        # moov atom claims 9000 frames, decode stopped at 292.
        assert _effective_frame_count(9000, 292) == 292

    def test_missing_metadata_uses_frames_read(self):
        assert _effective_frame_count(0, 300) == 300
        assert _effective_frame_count(-1, 300) == 300

    def test_zero_frames_read_falls_back_to_metadata(self):
        assert _effective_frame_count(9000, 0) == 9000

    def test_all_zero_stays_zero(self):
        assert _effective_frame_count(0, 0) == 0


# --------------------------------------------------------------------------- #
# _validate_video_path
# --------------------------------------------------------------------------- #


class TestValidateVideoPath:
    def test_rejects_http_url(self):
        with pytest.raises(ValueError):
            _validate_video_path("http://169.254.169.254/latest/meta-data/")

    def test_rejects_rtsp_url(self):
        with pytest.raises(ValueError):
            _validate_video_path("rtsp://internal-host:554/stream")

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ValueError):
            _validate_video_path(str(tmp_path / "nope.mp4"))

    def test_rejects_directory(self, tmp_path):
        with pytest.raises(ValueError):
            _validate_video_path(str(tmp_path))

    def test_accepts_existing_file_when_data_dir_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATA_DIR", raising=False)
        f = tmp_path / "original.mp4"
        f.write_bytes(b"x")
        assert _validate_video_path(str(f)) == str(f.resolve())

    def test_enforces_data_dir_containment(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        (data_dir / "videos" / "abc").mkdir(parents=True)
        inside = data_dir / "videos" / "abc" / "original.mp4"
        inside.write_bytes(b"x")
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"x")
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        assert _validate_video_path(str(inside)) == str(inside.resolve())
        with pytest.raises(ValueError):
            _validate_video_path(str(outside))

    def test_rejects_traversal_out_of_data_dir(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        secret = tmp_path / "secret.mp4"
        secret.write_bytes(b"x")
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        with pytest.raises(ValueError):
            _validate_video_path(str(data_dir / ".." / "secret.mp4"))


# --------------------------------------------------------------------------- #
# resolve_video_source: object-store URL fetch (allowlist-gated) vs local path
# --------------------------------------------------------------------------- #


def test_media_url_host_allowed(monkeypatch):
    from app import detector as D
    from app.config import get_settings

    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "")
    get_settings.cache_clear()
    assert D._media_url_host_allowed("http://localhost:9000/b/k.mp4") is False
    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "minio:9000,localhost:9000")
    get_settings.cache_clear()
    assert D._media_url_host_allowed("http://localhost:9000/b/k.mp4") is True
    assert D._media_url_host_allowed("http://evil:9000/b/k.mp4") is False
    get_settings.cache_clear()


def test_resolve_url_rejected_without_allowlist(monkeypatch):
    from app import detector as D
    from app.config import get_settings

    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="media URL host"):
        D.resolve_video_source("https://cdn.example.com/x.mp4")
    get_settings.cache_clear()


def test_resolve_url_downloads_when_allowlisted(monkeypatch):
    import io

    from app import detector as D
    from app.config import get_settings

    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "host:1234")
    monkeypatch.delenv("DATA_DIR", raising=False)
    get_settings.cache_clear()
    payload = b"FAKE-VIDEO-BYTES" * 1000
    # The fetch now goes through the allowlist-revalidating opener, not the bare
    # urlopen, so patch the opener's open().
    monkeypatch.setattr(D._MEDIA_URL_OPENER, "open", lambda url, timeout=30: io.BytesIO(payload))
    path, is_temp = D.resolve_video_source("http://host:1234/bucket/key.mp4")
    try:
        assert is_temp is True
        assert Path(path).read_bytes() == payload
    finally:
        Path(path).unlink(missing_ok=True)
    get_settings.cache_clear()


def test_download_rejects_redirect_to_non_allowlisted_host(monkeypatch):
    """A 3xx from an allowlisted origin to an internal host is refused (the
    SSRF-via-redirect bypass): the redirect handler re-checks every hop."""
    import io
    import urllib.request

    from app import detector as D
    from app.config import get_settings

    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "cache.example.com")
    get_settings.cache_clear()
    handler = D._AllowlistRedirectHandler()
    with pytest.raises(ValueError, match="non-allowlisted host"):
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/"
        )
    # A redirect back to an allowlisted host is allowed to proceed.
    req = handler.redirect_request(
        urllib.request.Request("http://cache.example.com/a.mp4"),
        io.BytesIO(b""),
        302,
        "Found",
        {},
        "http://cache.example.com/b.mp4",
    )
    assert req is not None
    get_settings.cache_clear()


def test_download_rejects_oversize_object(monkeypatch):
    """An allowlisted object larger than the byte cap aborts (disk-exhaustion
    guard) and leaves no temp file behind."""
    import io

    from app import detector as D
    from app.config import get_settings

    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "host:1234")
    monkeypatch.delenv("DATA_DIR", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(D, "_MEDIA_MAX_DOWNLOAD_BYTES", 4096)
    big = io.BytesIO(b"x" * (4096 * 4))
    monkeypatch.setattr(D._MEDIA_URL_OPENER, "open", lambda url, timeout=30: big)
    with pytest.raises(ValueError, match="byte cap"):
        D.resolve_video_source("http://host:1234/bucket/huge.mp4")
    get_settings.cache_clear()


def test_resolve_local_path_passthrough(monkeypatch):
    from app import detector as D

    monkeypatch.delenv("DATA_DIR", raising=False)
    asset = str((Path(__file__).resolve().parent / "assets" / "synthetic.mp4"))
    path, is_temp = D.resolve_video_source(asset)
    assert is_temp is False
    assert path == str(Path(asset).resolve())


# --------------------------------------------------------------------------- #
# _optimize: precision and trace batch size
# --------------------------------------------------------------------------- #


class TestOptimize:
    """The JIT trace must be built for the precision and batch it will run.

    Both rfdetr defaults are wrong for a GPU deployment (fp32, batch 1), and
    getting this wrong is invisible: the run is correct, just half the speed
    and twice the VRAM. That is exactly how the previous pipeline's "~5x
    TensorRT speedup" turned out to be an fp16 bug.
    """

    def _capture(self, monkeypatch, device, batch=8):
        import torch

        from app import detector as D
        from app.config import get_settings

        monkeypatch.setenv("RFDETR_BATCH", str(batch))
        get_settings.cache_clear()
        seen = {}

        class _M:
            def inference(self, compile, batch_size, dtype):
                seen.update(compile=compile, batch_size=batch_size, dtype=dtype)

        D._optimize(_M(), device)
        get_settings.cache_clear()
        return seen, torch

    def test_cuda_traces_fp16_at_the_real_batch_size(self, monkeypatch):
        seen, torch = self._capture(monkeypatch, "cuda", batch=16)
        assert seen["dtype"] is torch.float16
        assert seen["batch_size"] == 16
        assert seen["compile"] is True

    def test_specific_cuda_device_still_gets_fp16(self, monkeypatch):
        seen, torch = self._capture(monkeypatch, "cuda:1")
        assert seen["dtype"] is torch.float16

    def test_mps_stays_fp32_batch_one(self, monkeypatch):
        """Half precision on MPS is uneven and batching buys nothing off-GPU."""
        seen, torch = self._capture(monkeypatch, "mps", batch=16)
        assert seen["dtype"] is torch.float32
        assert seen["batch_size"] == 1

    def test_cpu_stays_fp32_batch_one(self, monkeypatch):
        seen, torch = self._capture(monkeypatch, "cpu", batch=16)
        assert seen["dtype"] is torch.float32
        assert seen["batch_size"] == 1

    def test_a_tracing_failure_degrades_instead_of_raising(self, monkeypatch):
        """Optimization is throughput, not correctness: losing it must not sink
        a multi-minute analysis."""
        from app import detector as D

        class _Boom:
            def inference(self, **kw):
                raise RuntimeError("no compiler")

        D._optimize(_Boom(), "cuda")  # must not raise


class TestRecordPrecision:
    """The read-back must report what the MODEL carries, not what we asked for.

    This is the guard the previous pipeline had and this rewrite briefly lost.
    Its whole point is that a check built on the request shows a green light on
    exactly the broken configuration it exists to catch — so these tests drive
    it with a module whose dtype disagrees with the request.
    """

    def _model_with(self, dtype):
        import torch

        class _Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 2).to(dtype)

        class _Ctx:
            inference_model = _Mod()

        class _M:
            model = _Ctx()

        return _M()

    def test_reports_error_when_the_model_is_not_the_dtype_we_asked_for(self, caplog):
        import torch

        from app import detector as D

        m = self._model_with(torch.float32)
        with caplog.at_level("ERROR"):
            D._record_precision(m, torch.float16)
        assert any("DEGRADED" in r.message for r in caplog.records)

    def test_silent_when_the_model_matches(self, caplog):
        import torch

        from app import detector as D

        m = self._model_with(torch.float32)
        with caplog.at_level("ERROR"):
            D._record_precision(m, torch.float32)
        assert not [r for r in caplog.records if "DEGRADED" in r.message]

    def test_unreadable_model_does_not_raise(self):
        """A diagnostic must never be the thing that fails an analysis."""
        import torch

        from app import detector as D

        class _Broken:
            model = None

        D._record_precision(_Broken(), torch.float16)  # must not raise


class TestRaggedLastBatch:
    """A traced model accepts EXACTLY its compiled batch; videos end ragged.

    150 frames at batch 16 leaves a remainder of 6, and rfdetr raises on the
    mismatch rather than adapting — so every video died on its final flush.
    This cannot reproduce off-GPU (the trace is built at batch 1 there, so every
    batch is full), which is why it took a real pod to surface.
    """

    def _model(self, seen):
        class _M:
            def predict(self, images, threshold):
                seen.append(len(images))
                return [f"det{i}" for i in range(len(images))]

        return _M()

    def test_short_batch_is_padded_and_results_trimmed(self, monkeypatch):
        from app import detector as D

        monkeypatch.setattr(D, "_traced_batch", 16)
        monkeypatch.setattr(D, "_trt", None)
        seen: list[int] = []
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(6)]
        out = D._predict_batch(self._model(seen), frames)
        assert seen == [16], "short batch must be padded up to the traced size"
        assert len(out) == 6, "padding must not leak into the results"

    def test_full_batch_is_untouched(self, monkeypatch):
        from app import detector as D

        monkeypatch.setattr(D, "_traced_batch", 16)
        monkeypatch.setattr(D, "_trt", None)
        seen: list[int] = []
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(16)]
        out = D._predict_batch(self._model(seen), frames)
        assert seen == [16] and len(out) == 16

    def test_eager_model_is_never_padded(self, monkeypatch):
        """Off-GPU there is no trace, so a short batch must pass through as-is."""
        from app import detector as D

        monkeypatch.setattr(D, "_traced_batch", None)
        monkeypatch.setattr(D, "_trt", None)
        seen: list[int] = []
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        out = D._predict_batch(self._model(seen), frames)
        assert seen == [3] and len(out) == 3


# --------------------------------------------------------------------------- #
# DATA_DIR: one definition, shared by the downloader and the guard
# --------------------------------------------------------------------------- #
#
# Regression for a defect that killed a paid GPU run six minutes in. The pod
# env sets DATA_DIR=/workspace/data, the image never creates it, and the two
# functions that read the variable disagreed about what a missing directory
# means: the downloader called it "unconfigured" and used /tmp, the guard called
# it "configured" and rejected /tmp. The failure surfaced as
# "video_path must be inside DATA_DIR" — the symptom at the far end of the
# pipeline, naming nothing about the missing mkdir at the near end.


def test_data_dir_is_created_when_it_does_not_exist(tmp_path, monkeypatch):
    from app import detector as D

    missing = tmp_path / "workspace" / "data"
    monkeypatch.setenv("DATA_DIR", str(missing))
    assert not missing.exists()

    base = D._data_dir()
    assert base == missing.resolve()
    assert missing.is_dir(), "a configured DATA_DIR must be created, not ignored"


def test_data_dir_unset_is_none(monkeypatch):
    from app import detector as D

    monkeypatch.delenv("DATA_DIR", raising=False)
    assert D._data_dir() is None


def test_download_into_a_missing_data_dir_survives_the_guard(tmp_path, monkeypatch):
    """THE REGRESSION. A fresh pod, verbatim: DATA_DIR configured, directory
    absent, video fetched from an allowlisted object store.

    Before the fix this downloaded to /tmp and then _validate_video_path
    refused it. The two calls are asserted together because either one alone
    passes on the broken code — it is their DISAGREEMENT that was the bug.
    """
    import io

    from app import detector as D
    from app.config import get_settings

    missing = tmp_path / "workspace" / "data"
    monkeypatch.setenv("DATA_DIR", str(missing))
    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "host:1234")
    get_settings.cache_clear()
    payload = b"FAKE-VIDEO-BYTES" * 1000
    monkeypatch.setattr(D._MEDIA_URL_OPENER, "open", lambda url, timeout=30: io.BytesIO(payload))

    path, is_temp = D.resolve_video_source("http://host:1234/bucket/key.mp4")
    try:
        assert is_temp is True
        assert Path(path).read_bytes() == payload
        # Landed inside DATA_DIR, not /tmp...
        assert Path(path).resolve().is_relative_to(missing.resolve())
        # ...and therefore survives the re-validation the pipeline does next.
        assert D._validate_video_path(path) == str(Path(path).resolve())
    finally:
        Path(path).unlink(missing_ok=True)
    get_settings.cache_clear()


def test_unusable_data_dir_fails_where_the_cause_is(tmp_path, monkeypatch):
    """When DATA_DIR cannot be created, say so at download time rather than
    letting it surface later as a containment rejection naming the wrong thing.
    """
    import io

    from app import detector as D
    from app.config import get_settings

    # A file where the directory should be: mkdir cannot succeed.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    monkeypatch.setenv("DATA_DIR", str(blocked))
    monkeypatch.setenv("MEDIA_URL_ALLOWLIST", "host:1234")
    get_settings.cache_clear()
    monkeypatch.setattr(D._MEDIA_URL_OPENER, "open", lambda url, timeout=30: io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="does not exist and could not be created"):
        D.resolve_video_source("http://host:1234/bucket/key.mp4")
    get_settings.cache_clear()
