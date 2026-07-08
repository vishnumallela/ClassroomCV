"""Regression tests for detector helpers (pure functions, no GPU/model).

Covers:
- _is_standing frame-aspect correction (normalized h/w must be converted back
  to pixel aspect before the 1.6 threshold);
- _clip_bbox interval clamping (stored bboxes must satisfy 0<=x<=x+w<=1);
- _effective_frame_count (frames actually decoded win over container metadata);
- _validate_video_path (SSRF / arbitrary-path guard in front of cv2).
"""

import numpy as np
import pytest

from app.detector import (
    L_HIP,
    L_KNEE,
    R_HIP,
    R_KNEE,
    _clip_bbox,
    _effective_frame_count,
    _is_standing,
    _validate_video_path,
)


# --------------------------------------------------------------------------- #
# _is_standing
# --------------------------------------------------------------------------- #


class TestIsStandingFrameAspect:
    def test_seated_person_on_16_9_frame_is_not_standing(self):
        # 120x140 px bbox on 1920x1080 (pixel aspect 1.167, clearly seated).
        # Normalized h/w = 2.07 used to trip the 1.6 threshold unconditionally.
        w, h = 120 / 1920, 140 / 1080
        assert _is_standing(w, h, None, None, frame_aspect=1920 / 1080) is False

    def test_square_bbox_on_16_9_frame_is_not_standing(self):
        # A perfectly square pixel bbox used to count as standing on 16:9.
        w, h = 150 / 1920, 150 / 1080
        assert _is_standing(w, h, None, None, frame_aspect=1920 / 1080) is False

    def test_standing_person_on_16_9_frame_is_standing(self):
        # 100x200 px bbox: pixel aspect 2.0 > 1.6.
        w, h = 100 / 1920, 200 / 1080
        assert _is_standing(w, h, None, None, frame_aspect=1920 / 1080) is True

    def test_standing_person_on_portrait_frame_is_standing(self):
        # 200x560 px on 1080x1920 (pixel aspect 2.8): normalized h/w = 1.575
        # used to fall below the threshold and miss true standing.
        w, h = 200 / 1080, 560 / 1920
        assert _is_standing(w, h, None, None, frame_aspect=1080 / 1920) is True

    def test_default_frame_aspect_is_neutral(self):
        # Square frames (aspect 1.0) behave exactly as before the fix.
        assert _is_standing(0.1, 0.2, None, None) is True
        assert _is_standing(0.2, 0.2, None, None) is False

    def test_keypoint_fallback_still_consulted_when_aspect_fails(self):
        # Seated-shaped bbox but knees well below hips -> standing via
        # keypoint geometry (branch order regression guard).
        h = 0.4
        kxy = np.zeros((17, 2), dtype=np.float32)
        kconf = np.zeros(17, dtype=np.float32)
        for i in (L_HIP, R_HIP):
            kxy[i] = [0.5, 0.5]
            kconf[i] = 0.9
        for i in (L_KNEE, R_KNEE):
            kxy[i] = [0.5, 0.5 + 0.3 * h]
            kconf[i] = 0.9
        assert _is_standing(0.3, h, kxy, kconf, frame_aspect=16 / 9) is True


# --------------------------------------------------------------------------- #
# _clip_bbox
# --------------------------------------------------------------------------- #


class TestClipBbox:
    def test_box_inside_frame_is_unchanged(self):
        bbox = _clip_bbox(0.5, 0.5, 0.2, 0.4)
        assert bbox == {"x": 0.4, "y": 0.3, "w": 0.2, "h": 0.4}

    def test_right_edge_overflow_is_clamped_to_frame(self):
        # cx=0.95, w=0.20 -> raw box spans 0.85..1.05; stored bbox used to
        # keep w=0.20 so x+w=1.05, violating the SPEC 0-1 contract.
        bbox = _clip_bbox(0.95, 0.5, 0.20, 0.4)
        assert bbox["x"] == pytest.approx(0.85)
        assert bbox["w"] == pytest.approx(0.15)
        assert bbox["x"] + bbox["w"] <= 1.0 + 1e-9

    def test_left_edge_overflow_keeps_center_of_visible_region(self):
        # cx - w/2 = -0.08: clamping x alone used to shift the center right
        # by 0.04 while keeping the full width.
        bbox = _clip_bbox(0.02, 0.5, 0.2, 0.4)
        assert bbox["x"] == 0.0
        assert bbox["w"] == pytest.approx(0.12)
        # Center of the stored box is the center of the visible region.
        assert bbox["x"] + bbox["w"] / 2 == pytest.approx(0.06)

    def test_vertical_overflow_is_clamped(self):
        bbox = _clip_bbox(0.5, 0.98, 0.2, 0.3)
        assert bbox["y"] == pytest.approx(0.83)
        assert bbox["h"] == pytest.approx(0.17)
        assert bbox["y"] + bbox["h"] <= 1.0 + 1e-9


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
# _embed_tracks streams crops in batches (bounded memory for long videos)
# --------------------------------------------------------------------------- #


def test_embed_tracks_streams_and_normalizes(monkeypatch):
    """CLIP embedding must batch INSIDE the loop (not materialize every tensor
    up front) so a 1-hour video's tens of thousands of crops cannot OOM, and it
    must return a unit-norm median vector per raw track."""
    import numpy as np
    import torch

    from app import detector as D

    # Force multiple batches: 5 crops across 2 tracks with batch size 2.
    monkeypatch.setattr(D, "CLIP_BATCH_SIZE", 2)

    seen_batch_sizes: list[int] = []

    class FakeModel:
        def encode_image(self, batch):
            seen_batch_sizes.append(int(batch.shape[0]))
            # flatten each fake CxHxW crop tensor into a feature vector
            return batch.reshape(batch.shape[0], -1)

    def fake_preprocess(img):
        m = float(np.asarray(img).mean())
        return torch.full((3, 2, 2), m)  # a fake 3x2x2 "image" tensor

    monkeypatch.setattr(D, "_get_clip", lambda: (FakeModel(), fake_preprocess, "cpu"))

    crops = {
        7: [np.full((8, 8, 3), v, np.uint8) for v in (10, 20, 30)],
        9: [np.full((8, 8, 3), v, np.uint8) for v in (40, 50)],
    }
    out = D._embed_tracks(crops)

    assert set(out.keys()) == {7, 9}
    for vec in out.values():
        assert len(vec) == 12  # 3*2*2 flattened
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5  # unit-normalized
    # 5 crops at batch size 2 -> batches of [2, 2, 1]; never all 5 at once.
    assert seen_batch_sizes == [2, 2, 1]
    assert max(seen_batch_sizes) <= 2
