"""Unit tests for board detection (app/board_detect.py + /detect-board route).

Offline and model-free: the SAM 2 / YOLO-World layer is monkeypatched; only
the pure geometry/scoring helpers and the FastAPI route wiring run for real.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import board_detect as bd
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _disable_yoloe(monkeypatch):
    """YOLOE-26-seg is the preferred strategy but needs a real model, so default
    it off (returns no masks) for every test. The existing tests then exercise
    the YOLO-World/SAM2 fallback exactly as before; the YOLOE path opts back in
    via its own monkeypatch."""
    monkeypatch.setattr(bd, "_yoloe_masks", lambda frame: [])

H, W = 720, 1280
ASSETS = Path(__file__).resolve().parent / "assets"
VIDEO = ASSETS / "classroom_synth.mp4"  # real file so path validation passes


def rect_mask(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)] = True
    return m


BOARD_MASK = rect_mask(0.20, 0.12, 0.75, 0.45)  # wide, upper — board-like
TALL_BLOB = rect_mask(0.42, 0.50, 0.55, 0.95)  # tall, lower — person-like


# --------------------------------------------------------------------------- #
# Geometric scoring
# --------------------------------------------------------------------------- #


def test_wide_upper_rectangle_beats_tall_lower_blob():
    wide = bd.score_mask(BOARD_MASK)
    tall = bd.score_mask(TALL_BLOB)
    assert wide > tall
    assert wide >= 0.5  # confidently board-like
    assert tall < bd.MIN_SCORE  # would not even produce a polygon


def test_score_empty_and_degenerate_masks_are_zero():
    assert bd.score_mask(np.zeros((H, W), dtype=bool)) == 0.0
    sliver = np.zeros((H, W), dtype=bool)
    sliver[0, :3] = True  # sub-16px blob -> degenerate
    assert bd.score_mask(sliver) == 0.0


def test_score_penalizes_wall_sized_and_full_width_regions():
    wall = rect_mask(0.0, 0.0, 1.0, 0.77)  # ~everything above the floor
    assert bd.score_mask(wall) < bd.score_mask(BOARD_MASK)
    band = rect_mask(0.0, 0.15, 1.0, 0.35)  # full-width horizontal band
    assert bd.score_mask(band) < bd.score_mask(BOARD_MASK)


def test_color_uniform_region_scores_higher_than_busy_texture():
    flat = np.full((H, W, 3), (45, 90, 35), dtype=np.uint8)  # dark green
    rng = np.random.default_rng(7)
    busy = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    assert bd.score_mask(BOARD_MASK, flat) > bd.score_mask(BOARD_MASK, busy)
    assert bd.score_mask(BOARD_MASK, flat) >= 0.5


# --------------------------------------------------------------------------- #
# Polygon simplification
# --------------------------------------------------------------------------- #


def polygon_area(poly: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def assert_polygon_sane(poly):
    assert poly is not None
    assert bd.MIN_POLYGON_POINTS <= len(poly) <= bd.MAX_POLYGON_POINTS
    for x, y in poly:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    assert polygon_area(poly) > 0.0  # closed, non-degenerate ring
    assert len({(x, y) for x, y in poly}) == len(poly)  # distinct vertices


def test_rectangle_mask_simplifies_to_four_corners():
    poly = bd.mask_to_polygon(BOARD_MASK)
    assert_polygon_sane(poly)
    assert len(poly) == 4
    # area preserved within tolerance of the source rectangle (0.55 x 0.33)
    assert abs(polygon_area(poly) - 0.55 * 0.33) < 0.02


def test_curvy_mask_simplifies_to_at_most_12_points():
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (640, 300), (350, 130), 0, 0, 360, 1, thickness=-1)
    poly = bd.mask_to_polygon(mask.astype(bool))
    assert_polygon_sane(poly)


def test_triangle_mask_falls_back_to_min_area_rect():
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([[200, 600], [1000, 600], [600, 150]])], 1)
    poly = bd.mask_to_polygon(mask.astype(bool))
    assert_polygon_sane(poly)  # 3-point simplification must be promoted to >=4


def test_empty_mask_has_no_polygon():
    assert bd.mask_to_polygon(np.zeros((H, W), dtype=bool)) is None


# --------------------------------------------------------------------------- #
# Strategy chain (model layer monkeypatched)
# --------------------------------------------------------------------------- #

FRAME = np.full((H, W, 3), 200, dtype=np.uint8)


def test_chain_reports_sam2_geometric_when_world_unavailable(monkeypatch):
    monkeypatch.setattr(bd, "_yolo_world_proposals", lambda frame: [])
    monkeypatch.setattr(bd, "_sam_segment", lambda frame, **kw: [BOARD_MASK])
    score, poly, method = bd._detect_on_frame(FRAME)
    assert method == "sam2_geometric"
    assert score >= 0.5
    assert_polygon_sane(poly)


def test_chain_prefers_yoloe_and_skips_sam2(monkeypatch):
    # YOLOE returns a board mask (class 0) and a door mask (class 5); the board
    # is used and the door class is filtered out. SAM 2 must not run at all.
    monkeypatch.setattr(bd, "_yoloe_masks", lambda frame: [(BOARD_MASK, 0.65, 0), (TALL_BLOB, 0.6, 5)])

    def _boom(*a, **k):
        raise AssertionError("SAM2 must not run when YOLOE finds the board")

    monkeypatch.setattr(bd, "_sam_segment", _boom)
    monkeypatch.setattr(bd, "_yolo_world_proposals", _boom)
    score, poly, method = bd._detect_on_frame(FRAME)
    assert method == "yoloe26_seg"
    assert score >= bd.MIN_SCORE
    assert_polygon_sane(poly)


def test_chain_prefers_yolo_world_proposal_when_it_wins(monkeypatch):
    def fake_sam(frame, *, bboxes=None, points=None):
        return [BOARD_MASK] if bboxes else [TALL_BLOB]

    monkeypatch.setattr(
        bd, "_yolo_world_proposals", lambda frame: [([256.0, 86.0, 960.0, 324.0], 0.4, 0)]
    )
    monkeypatch.setattr(bd, "_sam_segment", fake_sam)
    score, poly, method = bd._detect_on_frame(FRAME)
    assert method == "yolo_world_sam2"
    assert score >= 0.5


def test_chain_falls_back_when_world_proposal_scores_worse(monkeypatch):
    def fake_sam(frame, *, bboxes=None, points=None):
        return [TALL_BLOB] if bboxes else [BOARD_MASK]

    monkeypatch.setattr(
        bd, "_yolo_world_proposals", lambda frame: [([500.0, 400.0, 700.0, 690.0], 0.9, 0)]
    )
    monkeypatch.setattr(bd, "_sam_segment", fake_sam)
    score, poly, method = bd._detect_on_frame(FRAME)
    assert method == "sam2_geometric"


def test_detect_board_null_polygon_below_threshold(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(bd, "_sample_frames", lambda p: [(300, FRAME)])
    monkeypatch.setattr(bd, "_yolo_world_proposals", lambda frame: [])
    monkeypatch.setattr(bd, "_sam_segment", lambda frame, **kw: [TALL_BLOB])
    res = bd.detect_board("vid-1", str(VIDEO))
    assert res["polygon"] is None
    assert 0.0 <= res["confidence"] < bd.MIN_SCORE
    assert res["method"] == "sam2_geometric"
    assert res["frame_ts_ms"] == 300


def test_detect_board_picks_best_frame(monkeypatch):
    frame_a = FRAME.copy()
    frame_b = FRAME.copy()
    masks_by_frame = [(frame_a, TALL_BLOB), (frame_b, BOARD_MASK)]

    def fake_sam(frame, **kw):
        return [mask for f, mask in masks_by_frame if f is frame]

    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(bd, "_sample_frames", lambda p: [(300, frame_a), (900, frame_b)])
    monkeypatch.setattr(bd, "_yolo_world_proposals", lambda frame: [])
    monkeypatch.setattr(bd, "_sam_segment", fake_sam)

    res = bd.detect_board("vid-2", str(VIDEO))
    assert res["frame_ts_ms"] == 900
    assert res["polygon"] is not None
    assert res["confidence"] >= 0.5


def test_detect_board_survives_per_frame_failures(monkeypatch):
    def exploding_sam(frame, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(bd, "_sample_frames", lambda p: [(0, FRAME)])
    monkeypatch.setattr(bd, "_yolo_world_proposals", lambda frame: [])
    monkeypatch.setattr(bd, "_sam_segment", exploding_sam)
    res = bd.detect_board("vid-3", str(VIDEO))
    assert res["polygon"] is None
    assert res["confidence"] == 0.0
    assert res["method"] == "sam2_geometric"


def test_sample_frames_returns_three_distinct_timestamps():
    frames = bd._sample_frames(str(VIDEO))  # 6 s @ 30 fps -> idx 9, 27, 54
    assert [ts for ts, _ in frames] == [300, 900, 1800]
    for _, frame in frames:
        assert frame.shape == (720, 1280, 3)


# --------------------------------------------------------------------------- #
# /detect-board route
# --------------------------------------------------------------------------- #


def test_detect_board_route_400_on_bad_paths(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    for bad in ("relative/path.mp4", "http://evil.example/x.mp4", "/nope/missing.mp4"):
        r = client.post(
            "/detect-board", json={"video_id": "v", "video_path": bad}
        )
        assert r.status_code == 400, (bad, r.text)


def test_detect_board_route_400_outside_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    r = client.post(
        "/detect-board", json={"video_id": "v", "video_path": str(VIDEO)}
    )
    assert r.status_code == 400


def test_detect_board_route_422_on_missing_fields():
    assert client.post("/detect-board", json={"video_id": "v"}).status_code == 422


def test_detect_board_route_response_shape(monkeypatch):
    polygon = [[0.2, 0.12], [0.75, 0.12], [0.75, 0.45], [0.2, 0.45]]

    def stub(video_id, video_path):
        assert video_id == "vid-9"
        return {
            "polygon": polygon,
            "confidence": 0.87,
            "method": "sam2_geometric",
            "frame_ts_ms": 300,
        }

    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(bd, "detect_board", stub)
    r = client.post(
        "/detect-board", json={"video_id": "vid-9", "video_path": str(VIDEO)}
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"polygon", "confidence", "method", "frame_ts_ms"}
    assert body["polygon"] == polygon
    assert body["confidence"] == 0.87
    assert body["method"] == "sam2_geometric"
    assert body["frame_ts_ms"] == 300


def test_detect_board_route_null_polygon_shape(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(
        bd,
        "detect_board",
        lambda vid, path: {
            "polygon": None,
            "confidence": 0.31,
            "method": "sam2_geometric",
            "frame_ts_ms": 900,
        },
    )
    r = client.post(
        "/detect-board", json={"video_id": "v", "video_path": str(VIDEO)}
    )
    assert r.status_code == 200
    assert r.json()["polygon"] is None
    assert r.json()["confidence"] == 0.31
