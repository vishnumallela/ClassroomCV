"""API-level tests via TestClient (no GPU/DB: DB access is monkeypatched)."""

from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.models import AnalysisResult, Detection

client = TestClient(app)


def test_health_shape():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] in ("mps", "cpu")
    assert isinstance(body["model_loaded"], bool)


def test_unknown_job_404():
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/result").status_code == 404


def test_analyze_validation_error():
    r = client.post("/analyze", json={"video_id": "x"})  # missing video_path
    assert r.status_code == 422


def test_zone_polygon_rejects_malformed_points():
    """Regression: a point that is not exactly [x, y] must 422 at validation
    instead of crashing derive with an IndexError 500 (only the OUTER list
    length used to be constrained)."""
    for bad_polygon in (
        [[0.5], [0.1, 0.1], [0.2, 0.2]],  # 1-element point
        [[0.1, 0.1, 0.1], [0.1, 0.1], [0.2, 0.2]],  # 3-element point
    ):
        for endpoint, extra in (
            ("/rederive", {}),
            ("/analyze", {"video_path": "/tmp/x.mp4"}),
        ):
            r = client.post(
                endpoint,
                json={
                    "video_id": "11111111-2222-3333-4444-555555555555",
                    "zones": [{"kind": "board", "polygon": bad_polygon}],
                    **extra,
                },
            )
            assert r.status_code == 422, (endpoint, bad_polygon, r.text)


def test_zone_polygon_rejects_non_finite_coordinates():
    """NaN/Infinity coordinates must fail validation (finite-only, to match
    the dashboard's parseZones), not flow into geometry math. Asserted at the
    model level: FastAPI's default 422 handler cannot JSON-encode the
    offending non-finite input value in its error detail, so the wire-level
    response shape for this case is owned by the app layer, while rejection
    itself is owned by RederiveRequest validation."""
    import math

    import pytest
    from pydantic import ValidationError

    from app.models import RederiveRequest

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            RederiveRequest.model_validate(
                {
                    "video_id": "11111111-2222-3333-4444-555555555555",
                    "zones": [
                        {
                            "kind": "board",
                            "polygon": [[bad, 0.1], [0.1, 0.1], [0.2, 0.2]],
                        }
                    ],
                }
            )


def test_zone_polygon_valid_shape_still_accepted():
    """Sanity: a well-formed polygon still passes ZoneIn validation."""
    from app.models import ZoneIn

    zone = ZoneIn.model_validate(
        {"kind": "board", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3]]}
    )
    assert len(zone.polygon) == 3


def test_rederive_with_zero_detections_returns_empty_result(monkeypatch):
    """Zero stored detections is a legitimate outcome (clip with no people):
    /rederive must return a valid empty AnalysisResult, not 409, so zone
    edits on such videos keep working (regression: PUT /zones used to 502)."""

    async def fake_fetch(video_id, dsn=None):
        return []

    async def fake_info(video_id, dsn=None):
        return {"duration_ms": 20_000, "fps": 25.0, "width": 1280, "height": 720}

    monkeypatch.setattr(db, "fetch_detections", fake_fetch)
    monkeypatch.setattr(db, "fetch_video_info", fake_info)

    r = client.post(
        "/rederive",
        json={
            "video_id": "some-id",
            "zones": [
                {
                    "kind": "board",
                    "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],
                }
            ],
        },
    )
    assert r.status_code == 200
    parsed = AnalysisResult.model_validate(r.json())
    assert parsed.tracks == []
    assert parsed.events == []
    assert parsed.video.duration_ms == 20_000
    assert parsed.analytics.teacher_present_ms == 0
    assert parsed.analytics.entries == 0
    assert parsed.analytics.exits == 0


def test_rederive_with_zero_detections_and_no_video_info(monkeypatch):
    """Empty detections + missing videos row must not crash (max_ts default 0)."""

    async def fake_fetch(video_id, dsn=None):
        return []

    async def fake_info(video_id, dsn=None):
        return None

    monkeypatch.setattr(db, "fetch_detections", fake_fetch)
    monkeypatch.setattr(db, "fetch_video_info", fake_info)

    r = client.post("/rederive", json={"video_id": "some-id", "zones": []})
    assert r.status_code == 200
    parsed = AnalysisResult.model_validate(r.json())
    assert parsed.tracks == []
    assert parsed.video.duration_ms == 0


def test_rederive_reruns_roles_and_events_only(monkeypatch):
    """Rederive REBUILDS identities from stored meta.raw_track_id (histograms
    are never persisted, so the merge falls back to spatial continuity) and
    persists the rebuilt track_no back through replace_detections — no
    histograms, no YOLO re-run."""
    teacher = [
        Detection(
            video_ts_ms=ts,
            raw_track_id=5,
            bbox={"x": 0.15, "y": 0.15, "w": 0.1, "h": 0.4},
            conf=0.9,
            standing=True,
            back_to_camera=True,
            track_no=1,
        )
        for ts in range(0, 30_001, 500)
    ]
    student = [
        Detection(
            video_ts_ms=ts,
            raw_track_id=8,
            bbox={"x": 0.7, "y": 0.6, "w": 0.1, "h": 0.2},
            conf=0.8,
            standing=False,
            back_to_camera=False,
            track_no=2,
        )
        for ts in range(0, 30_001, 500)
    ]

    async def fake_fetch(video_id, dsn=None):
        return teacher + student

    async def fake_info(video_id, dsn=None):
        return {"duration_ms": 60_000, "fps": 30.0, "width": 1280, "height": 720}

    persisted: dict = {}

    async def fake_replace(video_id, detections, **kwargs):
        persisted["rows"] = list(detections)
        return len(detections)

    monkeypatch.setattr(db, "fetch_detections", fake_fetch)
    monkeypatch.setattr(db, "fetch_video_info", fake_info)
    monkeypatch.setattr(db, "replace_detections", fake_replace)

    r = client.post(
        "/rederive",
        json={
            "video_id": "22222222-3333-4444-5555-666666666666",
            "zones": [
                {
                    "kind": "board",
                    "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],
                }
            ],
        },
    )
    assert r.status_code == 200
    parsed = AnalysisResult.model_validate(r.json())
    assert parsed.video.duration_ms == 60_000
    assert len(parsed.tracks) == 2
    roles = {t.track_no: t.role for t in parsed.tracks}
    assert roles[1] == "teacher" and roles[2] == "student"
    # raw ids reconstructed from stored meta
    assert [t for t in parsed.tracks if t.track_no == 1][0].meta.raw_track_ids == [5]
    assert parsed.analytics.teacher_board_ms is not None
    assert parsed.analytics.teacher_board_ms > 0
    assert parsed.analytics.presence_intervals == [[0, 30_000]]
    # rebuilt identity numbers were persisted back to detection_events
    assert persisted["rows"], "rederive must persist the rebuilt identities"
    assert {r.track_no for r in persisted["rows"]} == {1, 2}
