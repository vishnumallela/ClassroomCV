"""jobs.submit idempotency + graceful video-deleted abort.

Regression for a duplicate-work defect: the dashboard's Workflow DevKit step
runs at-least-once, so a retry after a lost HTTP response re-POSTs /analyze.
Without idempotency_key dedup the service enqueued a DUPLICATE full YOLO job.

Tests isolate the module-global registry/queue with monkeypatch (fresh dicts,
fresh queue, worker disabled) so nothing actually runs and queue depth can be
asserted exactly.
"""

import queue

import pytest

from app import db, jobs
from app.models import VideoMeta


@pytest.fixture
def isolated_registry(monkeypatch):
    """Fresh job registry + queue, with the worker thread disabled."""
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_jobs_by_key", {})
    monkeypatch.setattr(jobs, "_queue", queue.Queue())
    monkeypatch.setattr(jobs, "_ensure_worker", lambda: None)


def test_submit_same_key_returns_existing_job_and_enqueues_once(
    isolated_registry,
):
    j1 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    j2 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    assert j1.id == j2.id
    assert j2 is j1
    assert jobs._queue.qsize() == 1  # only ONE job was enqueued
    assert len(jobs._jobs) == 1


def test_submit_dedups_even_after_job_reaches_terminal_state(isolated_registry):
    """A late retry (job already done/failed) must still get the original
    job_id back, never a fresh duplicate YOLO run."""
    j1 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    j1.status = "done"
    j2 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    assert j2.id == j1.id
    assert j2.status == "done"
    assert jobs._queue.qsize() == 1

    j1.status = "failed"
    j3 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    assert j3.id == j1.id
    assert jobs._queue.qsize() == 1


def test_submit_different_keys_enqueue_two_jobs(isolated_registry):
    """Distinct keys (e.g. a reanalyze with a fresh attemptId) must NOT dedup."""
    j1 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:initial")
    j2 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [], idempotency_key="vid-1:attempt-2")
    assert j1.id != j2.id
    assert jobs._queue.qsize() == 2
    assert len(jobs._jobs) == 2


def test_submit_without_key_never_dedups(isolated_registry):
    """Backwards compatible: keyless submits always enqueue a new job."""
    j1 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [])
    j2 = jobs.submit("vid-1", "/tmp/a.mp4", 5.0, [])
    assert j1.id != j2.id
    assert jobs._queue.qsize() == 2


def test_analyze_endpoint_passes_idempotency_key_through(
    isolated_registry, monkeypatch
):
    """POST /analyze twice with the same idempotency_key -> same job_id."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    body = {
        "video_id": "11111111-2222-3333-4444-555555555555",
        "video_path": "/tmp/a.mp4",
        "idempotency_key": "11111111-2222-3333-4444-555555555555:initial",
    }
    r1 = client.post("/analyze", json=body)
    r2 = client.post("/analyze", json=body)
    assert r1.status_code == 202 and r2.status_code == 202
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert jobs._queue.qsize() == 1


def test_run_pipeline_propagates_video_deleted_unwrapped(monkeypatch):
    """VideoDeletedError from the DB fence must reach the worker loop as-is
    (not wrapped in RuntimeError) so it becomes a graceful
    'video deleted during analysis' job failure."""

    def fake_detect(video_path, sample_fps, progress_cb=None):
        meta = VideoMeta(duration_ms=1000, fps=5.0, width=100, height=100)
        return meta, [], {}, {}

    async def fake_replace(video_id, detections, dsn=None, batch_size=5000, run_tokens=None, track_hists=None, track_embeds=None):
        raise db.VideoDeletedError(f"video {video_id} deleted during analysis")

    monkeypatch.setattr(jobs.detector, "detect_video", fake_detect)
    monkeypatch.setattr(jobs.db, "replace_detections", fake_replace)

    with pytest.raises(db.VideoDeletedError):
        jobs.run_pipeline("vid-gone", "/tmp/a.mp4", 5.0, [], write_db=True)
