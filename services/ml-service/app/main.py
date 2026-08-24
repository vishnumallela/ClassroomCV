"""Classroom Surveillance ML service (FastAPI).

Routes:
- GET  /health
- POST /analyze            -> 202 {job_id}, runs in the single worker thread
- GET  /jobs/{job_id}      -> status/progress/stage/error
- GET  /jobs/{job_id}/result -> AnalysisResult (404 until done)
- POST /rederive           -> synchronous re-derive from stored detections,
                              WITHOUT re-running the detector
- POST /detect-board       -> board zone proposal from RF-DETR screen boxes
- POST /detect-door        -> door zone proposal from RF-DETR door boxes
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from app import db, detector, jobs, zones as zones_mod
from app.config import get_settings
from app.models import (
    AnalysisResult,
    AnalyzeAccepted,
    AnalyzeRequest,
    DetectBoardRequest,
    DetectBoardResponse,
    JobStatusOut,
    RederiveRequest,
    VideoMeta,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Classroom Surveillance ML Service")

# Frames sampled for a zone proposal. The board and door do not move, so this
# is about beating occasional false positives with a median, not about
# catching a moment — measured presence is 96-100% (screen) and 77-81% (door),
# and their centres vary by +/-0.002 over a whole lesson.
#
# 0.5 is the sparsest rate iter_frames actually honours (it floors there), so
# asking for less would quietly get this anyway. A 5-minute lesson gives ~140
# frames to take a median over, which is far more than the estimate needs.
ZONE_SAMPLE_FPS = 0.5
# Below this many sampled frames a median is not worth the name, so a short
# clip gets a denser second pass rather than no zone at all.
ZONE_MIN_FRAMES = 12
ZONE_DENSE_FPS = 2.0


@app.get("/health")
def health() -> dict:
    get_settings()  # ensure settings load cleanly
    return {
        "status": "ok",
        "device": detector.get_device(),
        "model": detector.resolve_model_name(),
        "model_loaded": detector.model_loaded(),
        # What is ACTUALLY serving, read back rather than inferred from config:
        # "TensorRT is enabled" and "TensorRT is running" are different claims.
        "backend": detector.serving_backend(),
    }


@app.post("/analyze", status_code=202, response_model=AnalyzeAccepted)
def analyze(req: AnalyzeRequest) -> AnalyzeAccepted:
    job = jobs.submit(
        video_id=req.video_id,
        video_path=req.video_path,
        sample_fps=req.sample_fps,
        zones=[z.model_dump() for z in req.zones],
        idempotency_key=req.idempotency_key,
        run_tokens=req.run_tokens,
    )
    return AnalyzeAccepted(job_id=job.id)


@app.get("/jobs/{job_id}", response_model=JobStatusOut)
def job_status(job_id: str) -> JobStatusOut:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return JobStatusOut(
        status=job.status, progress=job.progress, stage=job.stage, error=job.error
    )


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=404, detail="result not available")
    return job.result


def _propose(video_path: str, kind: str) -> dict:
    """Sample a video sparsely and place `kind`'s zone from the detections.

    Sync path: FastAPI runs these routes in the threadpool, so the seconds of
    inference do not block the event loop. Path validation reuses the same
    SSRF/arbitrary-read guard /analyze uses.
    """
    try:
        local_path, is_temp = detector.resolve_video_source(video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        _meta, detections = detector.detect_video(local_path, sample_fps=ZONE_SAMPLE_FPS)
        frames = len({d.video_ts_ms for d in detections})
        if frames < ZONE_MIN_FRAMES:
            # A short clip yields too few sparse samples to median over; take a
            # denser pass rather than refusing to place a zone.
            _meta, detections = detector.detect_video(local_path, sample_fps=ZONE_DENSE_FPS)
            frames = len({d.video_ts_ms for d in detections})
        return zones_mod.propose_zone(detections, kind, frames_seen=frames)
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except OSError:
                pass


@app.post("/detect-board", response_model=DetectBoardResponse)
def detect_board(req: DetectBoardRequest) -> DetectBoardResponse:
    """Propose a board zone from the lesson's own screen detections."""
    return DetectBoardResponse(**_propose(req.video_path, "board"))


@app.post("/detect-door", response_model=DetectBoardResponse)
def detect_door(req: DetectBoardRequest) -> DetectBoardResponse:
    """Propose a door zone from the lesson's own door detections."""
    return DetectBoardResponse(**_propose(req.video_path, "door"))


@app.post("/rederive", response_model=AnalysisResult)
async def rederive(req: RederiveRequest) -> dict:
    """Re-derive the teacher timeline + events from stored detections.

    The cheap way to apply a zone edit: board time and entry/exit both depend
    on the zone polygons, and recomputing them from stored rows takes
    milliseconds against the minutes a detection re-run would cost.
    """
    try:
        detections = await db.fetch_detections(req.video_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")
    # Zero stored detections is a legitimate outcome of a successful analysis
    # (a lesson where the teacher was never found): derive an empty-but-valid
    # result instead of erroring, so zone edits on such videos keep working.

    info = await db.fetch_video_info(req.video_id) or {}
    max_ts = max((d.video_ts_ms for d in detections), default=0)
    meta = VideoMeta(
        duration_ms=int(info.get("duration_ms") or max_ts),
        fps=float(info.get("fps") or 0.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
    )
    result = jobs.derive_result(
        meta, detections, [z.model_dump() for z in req.zones]
    )
    if detections:
        # Persist the refreshed teacher assignment so detection_events matches
        # the tracks/analytics we just returned.
        try:
            await db.replace_detections(req.video_id, detections)
        except db.VideoDeletedError:
            raise HTTPException(
                status_code=409, detail="video was deleted during rederive"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"database unavailable: {exc}"
            )
    return result
