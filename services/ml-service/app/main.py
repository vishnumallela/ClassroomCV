"""Classroom Surveillance ML service (FastAPI).

Routes per SPEC.md "ML service API":
- GET  /health
- POST /analyze            -> 202 {job_id}, runs in the single worker thread
- GET  /jobs/{job_id}      -> status/progress/stage/error
- GET  /jobs/{job_id}/result -> AnalysisResult (404 until done)
- POST /rederive           -> re-derive (roles+events) from stored detection_events
                              detection_events, WITHOUT re-running YOLO
- POST /detect-board       -> board zone proposal (YOLO-World / SAM 2 chain);
                              400 on bad/missing video_path
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException

from app import board_detect, db, detector, jobs
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

app = FastAPI(title="Classroom Surveillance ML Service")


@app.get("/health")
def health() -> dict:
    get_settings()  # ensure settings load cleanly
    return {
        "status": "ok",
        "device": detector.get_device(),
        "model_loaded": detector.model_loaded(),
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


@app.post("/detect-board", response_model=DetectBoardResponse)
def detect_board(req: DetectBoardRequest) -> DetectBoardResponse:
    """Propose a board zone polygon for a stored video.

    Sync def route: FastAPI runs it in the threadpool, so the seconds-long
    SAM 2 inference does not block the event loop. Path validation reuses
    detector._validate_video_path (same SSRF/arbitrary-read guard as
    /analyze) and maps its rejection to 400 per the feature contract.
    """
    try:
        # Shared cache: /detect-board, /detect-door and /analyze all want the
        # same file, and on a remote worker each fetch is the whole video.
        video_path = detector.resolve_video_cached(req.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = board_detect.detect_board(req.video_id, video_path)
    return DetectBoardResponse(**result)


@app.post("/detect-door", response_model=DetectBoardResponse)
def detect_door(req: DetectBoardRequest) -> DetectBoardResponse:
    """Propose a door zone polygon for a stored video.

    Same SAM 2 / YOLO-World chain and response contract as /detect-board, with
    door-shaped geometric scoring (tall, narrow, reaching toward the floor).
    """
    try:
        # Shared cache: /detect-board, /detect-door and /analyze all want the
        # same file, and on a remote worker each fetch is the whole video.
        video_path = detector.resolve_video_cached(req.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = board_detect.detect_door(req.video_id, video_path)
    return DetectBoardResponse(**result)


@app.post("/rederive/start", status_code=202, response_model=AnalyzeAccepted)
def rederive_start(req: RederiveRequest) -> AnalyzeAccepted:
    """Queue a re-derive; poll /jobs/{id} exactly as for /analyze.

    The synchronous /rederive below holds the connection for the whole derive,
    which a long lesson outlives -- the caller's socket idles out and it retries
    while the completed work is discarded. Prefer this for anything long.
    """
    job = jobs.submit_rederive(req.video_id, [z.model_dump() for z in req.zones])
    return AnalyzeAccepted(job_id=job.id)


@app.post("/rederive", response_model=AnalysisResult)
async def rederive(req: RederiveRequest) -> dict:
    """Re-derive identities + roles + events from stored detection_events.

    SYNCHRONOUS: fine for short clips, but a long lesson takes minutes and the
    caller's connection will not survive it -- use /rederive/start for those.
    """
    try:
        return await jobs.rederive_video(
            req.video_id, [z.model_dump() for z in req.zones]
        )
    except db.VideoDeletedError:
        raise HTTPException(status_code=409, detail="video was deleted during rederive")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"rederive failed: {exc}")
