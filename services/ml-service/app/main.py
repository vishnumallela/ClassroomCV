"""Classroom Surveillance ML service (FastAPI).

Routes per SPEC.md "ML service API":
- GET  /health
- POST /analyze            -> 202 {job_id}, runs in the single worker thread
- GET  /jobs/{job_id}      -> status/progress/stage/error
- GET  /jobs/{job_id}/result -> AnalysisResult (404 until done)
- POST /rederive           -> synchronous re-derive (roles+events) from stored
                              detection_events, WITHOUT re-running YOLO
- POST /detect-board       -> board zone proposal (YOLO-World / SAM 2 chain);
                              400 on bad/missing video_path
"""

from __future__ import annotations

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
        video_path, is_temp = detector.resolve_video_source(req.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        result = board_detect.detect_board(req.video_id, video_path)
    finally:
        if is_temp:
            try:
                os.unlink(video_path)
            except OSError:
                pass
    return DetectBoardResponse(**result)


@app.post("/detect-door", response_model=DetectBoardResponse)
def detect_door(req: DetectBoardRequest) -> DetectBoardResponse:
    """Propose a door zone polygon for a stored video.

    Same SAM 2 / YOLO-World chain and response contract as /detect-board, with
    door-shaped geometric scoring (tall, narrow, reaching toward the floor).
    """
    try:
        video_path, is_temp = detector.resolve_video_source(req.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        result = board_detect.detect_door(req.video_id, video_path)
    finally:
        if is_temp:
            try:
                os.unlink(video_path)
            except OSError:
                pass
    return DetectBoardResponse(**result)


@app.post("/rederive", response_model=AnalysisResult)
async def rederive(req: RederiveRequest) -> dict:
    """Re-derive identities + roles + events from stored detection_events.

    Identities are REBUILT from meta.raw_track_id (remerge_from_raw): the
    stored track_no may come from an older merge (e.g. histogram-driven
    chimeras), so /rederive is the cheap way to apply merge/role fixes
    without a 30-40 min YOLO re-run. The refreshed track_no assignment is
    written back to detection_events through the same transactional replace
    machinery /analyze uses, keeping stored rows consistent with the
    returned tracks/events/analytics.
    """
    try:
        detections = await db.fetch_detections(req.video_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")
    # Zero stored detections is a legitimate outcome of a successful analysis
    # (e.g. a short clip with no detectable people): derive an empty-but-valid
    # result instead of erroring, so zone edits on such videos keep working.

    info = await db.fetch_video_info(req.video_id) or {}
    max_ts = max((d.video_ts_ms for d in detections), default=0)
    meta = VideoMeta(
        duration_ms=int(info.get("duration_ms") or max_ts),
        fps=float(info.get("fps") or 0.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
    )
    identities: list[dict] = []
    track_hists: dict[int, list[float]] = {}
    track_embeds: dict[int, list[float]] = {}
    if detections:
        try:
            track_hists = await db.fetch_track_hists(req.video_id)
        except Exception:
            track_hists = {}
        try:
            track_embeds = await db.fetch_track_embeds(req.video_id)
        except Exception:
            track_embeds = {}
        identities = jobs.remerge_from_raw(detections, track_hists, track_embeds)
    result = jobs.derive_result(
        meta,
        detections,
        identities,
        [z.model_dump() for z in req.zones],
        track_embeds=track_embeds,
    )
    if detections:
        # Persist the rebuilt identity numbers (including teacher-fragment
        # absorption applied by derive_result) so detection_events.track_no
        # matches the tracks/analytics we return.
        try:
            await db.replace_detections(
                req.video_id,
                detections,
                track_hists=track_hists,
                track_embeds=track_embeds,
            )
        except db.VideoDeletedError:
            raise HTTPException(
                status_code=409, detail="video was deleted during rederive"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"database unavailable: {exc}"
            )
    return result
