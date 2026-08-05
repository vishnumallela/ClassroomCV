"""In-memory job registry + single background worker thread.

One daemon worker thread pulls jobs off a queue, so only one analysis runs at
a time. Job progress mapping: detection = 0..0.8, merging = 0.8..0.9,
deriving + DB write = 0.9..1.0.

run_pipeline / derive_result are also directly callable (used by /rederive and
by tests, which monkeypatch app.detector.detect_video / app.db.replace_detections).
Module boundaries:
- /analyze merges with torso histograms + CLIP track embeddings collected
  during detection;
- /rederive REBUILDS identities from stored detections' meta.raw_track_id via
  remerge_from_raw, feeding back the per-track hists/embeds persisted in the
  first-row meta (rows written before that persistence existed fall back to
  spatial continuity — see merge.spatial_continuity) and then rewrites
  detection_events.track_no through the same replace machinery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from app import db, detector, events as events_mod, merge, roles, teacher_id, teacher_track
from app.geometry import rdp_indices
from app.models import AnalysisResult, Detection, VideoMeta

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]

# Permanent overlay tier (plan section 6): per-track RDP center polyline +
# sparse bbox keyframes stored in tracks.meta, so playback overlays survive
# detection_events compression/retention.
OVERLAY_RDP_EPSILON = 0.005
OVERLAY_KEYFRAME_MS = 2_000

# A video longer than this that decodes to ZERO frames or ZERO detections is
# a failed analysis (codec/model breakage), not a legitimate empty result:
# ingesting it as 'done' silently zeroes every dashboard metric.
EMPTY_RESULT_GUARD_MS = 5_000


@dataclass
class Job:
    id: str
    video_id: str
    status: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    stage: str = "detecting"  # detecting | merging | deriving
    error: Optional[str] = None
    result: Optional[dict] = None
    idempotency_key: Optional[str] = None


_jobs: dict[str, Job] = {}
# idempotency_key -> job id, guarded by _lock alongside _jobs.
_jobs_by_key: dict[str, str] = {}
# Bounded so a burst of /analyze submits backpressures the caller (put blocks)
# instead of holding an unbounded backlog of queued videos in memory.
_queue: "queue.Queue[tuple[Job, dict]]" = queue.Queue(maxsize=4)
_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_worker_loop, name="analysis-worker", daemon=True
            )
            _worker.start()


def submit(
    video_id: str,
    video_path: str,
    sample_fps: float,
    zones: list[dict],
    idempotency_key: Optional[str] = None,
    run_tokens: Optional[list[str]] = None,
) -> Job:
    """Register + enqueue a job. Idempotent on `idempotency_key`.

    The caller (a Workflow DevKit step) runs at-least-once: a retry after a
    lost HTTP response re-POSTs the same request. If a job — non-terminal OR
    terminal — already exists for the key, return it instead of enqueuing a
    duplicate full YOLO run; the caller then polls the original job as usual.
    Check + registration happen under _lock so two concurrent submits with
    the same key can never both enqueue.

    `run_tokens` are forwarded to db.replace_detections so a job whose run
    has been superseded by a newer reanalyze cannot rewrite detection_events.
    """
    with _lock:
        if idempotency_key is not None:
            existing_id = _jobs_by_key.get(idempotency_key)
            if existing_id is not None and existing_id in _jobs:
                existing = _jobs[existing_id]
                logger.info(
                    "duplicate /analyze submit for key %s -> returning existing job %s (%s)",
                    idempotency_key,
                    existing.id,
                    existing.status,
                )
                return existing
        job = Job(
            id=str(uuid.uuid4()),
            video_id=video_id,
            idempotency_key=idempotency_key,
        )
        _jobs[job.id] = job
        if idempotency_key is not None:
            _jobs_by_key[idempotency_key] = job.id
    _queue.put(
        (
            job,
            {
                "video_path": video_path,
                "sample_fps": sample_fps,
                "zones": zones,
                "run_tokens": list(run_tokens or []),
            },
        )
    )
    _ensure_worker()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def _worker_loop() -> None:  # pragma: no cover - exercised via smoke test
    while True:
        job, params = _queue.get()
        job.status = "running"

        def cb(stage: str, frac: float, _job: Job = job) -> None:
            _job.stage = stage
            _job.progress = round(min(1.0, max(_job.progress, frac)), 4)

        try:
            result = run_pipeline(
                job.video_id,
                params["video_path"],
                params["sample_fps"],
                params["zones"],
                progress_cb=cb,
                write_db=True,
                run_tokens=params.get("run_tokens") or None,
            )
            job.result = result
            job.progress = 1.0
            job.status = "done"
        except db.VideoDeletedError:
            # Graceful abort, not a crash: the video was deleted mid-analysis,
            # nothing was written (the fence rolled the transaction back), and
            # the worker stays alive for the next job.
            logger.info(
                "analysis job %s aborted: video %s deleted during analysis",
                job.id,
                job.video_id,
            )
            job.error = "video deleted during analysis"
            job.status = "failed"
        except db.StaleRunError as exc:
            # Same graceful shape: a newer reanalyze/rederive owns the video
            # now; this job's rows were rolled back, nothing to clean up.
            logger.info(
                "analysis job %s aborted: superseded run for video %s (%s)",
                job.id,
                job.video_id,
                exc,
            )
            job.error = "analysis run superseded by a newer request"
            job.status = "failed"
        except Exception as exc:
            logger.exception("analysis job %s failed", job.id)
            job.error = str(exc) or exc.__class__.__name__
            job.status = "failed"
        finally:
            _queue.task_done()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(
    video_id: str,
    video_path: str,
    sample_fps: float,
    zones: list[dict],
    progress_cb: Optional[ProgressCb] = None,
    write_db: bool = True,
    run_tokens: Optional[list[str]] = None,
) -> dict:
    """detect -> merge -> roles+events -> (COPY to DB). Returns AnalysisResult dict.

    The DB write happens AFTER derivation so teacher-fragment absorption
    (derive_result may fold short fragments into the teacher identity,
    rewriting their track_no) is reflected in the stored detection_events.
    """
    cb: ProgressCb = progress_cb or (lambda stage, frac: None)

    cb("detecting", 0.0)
    stage_start = time.perf_counter()
    # Resolve an allowlisted object-store URL to a local temp (so a remote GPU
    # worker can fetch the video itself). The file is kept until AFTER derive:
    # the teacher-id vote decodes a handful of frames there. Deleted in the
    # outer finally either way.
    local_path, is_temp = detector.resolve_video_source(video_path)
    try:
        meta, detections, hists, embeds = detector.detect_video(
            local_path,
            sample_fps=sample_fps,
            progress_cb=lambda f: cb("detecting", f * 0.8),
        )
        detect_s = time.perf_counter() - stage_start

        if not detections:
            probed_ms = meta.duration_ms
            if probed_ms <= 0 and write_db:
                # 0 decoded frames leaves duration unknown; fall back to the
                # dashboard's ffprobe duration (best-effort, absent in tests).
                info = asyncio.run(db.fetch_video_info(video_id)) or {}
                probed_ms = int(info.get("duration_ms") or 0)
            if probed_ms > EMPTY_RESULT_GUARD_MS:
                raise RuntimeError(
                    f"analysis produced zero detections for a "
                    f"{probed_ms / 1000.0:.1f}s video — treating the empty result "
                    f"as a failure instead of silently zeroing all analytics"
                )

        cb("merging", 0.8)
        stage_start = time.perf_counter()
        raw_tracks = merge.build_raw_tracks(
            detections,
            hists,
            {rid: detector.gallery_vectors(g) for rid, g in embeds.items()},
        )
        mapping, identities = merge.merge_tracks(raw_tracks)
        for d in detections:
            d.track_no = mapping.get(d.raw_track_id)
        detections = [d for d in detections if d.track_no is not None]
        merge_s = time.perf_counter() - stage_start
        cb("merging", 0.9)

        cb("deriving", 0.9)
        stage_start = time.perf_counter()
        result = derive_result(
            meta,
            detections,
            identities,
            zones,
            track_embeds=embeds,
            track_hists=hists,
            video_path=local_path,
        )
        derive_s = time.perf_counter() - stage_start
        cb("deriving", 0.95)
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except OSError:
                pass

    if write_db:
        try:
            asyncio.run(
                db.replace_detections(
                    video_id,
                    detections,
                    run_tokens=run_tokens,
                    track_hists={
                        t.raw_id: [float(v) for v in t.hist.ravel()]
                        for t in raw_tracks
                        if t.hist is not None
                    },
                    # One median vector per raw track, not the whole gallery:
                    # the gallery is ~10x the bytes and rides in a jsonb meta
                    # column. /rederive therefore re-identifies from a single
                    # view per track (teacher_track handles both shapes).
                    track_embeds={
                        t.raw_id: [float(v) for v in t.embed.ravel()]
                        for t in raw_tracks
                        if t.embed is not None
                    },
                )
            )
        except (db.VideoDeletedError, db.StaleRunError):
            # Propagate untouched: the worker loop turns these into graceful
            # job failures ('video deleted' / 'run superseded').
            raise
        except Exception as exc:
            raise RuntimeError(
                f"database write failed for video {video_id}: {exc}"
            ) from exc
    cb("deriving", 1.0)
    logger.info(
        "pipeline stage timings for video %s: detect_s=%.2f merge_s=%.2f derive_s=%.2f",
        video_id,
        detect_s,
        merge_s,
        derive_s,
    )
    return result


def remerge_from_raw(
    detections: list[Detection],
    track_hists: Optional[dict[int, list[float]]] = None,
    track_embeds: Optional[dict[int, list[float]]] = None,
) -> list[dict]:
    """Rebuild identities from stored detections' raw_track_id (for /rederive).

    When persisted per-track histograms / CLIP embeddings are available they
    are fed back into the merge so /rederive scores appearance exactly like
    /analyze did; otherwise spatial continuity carries the appearance slot.
    Mutates each Detection's track_no to the fresh identity number and
    returns the identity summaries.
    """
    raw_tracks = merge.build_raw_tracks(
        detections,
        {rid: _appearance_samples(h) for rid, h in (track_hists or {}).items()},
        {rid: _appearance_samples(e) for rid, e in (track_embeds or {}).items()},
    )
    mapping, identities = merge.merge_tracks(raw_tracks)
    for d in detections:
        d.track_no = mapping.get(d.raw_track_id)
    return identities


def _track_overlay(dets: list[Detection]) -> dict:
    """RDP-simplified center polyline + bbox keyframes for one merged track."""
    dets = sorted(dets, key=lambda d: d.video_ts_ms)
    centers = [
        (d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0)
        for d in dets
    ]
    polyline = [
        [dets[i].video_ts_ms, round(centers[i][0], 4), round(centers[i][1], 4)]
        for i in rdp_indices(centers, OVERLAY_RDP_EPSILON)
    ]
    keyframes: list[list[float]] = []
    next_ts: Optional[int] = None
    for d in dets:
        if next_ts is not None and d.video_ts_ms < next_ts:
            continue
        b = d.bbox
        keyframes.append(
            [
                d.video_ts_ms,
                round(b["x"], 4),
                round(b["y"], 4),
                round(b["w"], 4),
                round(b["h"], 4),
            ]
        )
        next_ts = d.video_ts_ms + OVERLAY_KEYFRAME_MS
    return {"polyline": polyline, "keyframes": keyframes}


def _appearance_samples(value) -> list[list[float]]:
    """One track's appearance evidence as plain vectors, whatever shape it is in."""
    return detector.gallery_vectors(value)


def _zero_shot_ages(track_embeds: Optional[dict]) -> Optional[dict[int, float]]:
    """CLIP adult-vs-child readings, but only when CLIP is already resident.

    /analyze has just embedded every crop, so the text encode is microseconds.
    /rederive and the offline harness have no crops and no reason to pull a
    350 MB checkpoint onto the box for one optional signal, so they simply run
    without it — the assignment is designed to lose signals gracefully.
    """
    if not track_embeds or not detector.clip_loaded():
        return None
    try:
        return detector.zero_shot_adult(track_embeds)
    except Exception:
        logger.warning("zero-shot age scoring failed", exc_info=True)
        return None


def _apply_timeline(
    timeline: Optional["teacher_track.TeacherTimeline"],
    detections: list[Detection],
    dets_by_track: dict[int, list[Detection]],
) -> Optional[int]:
    """Stamp the teacher's timeline onto ONE identity; evict everyone else from it.

    The identity that already holds most of her detections keeps its number, so
    a correct merge is left alone and only the disagreements move. Detections
    of OTHER people that the merge had put in that identity are evicted to
    fresh identities, one per raw tracker id so an evicted stretch stays a
    coherent person rather than a bag of fragments.
    """
    if timeline is None or not timeline.tracklets:
        return None
    hers = timeline.detections()
    claimed = {id(d) for d in hers}

    counts: dict[int, int] = {}
    for d in hers:
        if d.track_no is not None:
            counts[d.track_no] = counts.get(d.track_no, 0) + 1
    next_no = max(dets_by_track, default=0) + 1
    teacher_no = max(counts, key=lambda no: counts[no]) if counts else next_no
    if teacher_no >= next_no:
        next_no = teacher_no + 1

    evicted: dict[int, int] = {}
    for d in detections:
        if d.track_no == teacher_no and id(d) not in claimed:
            if d.raw_track_id not in evicted:
                evicted[d.raw_track_id] = next_no
                next_no += 1
            d.track_no = evicted[d.raw_track_id]
    for d in hers:
        d.track_no = teacher_no

    logger.info(
        "teacher timeline: identity %d from %d tracklet(s) over raw ids %s "
        "(%d detections, confidence %.2f); evicted %d foreign fragment(s)",
        teacher_no,
        len(timeline.tracklets),
        timeline.raw_ids,
        len(hers),
        timeline.confidence,
        len(evicted),
    )
    return teacher_no


def _roles_from_timeline(
    dets_by_track: dict[int, list[Detection]],
    teacher_no: Optional[int],
    timeline: Optional["teacher_track.TeacherTimeline"],
) -> dict[int, tuple[str, Optional[float]]]:
    """teacher + students, or all-unknown when no adult was identified.

    Everyone stays 'unknown' rather than being called a student when there is
    no teacher: without an identified adult the pipeline has no basis for
    saying what anyone is, and the dashboard degrades to occupancy.
    """
    if teacher_no is None:
        return {no: ("unknown", None) for no in dets_by_track}
    conf = timeline.confidence if timeline is not None else None
    roles_map: dict[int, tuple[str, Optional[float]]] = {
        no: ("student", None) for no in dets_by_track
    }
    roles_map[teacher_no] = ("teacher", conf)
    return roles_map


def derive_result(
    meta: VideoMeta,
    detections: list[Detection],
    identities: list[dict],
    zones: list[dict],
    track_embeds: Optional[dict] = None,
    video_path: Optional[str] = None,
    track_hists: Optional[dict] = None,
) -> dict:
    """roles + events + analytics from merged detections. Shared by analyze & rederive.

    WHO IS THE TEACHER is decided by teacher_track.find_teacher, which solves
    her whole timeline globally over tracklets rather than picking the most
    teacher-looking merged identity. That ordering matters: on real footage her
    timeline routinely lands in several merged identities, which then compete
    with each other and cancel out any margin, so an identity-first pipeline
    reports no teacher at all for a lesson she never left (see the module
    docstring in app/teacher_track.py for the measured case).

    Her detections are then re-stamped onto ONE identity and anybody else's
    are evicted out of it, so the stored detections, the overlays and the
    analytics all describe the same person. Detection.track_no is rewritten IN
    PLACE for callers that persist `detections` afterwards.

    A HARD-BUDGETED vision vote (teacher_id.verify_teacher, at most vlm_frames
    Gemini calls, zero when no key) still gets the last word when it is
    configured and the assignment was not confident: it needs frames, so it
    runs only when video_path is a readable local file.
    """
    dets_by_track: dict[int, list[Detection]] = {}
    for d in detections:
        if d.track_no is not None:
            dets_by_track.setdefault(d.track_no, []).append(d)

    raw_ids_by_track = {
        i["track_no"]: sorted(i["raw_track_ids"]) for i in identities
    }
    board_polygon = next(
        (z["polygon"] for z in zones if z.get("kind") == "board"), None
    )

    dets_by_raw: dict[int, list[Detection]] = {}
    for d in detections:
        dets_by_raw.setdefault(d.raw_track_id, []).append(d)

    timeline = None
    try:
        timeline = teacher_track.find_teacher(
            dets_by_raw,
            galleries=track_embeds,
            hists=track_hists,
            zero_shot=_zero_shot_ages(track_embeds),
            duration_ms=meta.duration_ms,
            zones=zones,
        )
    except Exception:  # a derive must never die on the teacher search
        logger.exception("teacher assignment failed; continuing without a teacher")

    teacher_no = _apply_timeline(timeline, detections, dets_by_track)
    if teacher_no is not None:
        # The assignment moved detections between identities: rebuild the
        # per-track views so spans, overlays and analytics all describe the
        # stitched timeline.
        dets_by_track = {}
        for d in detections:
            if d.track_no is not None:
                dets_by_track.setdefault(d.track_no, []).append(d)
        for dets in dets_by_track.values():
            dets.sort(key=lambda d: d.video_ts_ms)
        raw_ids_by_track = {
            no: sorted({d.raw_track_id for d in dets})
            for no, dets in dets_by_track.items()
        }
    roles_map = _roles_from_timeline(dets_by_track, teacher_no, timeline)

    if video_path is not None:
        try:
            vote = teacher_id.verify_teacher(video_path, dets_by_track, meta.duration_ms)
            voted = teacher_id.apply_vote(vote, teacher_no, roles_map)
            if voted != teacher_no:
                logger.info(
                    "teacher-id vote moved the teacher from %s to %s", teacher_no, voted
                )
                teacher_no = voted
        except Exception:  # the vote is advisory; never sink a derive
            logger.warning("teacher-id vote failed; keeping the assigned teacher", exc_info=True)

    features = roles.compute_features(
        dets_by_track,
        meta.duration_ms,
        board_polygon=board_polygon,
        raw_ids_by_track=raw_ids_by_track,
    )

    events, analytics = events_mod.derive(
        dets_by_track, roles_map, meta.duration_ms, zones
    )

    tracks = []
    for f in sorted(features, key=lambda f: f.track_no):
        role, confidence = roles_map.get(f.track_no, ("unknown", None))
        tracks.append(
            {
                "track_no": f.track_no,
                "role": role,
                "role_confidence": confidence,
                "first_ms": f.first_ms,
                "last_ms": f.last_ms,
                "meta": {
                    "standing_ratio": round(f.standing_ratio, 4),
                    "movement": round(f.movement, 4),
                    "raw_track_ids": f.raw_track_ids,
                    "overlay": _track_overlay(dets_by_track.get(f.track_no, [])),
                },
            }
        )

    result = AnalysisResult.model_validate(
        {
            "video": {
                "duration_ms": meta.duration_ms,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
            },
            "tracks": tracks,
            "events": events,
            "analytics": analytics,
        }
    )
    return result.model_dump()


def identities_from_detections(detections: list[Detection]) -> list[dict]:
    """Rebuild identity summaries from stored detections (for /rederive)."""
    by_track: dict[int, dict] = {}
    for d in detections:
        if d.track_no is None:
            continue
        info = by_track.setdefault(
            d.track_no,
            {
                "track_no": d.track_no,
                "raw_track_ids": set(),
                "first_ms": d.video_ts_ms,
                "last_ms": d.video_ts_ms,
            },
        )
        info["raw_track_ids"].add(d.raw_track_id)
        info["first_ms"] = min(info["first_ms"], d.video_ts_ms)
        info["last_ms"] = max(info["last_ms"], d.video_ts_ms)
    return [
        {**info, "raw_track_ids": sorted(info["raw_track_ids"])}
        for info in sorted(by_track.values(), key=lambda i: i["track_no"])
    ]
