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

from app import db, detector, events as events_mod, merge, roles, teacher_chain, vlm_teacher
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
    # worker can fetch the video itself); delete it as soon as detection has
    # read every frame — merge/derive/write never touch the file.
    local_path, is_temp = detector.resolve_video_source(video_path)
    try:
        meta, detections, hists, embeds = detector.detect_video(
            local_path,
            sample_fps=sample_fps,
            progress_cb=lambda f: cb("detecting", f * 0.8),
        )
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except OSError:
                pass
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
        detections, hists, {rid: [e] for rid, e in embeds.items()}
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
        track_embeds={rid: list(e) for rid, e in embeds.items()},
        video_path=video_path,
    )
    derive_s = time.perf_counter() - stage_start
    cb("deriving", 0.95)

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
        {rid: [h] for rid, h in (track_hists or {}).items()},
        {rid: [e] for rid, e in (track_embeds or {}).items()},
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


def derive_result(
    meta: VideoMeta,
    detections: list[Detection],
    identities: list[dict],
    zones: list[dict],
    track_embeds: Optional[dict[int, list[float]]] = None,
    video_path: Optional[str] = None,
) -> dict:
    """roles + events + analytics from merged detections. Shared by analyze & rederive.

    After the teacher is chosen, her timeline is rebuilt by
    teacher_chain.stitch_teacher: detection ranges stolen by student raw ids
    (crouch handoffs, merge chimeras) are claimed back, and ranges of her
    merged identity the chain rejects (e.g. a corner student folded in by a
    bad merge) are evicted to fresh student tracks. Detection.track_no is
    rewritten IN PLACE so callers persisting `detections` store the same
    identity view the analytics were computed from.
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

    features = roles.compute_features(
        dets_by_track,
        meta.duration_ms,
        board_polygon=board_polygon,
        raw_ids_by_track=raw_ids_by_track,
    )
    roles_map = roles.assign_roles(features, has_board=board_polygon is not None)

    teacher_no = next(
        (t for t, (role, _) in roles_map.items() if role == "teacher"), None
    )

    # Vision-LLM fallback: the geometric ranker gives up (all-unknown) on a teacher
    # who sits the whole lesson, because she scores like a student. Ask a vision
    # model to point at the adult instructor and map that to a track. Only fires
    # when nothing was selected AND we still have the video file, so easy videos
    # never touch the API. The VLM-picked track is trusted as-is (skip the stitch,
    # which is tuned for mobile teachers and would risk chimeras here).
    teacher_from_vlm = False
    if teacher_no is None and video_path is not None:
        vlm = vlm_teacher.identify_teacher(video_path, dets_by_track, meta.duration_ms)
        if vlm is not None and vlm[0] in dets_by_track:
            teacher_no, conf, _votes = vlm
            roles_map[teacher_no] = ("teacher", conf)
            teacher_from_vlm = True

    if teacher_no is not None and not teacher_from_vlm:
        embeds_by_raw = None
        if track_embeds:
            embeds_by_raw = {
                int(rid): np.asarray(e, dtype=np.float64).ravel()
                for rid, e in track_embeds.items()
            }
        stitched = teacher_chain.stitch_teacher(
            teacher_no, dets_by_track, embeds_by_raw
        )
        if stitched is not None:
            claims, evictions = stitched
            next_no = max(dets_by_track, default=0) + 1
            for frag, lo, hi in evictions:
                for d in frag.dets[lo:hi]:
                    d.track_no = next_no
                roles_map[next_no] = ("student", None)
                next_no += 1
            for c in claims:
                for d in c.dets:
                    d.track_no = teacher_no
            # The chain moved detections between identities: rebuild the
            # per-track views from the rewritten track_nos, drop identities
            # the chain fully consumed, and recompute features so spans,
            # overlays, and analytics all describe the stitched timeline.
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
            roles_map = {
                no: role for no, role in roles_map.items() if no in dets_by_track
            }
            logger.info(
                "teacher chain: %d claim(s) %s, %d evicted range(s) from track %d",
                len(claims),
                [(c.fragment.raw_id, c.fragment.dets[c.from_idx].video_ts_ms) for c in claims],
                len(evictions),
                teacher_no,
            )
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
