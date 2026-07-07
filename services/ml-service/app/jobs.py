"""In-memory job registry + single background worker thread.

One daemon worker thread pulls jobs off a queue, so only one analysis runs at
a time. Job progress mapping: detection = 0..0.8, merging = 0.8..0.9,
deriving + DB write = 0.9..1.0.

run_pipeline / derive_result are also directly callable (used by /rederive and
by tests, which monkeypatch app.detector.detect_video / app.db.replace_detections).
Module boundaries:
- /analyze merges with torso histograms collected during detection;
- /rederive REBUILDS identities from stored detections' meta.raw_track_id via
  remerge_from_raw (histograms are never persisted, so the merge falls back
  to spatial continuity — see merge.spatial_continuity) and then rewrites
  detection_events.track_no through the same replace machinery.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from app import db, detector, events as events_mod, merge, roles
from app.models import AnalysisResult, Detection, VideoMeta

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]

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
_queue: "queue.Queue[tuple[Job, dict]]" = queue.Queue()
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
    meta, detections, hists = detector.detect_video(
        video_path,
        sample_fps=sample_fps,
        progress_cb=lambda f: cb("detecting", f * 0.8),
    )

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
    raw_tracks = merge.build_raw_tracks(detections, hists)
    mapping, identities = merge.merge_tracks(raw_tracks)
    for d in detections:
        d.track_no = mapping.get(d.raw_track_id)
    detections = [d for d in detections if d.track_no is not None]
    cb("merging", 0.9)

    cb("deriving", 0.9)
    result = derive_result(meta, detections, identities, zones)
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
    return result


def remerge_from_raw(
    detections: list[Detection],
    track_hists: Optional[dict[int, list[float]]] = None,
) -> list[dict]:
    """Rebuild identities from stored detections' raw_track_id (for /rederive).

    When persisted per-track histograms are available they are fed back into
    the merge so /rederive scores appearance exactly like /analyze did;
    otherwise spatial continuity carries the appearance slot. Mutates each
    Detection's track_no to the fresh identity number and returns the
    identity summaries.
    """
    raw_tracks = merge.build_raw_tracks(
        detections, {rid: [h] for rid, h in (track_hists or {}).items()}
    )
    mapping, identities = merge.merge_tracks(raw_tracks)
    for d in detections:
        d.track_no = mapping.get(d.raw_track_id)
    return identities


def derive_result(
    meta: VideoMeta,
    detections: list[Detection],
    identities: list[dict],
    zones: list[dict],
) -> dict:
    """roles + events + analytics from merged detections. Shared by analyze & rederive.

    After the teacher is chosen, short unassigned fragments that fit the
    teacher's absence windows near its trajectory/board are folded into the
    teacher identity (roles.absorbable_fragments): their Detection.track_no
    is rewritten IN PLACE so callers persisting `detections` store the same
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
    if teacher_no is not None:
        absorbed = roles.absorbable_fragments(
            teacher_no,
            features,
            dets_by_track,
            meta.duration_ms,
            board_polygon=board_polygon,
            door_polygons=[
                z["polygon"] for z in zones if z.get("kind") == "door"
            ],
        )
        if absorbed:
            for track_no in absorbed:
                fragment_dets = dets_by_track.pop(track_no, [])
                for d in fragment_dets:
                    d.track_no = teacher_no
                dets_by_track[teacher_no].extend(fragment_dets)
                raw_ids_by_track[teacher_no] = sorted(
                    set(raw_ids_by_track.get(teacher_no, []))
                    | set(raw_ids_by_track.pop(track_no, []))
                )
                roles_map.pop(track_no, None)
            dets_by_track[teacher_no].sort(key=lambda d: d.video_ts_ms)
            logger.info(
                "absorbed %d teacher fragment(s) %s into track %d",
                len(absorbed),
                absorbed,
                teacher_no,
            )
            # Recompute features so the teacher's span/meta cover the
            # absorbed fragments; roles keep their already-assigned labels.
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
