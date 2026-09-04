"""In-memory job registry + single background worker thread.

One daemon worker thread pulls jobs off a queue, so only one analysis runs at
a time. Job progress mapping: detection = 0..0.9, deriving + DB write =
0.9..1.0. There is no merge stage any more — the detector names the teacher, so
there are no fragments to reassemble into identities.

run_pipeline / derive_result are also directly callable (used by /rederive and
by tests, which monkeypatch app.detector.detect_video / app.db.replace_detections).

The pipeline is:

    detect (RF-DETR)  ->  gate static classes to their zones
                      ->  follow the teacher
                      ->  KPIs from her timeline
                      ->  store every teacher box, attributed or not

/rederive replays everything after detection from the stored teacher rows, so
editing a zone recomputes board time and entries in milliseconds without
re-running the model.

That last step stores the boxes the timeline REJECTED as well as the ones it
kept (migration 0014). It is the difference between a rederive that can only
replay the tracker's old answer and one that can be given a new question — and
the question that needs asking is which of several adults the lesson assesses,
which nothing decides yet. Until it does, sustained co-presence makes the
lesson report Not Observed rather than a confident number for the wrong person.
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

from app import attribution as attribution_mod, db, detector, events as events_mod, heuristics as heuristics_mod, teacher as teacher_mod, zones as zones_mod
from app.geometry import rdp_indices
from app.models import AnalysisResult, Detection, VideoMeta

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]

# Permanent overlay tier: per-track RDP center polyline + sparse bbox keyframes
# stored in tracks.meta, so playback overlays survive detection_events
# compression/retention.
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
    stage: str = "detecting"  # detecting | deriving
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
    period_ms: Optional[tuple[int, int]] = None,
) -> Job:
    """Register + enqueue a job. Idempotent on `idempotency_key`.

    The caller (a Workflow DevKit step) runs at-least-once: a retry after a
    lost HTTP response re-POSTs the same request. If a job — non-terminal OR
    terminal — already exists for the key, return it instead of enqueuing a
    duplicate full detection run; the caller then polls the original job as
    usual. Check + registration happen under _lock so two concurrent submits
    with the same key can never both enqueue.

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
                "period_ms": period_ms,
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
                period_ms=params.get("period_ms"),
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
    period_ms: Optional[tuple[int, int]] = None,
) -> dict:
    """detect -> derive -> (COPY to DB). Returns AnalysisResult dict.

    The DB write happens AFTER derivation because derivation stamps track_no on
    the boxes the timeline accepted, and the stored rows carry that attribution.
    It writes `detections` — the whole detector output — and db.replace_detections
    keeps the teacher-class rows from it, attributed or not.
    """
    cb: ProgressCb = progress_cb or (lambda stage, frac: None)

    cb("detecting", 0.0)
    stage_start = time.perf_counter()
    # Resolve an allowlisted object-store URL to a local temp (so a remote GPU
    # worker can fetch the video itself); deleted in the finally either way.
    local_path, is_temp = detector.resolve_video_source(video_path)
    try:
        meta, detections = detector.detect_video(
            local_path,
            sample_fps=sample_fps,
            progress_cb=lambda f: cb("detecting", f * 0.9),
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

        cb("deriving", 0.9)
        stage_start = time.perf_counter()
        result = derive_result(meta, detections, zones, period_ms=period_ms)
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
                db.replace_detections(video_id, detections, run_tokens=run_tokens)
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
        "pipeline stage timings for video %s: detect_s=%.2f derive_s=%.2f",
        video_id,
        detect_s,
        derive_s,
    )
    return result


def _track_overlay(dets: list[Detection]) -> dict:
    """RDP-simplified center polyline + bbox keyframes for the teacher track."""
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


def _movement(dets: list[Detection]) -> float:
    """How much of the room she covered: spatial RANGE of her bbox centers.

    Range, not path length: per-frame bbox jitter accumulates a large fake path
    for someone standing still, while their spatial extent stays near zero.
    """
    if not dets:
        return 0.0
    xs = [d.bbox["x"] + d.bbox["w"] / 2.0 for d in dets]
    ys = [d.bbox["y"] + d.bbox["h"] / 2.0 for d in dets]
    return round(min(1.0, max(max(xs) - min(xs), max(ys) - min(ys))), 4)


def derive_result(
    meta: VideoMeta,
    detections: list[Detection],
    zones: list[dict],
    actions_available: bool = True,
    propose_zones: Optional[bool] = None,
    period_ms: Optional[tuple[int, int]] = None,
) -> dict:
    """Teacher timeline + events + analytics. Shared by analyze & rederive.

    `period_ms` is the scheduled period as (start, end) offsets into the
    video, when the caller knows it; attribution's primary rule needs it.

    Detection.track_no is rewritten IN PLACE (app/teacher.py stamps her
    accepted detections and clears everyone else's), because run_pipeline
    persists `detections` afterwards and only her rows are stored.
    """
    sampled_frames = len({d.video_ts_ms for d in detections})

    # Place any missing board/door zone from THIS pass's own detections.
    #
    # This used to be two separate HTTP endpoints, each re-scanning the whole
    # video before analysis started. That was wasteful — iter_frames grabs every
    # frame whatever the sample rate, so a zone scan costs about what the
    # analysis costs — and on a long lesson it was worse than wasteful: the
    # calls ran past RunPod's proxy timeout, came back 524, and the lesson was
    # analysed with no zones at all, reporting board time as null. A 37-minute
    # test lost ~4 minutes to two scans that returned nothing.
    #
    # The detections are already here, and propose_zone is a pure function over
    # them. Proposing BEFORE gating matters twice: gate_static would drop the
    # very out-of-zone boxes a first proposal must see, and adopting the zone
    # now means the first analysis already reports board time rather than
    # needing a second rederive pass to pick it up.
    proposed: list[dict] = []
    if propose_zones is None:
        propose_zones = actions_available
    if propose_zones:  # a fresh detector pass: screen/door boxes are in hand
        for kind in ("board", "door"):
            if any(z.get("kind") == kind for z in zones):
                continue
            p = zones_mod.propose_zone(detections, kind, frames_seen=sampled_frames)
            if p["polygon"] and p["confidence"] >= zones_mod.AUTO_ACCEPT[kind]:
                proposed.append({"kind": kind, **p})
                zones = [*zones, {"kind": kind, "polygon": p["polygon"]}]

    # Static classes are held to their configured zone, so a poster that reads
    # as a screen for a few frames cannot move the board. The teacher is never
    # gated: she has the run of the room.
    detections = zones_mod.gate_static(detections, zones)
    track = teacher_mod.build_teacher_track(detections, meta.duration_ms)

    # Phase 3: from the tracker's segments to PEOPLE, and from people to the
    # one this lesson assesses. The tracker's interim primary (its biggest
    # segment) is replaced by attribution's choice; when attribution cannot
    # decide, the biggest person stands in as track 1 so the page has a
    # timeline to show, and the low confidence keeps the numbers withheld.
    attribution = None
    people: list["attribution_mod.Person"] = []
    if track.found:
        attribution = attribution_mod.attribute(track.segments, meta.duration_ms, period_ms)
        people = list(attribution.persons)
        chosen = attribution.chosen or (
            max(people, key=lambda p: len(p.detections)) if people else None
        )
        # Re-stamp track_no by PERSON: the chosen one is 1, the rest follow in
        # order of first appearance; boxes belonging to nobody stay None.
        for d in detections:
            if d.cls == teacher_mod.CLASS_TEACHER:
                d.track_no = None
        ordered = ([chosen] if chosen else []) + sorted(
            (p for p in people if p is not chosen), key=lambda p: p.first_ms
        )
        for n, person in enumerate(ordered, start=teacher_mod.TEACHER_TRACK_NO):
            person.track_no = n
            for d in person.detections:
                d.track_no = n
        if chosen is not None:
            track = track.retarget(chosen.detections, sampled_frames)
            if len(people) > 1:
                # "Rejected as implausible jumps" was the single chain's count
                # of instants it could not reach. With several people it means
                # nothing: an instant the teacher is absent from is somebody
                # else's, not a jump. Replace it with the one number that still
                # has that meaning — teacher boxes attributed to NOBODY.
                nobody = {
                    d.video_ts_ms for d in detections
                    if d.cls == teacher_mod.CLASS_TEACHER and d.track_no is None
                } - {d.video_ts_ms for d in detections if d.track_no is not None}
                track.rejected_jumps = len(nobody)
                track.notes = [n for n in track.notes if "implausible jumps" not in n]
                if nobody:
                    track.notes.append(
                        f"{len(nobody)} instant(s) offered a teacher box that was "
                        "attributed to nobody (too brief to be a person, or a "
                        "duplicate)."
                    )
        people = ordered
    teacher_dets = track.detections

    roles_map: dict[int, tuple[str, Optional[float]]] = {}
    dets_by_track: dict[int, list[Detection]] = {}
    if track.found:
        roles_map[teacher_mod.TEACHER_TRACK_NO] = ("teacher", track.confidence)
        dets_by_track[teacher_mod.TEACHER_TRACK_NO] = teacher_dets
        # The other adults. events.derive only ever reads the "teacher" entry,
        # so these change no KPI; they are the candidates attribution chose
        # between and the people a review queue would show.
        for person in people[1:]:
            roles_map[person.track_no] = ("adult", None)
            dets_by_track[person.track_no] = person.detections

    # `detections` (not just hers) so the action classes reach the KPI layer —
    # pointing/writing are separate classes, and only her boxes are persisted,
    # so this is the one point where they are still available.
    #
    # actions_available says whether this list COULD contain them. /rederive
    # replays teacher-only rows out of detection_events, where an action box has
    # never been stored, so its empty result means "not looked at", not "did not
    # happen" — passing the list anyway would silently record a confident 0.
    # The flag cannot be inferred from the data: no action rows is exactly what
    # both cases look like.
    events, analytics = events_mod.derive(
        dets_by_track,
        roles_map,
        meta.duration_ms,
        zones,
        all_detections=detections if actions_available else None,
    )
    analytics["data_quality"] = quality_report(
        track, sampled_frames, meta.duration_ms,
        attribution=attribution.as_report() if attribution else None,
    )

    tracks = []
    if track.found:
        tracks.append(
            {
                "track_no": teacher_mod.TEACHER_TRACK_NO,
                "role": "teacher",
                "role_confidence": track.confidence,
                "first_ms": track.first_ms,
                "last_ms": track.last_ms,
                "meta": {
                    "movement": _movement(teacher_dets),
                    "detections": len(teacher_dets),
                    "coverage": track.coverage,
                    "mean_conf": track.mean_conf,
                    "overlay": _track_overlay(teacher_dets),
                    **_person_analytics(teacher_dets, zones),
                },
            }
        )
        # Same shape for each other adult, overlay included, so a reviewer can
        # be shown who else was in the room. role_confidence is None: nothing
        # has judged these yet, and a number here would read as a judgement.
        for person in people[1:]:
            pdets = person.detections
            tracks.append(
                {
                    "track_no": person.track_no,
                    "role": "adult",
                    "role_confidence": None,
                    "first_ms": person.first_ms,
                    "last_ms": person.last_ms,
                    "meta": {
                        "movement": _movement(pdets),
                        "detections": len(pdets),
                        "coverage": round(len(pdets) / max(sampled_frames, 1), 4),
                        "mean_conf": round(sum(d.conf for d in pdets) / max(len(pdets), 1), 4),
                        "overlay": _track_overlay(pdets),
                        **_person_analytics(pdets, zones),
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
            "proposed_zones": proposed,
        }
    )
    return result.model_dump()


def _person_analytics(dets: list[Detection], zones: list[dict]) -> dict:
    """Presence and board time for ONE tracked person, by the teacher's own
    rules (heuristics.py), so an adult who is not this lesson's teacher can
    still be credited to her own period by the API — the period-2 teacher's
    last minutes live in the period-3 file."""
    if not dets:
        return {}
    sdets = sorted(dets, key=lambda d: d.video_ts_ms)
    doors = [z["polygon"] for z in zones if z.get("kind") == "door"]
    presence, _ = heuristics_mod.classify_presence(
        heuristics_mod.presence_intervals([d.video_ts_ms for d in sdets]), sdets, doors, 0
    )
    board = next((z["polygon"] for z in zones if z.get("kind") == "board"), None)
    board_iv = (
        heuristics_mod.intervals_from_samples(
            [(d.video_ts_ms, heuristics_mod.board_condition(d, board)) for d in sdets]
        )
        if board is not None
        else []
    )
    return {
        "presence_intervals": presence,
        "present_ms": sum(end - start for start, end in presence),
        "board_intervals": board_iv if board is not None else None,
        "board_ms": sum(end - start for start, end in board_iv) if board is not None else None,
    }


def quality_report(
    track: "teacher_mod.TeacherTrack",
    sampled_frames: int,
    duration_ms: int,
    attribution: Optional[dict] = None,
) -> dict:
    """The trust report, with the teacher track's own notes folded in."""
    from app import quality

    report = quality.assess(
        track.detections,
        sampled_frames,
        duration_ms,
        teacher_confidence=track.confidence,
        mean_conf=track.mean_conf if track.found else None,
        attribution=attribution,
        multiple_adults=track.multiple_adults,
        co_presence_ms=track.co_presence_ms,
        max_simultaneous=track.max_simultaneous,
    )
    report["notes"] = [*track.notes, *report["notes"]]
    return report
