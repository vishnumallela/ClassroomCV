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
from app.config import get_settings
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


def submit_rederive(video_id: str, zones: list[dict]) -> Job:
    """Queue a re-derive and return immediately with a job to poll.

    Same shape as submit(): the caller polls /jobs/{id}. A re-derive of a long
    lesson takes minutes, and holding an HTTP request open for that does not
    work -- the client's socket idles out (Bun kills a quiet connection at
    300s, which an AbortSignal cannot extend), the caller retries, and the work
    is redone from scratch while each completed derive is thrown away. Sharing
    the analysis worker also keeps the one-job-at-a-time guarantee.
    """
    job = Job(id=str(uuid.uuid4()), video_id=video_id)
    with _lock:
        _jobs[job.id] = job
    _queue.put((job, {"kind": "rederive", "zones": zones}))
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
            if params.get("kind") == "rederive":
                job.stage = "deriving"
                job.progress = 0.5
                result = asyncio.run(rederive_video(job.video_id, params["zones"]))
                job.result = result
                job.progress = 1.0
                job.status = "done"
                continue
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


async def rederive_video(video_id: str, zones: list[dict]) -> dict:
    """Rebuild identities/roles/events from stored detections. No YOLO re-run.

    Shared by the /rederive endpoint and the queued rederive job, so both paths
    derive identically and only differ in how the caller waits.
    """
    detections = await db.fetch_detections(video_id)
    info = await db.fetch_video_info(video_id) or {}
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
            track_hists = await db.fetch_track_hists(video_id)
        except Exception:
            track_hists = {}
        try:
            track_embeds = await db.fetch_track_embeds(video_id)
        except Exception:
            track_embeds = {}
        identities = remerge_from_raw(detections, track_hists, track_embeds)
    # Off the event loop when called from the async endpoint; harmless in the
    # worker thread. derive_result is minutes of CPU on a long lesson.
    result = await asyncio.to_thread(
        derive_result,
        meta,
        detections,
        identities,
        zones,
        track_embeds,
        info.get("file_path"),
    )
    if detections:
        await db.replace_detections(
            video_id, detections, track_hists=track_hists, track_embeds=track_embeds
        )
    return result


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
    # Resolve an allowlisted object-store URL to a local file so a remote GPU
    # worker can fetch the video itself. The copy is NOT deleted here: the
    # vision-model layers in derive read frames from the same video, and the
    # zone endpoints want it too, so the shared cache keeps one copy and evicts
    # it when another video needs the slot.
    local_path = detector.resolve_video_cached(video_path)
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

    # Teacher selection, layered on top of the geometric ranker:
    #  - VLM VERIFY (vlm_verify_teacher on): the vision model is the authority on
    #    WHO. locate_teacher points at the teacher (point-in-bbox) and votes; the
    #    winner OVERRIDES the geometric pick, and if the model confirms NO teacher
    #    the unverified geometric pick is demoted — this is what pulls "teacher"
    #    off a student the ranker wrongly crowned.
    #  - LEGACY FALLBACK (off): the VLM runs only when the ranker gave up
    #    (all-unknown), e.g. a teacher who sits the whole lesson.
    # The stitch below runs on whichever teacher survives (un-gated) so a
    # crouch/lean id-steal is still reclaimed.
    settings = get_settings()
    # One vision-model budget for the whole derive. /rederive is synchronous
    # behind a 255s transport limit and the verify/veto/trim/gap-fill work grows
    # with how badly the teacher fragments, so the cap is what keeps a long,
    # messy lesson from running past it. Running dry degrades to the geometric
    # result rather than failing.
    # Scale the allowance with the lesson: the verify/veto/trim stages are
    # per-claim and a long video fragments into many more claims. Earlier stages
    # get only a share of it (see raise_budget) so they cannot spend everything
    # before gap-filling, which is where the coverage actually comes from.
    vlm_budget = min(
        settings.vlm_max_calls_cap,
        max(settings.vlm_max_calls, round(meta.duration_ms / 60_000 * 8)),
    )
    vlm_teacher.begin_derive(round(vlm_budget * settings.vlm_early_stage_share))
    if settings.vlm_verify_teacher and video_path is not None:
        vlm = vlm_teacher.locate_teacher(video_path, dets_by_track, meta.duration_ms)
        if vlm.track is not None and vlm.track in dets_by_track:
            # The model CONFIDENTLY pointed at a track -> it is the authority on
            # who. Confirm the geometric pick or OVERRIDE onto whoever it points
            # at (demoting a wrongly-crowned student).
            if teacher_no is not None and teacher_no != vlm.track:
                roles_map[teacher_no] = ("student", None)  # demote the wrong crown
                logger.info(
                    "VLM verify OVERRIDE: geometric teacher %s -> VLM %s",
                    teacher_no,
                    vlm.track,
                )
            teacher_no = vlm.track
            roles_map[teacher_no] = ("teacher", vlm.confidence)
            # Fold every fragment the model pooled as her into that identity, so
            # the chain starts owning her whole timeline instead of rediscovering
            # it fragment by fragment through a greedy walk that can take a wrong
            # fork. A fragment pooled in error is not permanent: the chain only
            # keeps what it can claim and evicts the rest back to a student track.
            pooled = set(vlm.fragments or ())
            moved = 0
            for d in detections:
                if d.raw_track_id in pooled and d.track_no != teacher_no:
                    d.track_no = teacher_no
                    moved += 1
            if moved:
                dets_by_track = {}
                for d in detections:
                    if d.track_no is not None:
                        dets_by_track.setdefault(d.track_no, []).append(d)
                for dets in dets_by_track.values():
                    dets.sort(key=lambda d: d.video_ts_ms)
                roles_map = {
                    no: role for no, role in roles_map.items() if no in dets_by_track
                }
                roles_map[teacher_no] = ("teacher", vlm.confidence)
                logger.info(
                    "VLM pooled %d det(s) from fragments %s into teacher %d",
                    moved,
                    sorted(pooled),
                    teacher_no,
                )
        elif (
            teacher_no is not None
            and vlm.answered >= settings.vlm_reject_min_answered
            and vlm.votes.get(teacher_no, 0) == 0
        ):
            # REJECT: the VLM answered enough frames and NEVER once pointed at the
            # geometric pick (and crowned no one of its own) -> she isn't the
            # teacher. Demote her so a student the ranker wrongly crowned loses the
            # label. Guarded on answered, so a network failure (answered 0) can
            # never un-label a correct teacher — that guard IS the fix for the
            # demote-on-failure bug (an HTTP 400 once demoted the real teacher).
            logger.info(
                "VLM verify REJECT: geometric teacher %s got 0/%d VLM votes -> no teacher",
                teacher_no,
                vlm.answered,
            )
            roles_map[teacher_no] = ("student", None)
            teacher_no = None
        # else: the VLM pointed at the geometric pick at least once, or answered
        # too few frames, or failed outright (answered 0). KEEP the geometric pick
        # — an unconfirmed pick beats throwing away a correct one.
    elif teacher_no is None and video_path is not None:
        vlm = vlm_teacher.identify_teacher(video_path, dets_by_track, meta.duration_ms)
        if vlm is not None and vlm[0] in dets_by_track:
            teacher_no, conf, _votes = vlm
            roles_map[teacher_no] = ("teacher", conf)

    if teacher_no is not None:
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
            # VLM-veto REACH-ACROSS claims: the stitcher can wrongly fold a student
            # fragment that ends where the teacher's tracked chain begins (a
            # backward claim passing the height-rise test) into the teacher — the
            # chimera seen as "student labelled teacher until the real teacher
            # comes". A claim whose fragment came from ANOTHER identity must
            # actually be the teacher across its span; if the VLM answered frames
            # there and never pointed inside it, drop the claim so the student keeps
            # her identity. Fragments already in the teacher's own identity (incl.
            # the seed) are no-ops here, so they never trigger a VLM call.
            if settings.vlm_verify_teacher and video_path is not None:
                surviving = []
                for c in claims:
                    if (
                        c.fragment.host_track_no != teacher_no
                        and vlm_teacher.vlm_supports_span(video_path, c.dets) is False
                    ):
                        logger.info(
                            "VLM VETO stitch claim: raw %s from track %s (%d dets) -> not the teacher",
                            c.fragment.raw_id,
                            c.fragment.host_track_no,
                            len(c.dets),
                        )
                        continue
                    surviving.append(c)
                claims = surviving

            # TRIM a claim whose fragment changes PERSON mid-way. A tracker id
            # can slide off the teacher onto a pupil she walks past, and because
            # the chain works at whole-fragment granularity the wrong person then
            # wears the teacher label for as long as the id stays on them. The
            # veto above cannot catch it: it only inspects claims from ANOTHER
            # identity, and it judges a span as a whole, so a fragment that is
            # genuinely her for most of its life passes. find_switch_index is a
            # free geometric screen (no API calls) that proposes a split; the VLM
            # only arbitrates the few fragments it flags, and a claim is trimmed
            # only when one side is confirmed hers and the other is refused.
            dropped: list[list[Detection]] = []
            if settings.vlm_verify_teacher and video_path is not None:
                kept: list[teacher_chain.Claim] = []
                for c in claims:
                    cut = teacher_chain.find_switch_index(c.dets)
                    if cut is None:
                        kept.append(c)
                        continue
                    head, tail = c.dets[:cut], c.dets[cut:]
                    head_ok = vlm_teacher.vlm_supports_span(video_path, head, n_frames=4)
                    tail_ok = vlm_teacher.vlm_supports_span(video_path, tail, n_frames=4)
                    if head_ok is True and tail_ok is False:
                        kept.append(
                            teacher_chain.Claim(c.fragment, c.from_idx, c.from_idx + cut)
                        )
                        dropped.append(tail)
                    elif tail_ok is True and head_ok is False:
                        kept.append(
                            teacher_chain.Claim(c.fragment, c.from_idx + cut, c.to_idx)
                        )
                        dropped.append(head)
                    else:
                        kept.append(c)  # both sides her (she walked), or unclear
                        continue
                    logger.info(
                        "VLM TRIM stitch claim: raw %s switches person at %.1fs "
                        "(head %s, tail %s) -> %d dets evicted",
                        c.fragment.raw_id,
                        c.dets[cut].video_ts_ms / 1000 if cut < len(c.dets) else -1,
                        head_ok,
                        tail_ok,
                        len(dropped[-1]),
                    )
                claims = kept

            # BOUNDED GAP-FILL: a trim leaves the teacher unlabelled across the
            # vacated span even though she IS tracked there under another id.
            # Offer only fragments INSIDE that window (never a re-chain: a full
            # re-derive is greedy and was measured trading a 20s gap for ~180s of
            # lost timeline), rank them, and claim the first the VLM confirms.
            filled: list[list[Detection]] = []
            if settings.vlm_verify_teacher and video_path is not None:
                # Holes in the teacher's timeline: the chain builds a PATH, so
                # wherever it stepped over her (or a trim vacated a span) she is
                # tracked under some other id and shows up as a student. The
                # chain already reclaimed her OWN skipped fragments for free;
                # what is left needs an identity judgement, so each hole offers
                # a couple of candidates and the VLM decides. Bounded by hole
                # length and count so a fragmented lesson cannot run away.
                blocked = {
                    (d.raw_track_id, d.video_ts_ms) for lst in dropped for d in lst
                }
                claimed = sorted(d.video_ts_ms for c in claims for d in c.dets)
                holes = [
                    (a, b)
                    for a, b in zip(claimed, claimed[1:])
                    if b - a >= teacher_chain.GAP_FILL_MIN_MS
                ]
                if claimed and meta.duration_ms - claimed[-1] >= teacher_chain.GAP_FILL_MIN_MS:
                    holes.append((claimed[-1], meta.duration_ms))  # she may teach on past the chain
                if claimed and claimed[0] >= teacher_chain.GAP_FILL_MIN_MS:
                    # ...and she is usually teaching BEFORE the chain picks her
                    # up. The chain seeds on her most teacher-like fragment and
                    # its backward walk stops at MAX_CHAIN_GAP_MS, so on a long
                    # lesson the first minutes stay unclaimed and she shows as a
                    # student from the opening frame. Short clips never showed
                    # this because the seed's backward walk reaches their start.
                    holes.append((0, claimed[0]))
                # An hour-long lesson has many more holes than the five-minute
                # clip this cap was set on, so it scales with the lesson; the
                # per-derive call budget is what actually bounds the cost.
                max_holes = min(
                    teacher_chain.GAP_FILL_MAX_HOLES_CAP,
                    max(
                        teacher_chain.GAP_FILL_MAX_HOLES,
                        round(meta.duration_ms / 60_000 / 3),
                    ),
                )
                # Gap-fill is the highest-value use of the remaining calls, so
                # lift the cap back to the full budget for it.
                vlm_teacher.raise_budget(vlm_budget)
                claimed_ts = {d.video_ts_ms for c in claims for d in c.dets}
                # Biggest holes first: the cap and the call budget both bite, and
                # a 7-minute hole is worth far more than a 4-second one. (In list
                # order the head window, appended last, could be dropped entirely.)
                holes.sort(key=lambda h: h[1] - h[0], reverse=True)
                # What "teacher-sized" means at each depth in this room, fit on
                # the timeline the chain is confident about. Anchoring needs it
                # to tell where a body stops being hers: without it a claim can
                # only speak for a fixed slice of clock, whose edges fall
                # mid-stride and rename her for no physical reason.
                anchor_model = (
                    teacher_chain.fit_height_model([d for c in claims for d in c.dets])
                    if claims
                    else None
                )
                for start_ms, end_ms in holes[:max_holes]:
                    if end_ms - start_ms >= teacher_chain.LONG_HOLE_MS:
                        # A long hole mixes "she left the room" with "she is here
                        # under a different id", sometimes several ids, so no
                        # single candidate explains it. Ask per sub-window.
                        got = vlm_teacher.anchor_teacher_in_window(
                            video_path,
                            dets_by_track,
                            teacher_no,
                            start_ms,
                            end_ms,
                            claimed_ts,
                            height_model=anchor_model,
                        )
                        if got:
                            filled.append(got)
                            claimed_ts.update(d.video_ts_ms for d in got)
                        continue
                    before = [
                        d for c in claims for d in c.dets if d.video_ts_ms <= start_ms
                    ]
                    if not before:
                        continue
                    tail = max(before, key=lambda d: d.video_ts_ms).bbox
                    anchor = (tail["x"] + tail["w"] / 2.0, tail["y"] + tail["h"] / 2.0)
                    for cand in teacher_chain.gap_fill_candidates(
                        dets_by_track, teacher_no, start_ms, end_ms, anchor, blocked
                    ):
                        if (
                            vlm_teacher.vlm_supports_span(video_path, cand, n_frames=4)
                            is True
                        ):
                            filled.append(cand)
                            blocked |= {(d.raw_track_id, d.video_ts_ms) for d in cand}
                            logger.info(
                                "VLM GAP-FILL: raw %s claimed for %.1f-%.1fs (%d dets)",
                                cand[0].raw_track_id,
                                start_ms / 1000,
                                end_ms / 1000,
                                len(cand),
                            )
                            break

            # RECLAIM BEFORE EVICTING. Every teacher-identity range the chain
            # did not claim goes to a fresh student track, and nothing later can
            # undo that: both repair stages skip detections already carrying her
            # number, and these still do when those stages run. So the vision
            # model looks straight at her, finds no eligible box, claims
            # nothing, and only afterwards is she renamed. Measured on a 37-min
            # lesson: 531 detections of her own body, sitting inside the two
            # largest stretches the dashboard painted "out of the room".
            # Refusing an eviction is strictly additive — the detections already
            # carry her number, so keeping one takes nothing from anyone else.
            teacher_ts = {d.video_ts_ms for c in claims for d in c.dets}
            teacher_ts |= {d.video_ts_ms for cand in filled for d in cand}
            reclaimed = teacher_chain.reclaimable_evictions(
                evictions,
                claims,
                embeds_by_raw,
                teacher_chain.fit_height_model([d for c in claims for d in c.dets]),
                teacher_ts,
            )
            next_no = max(dets_by_track, default=0) + 1
            for dets_out in dropped:
                for d in dets_out:
                    d.track_no = next_no
                roles_map[next_no] = ("student", None)
                next_no += 1
            for frag, lo, hi in evictions:
                out = [
                    d
                    for d in frag.dets[lo:hi]
                    if (frag.raw_id, d.video_ts_ms) not in reclaimed
                ]
                if not out:
                    continue  # wholly reclaimed: already carries teacher_no
                for d in out:
                    d.track_no = next_no
                roles_map[next_no] = ("student", None)
                next_no += 1
            for c in claims:
                for d in c.dets:
                    d.track_no = teacher_no
            for cand in filled:  # gap-fill runs last: it wins over the eviction
                for d in cand:
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
                "teacher chain: %d claim(s) %s, %d evicted range(s) from track %d, "
                "%d det(s) reclaimed from eviction",
                len(claims),
                [(c.fragment.raw_id, c.fragment.dets[c.from_idx].video_ts_ms) for c in claims],
                len(evictions),
                teacher_no,
                len(reclaimed),
            )
            features = roles.compute_features(
                dets_by_track,
                meta.duration_ms,
                board_polygon=board_polygon,
                raw_ids_by_track=raw_ids_by_track,
            )

        # VLM ENTRY-BACKFILL: the chain seeds on her MOBILE (teaching) phase and
        # skips seated/static fragments, so a teacher who enters and sits before
        # teaching stays labelled student until the chain kicks in. Sample backward
        # from her first detection, ask the VLM where she is, and reclaim the raw
        # fragments she is actually in. OFF by default (vlm_backfill_entry): on
        # crowded footage her entry is hyper-fragmented and occluded, so the trace
        # over-claims foreground students — a robust version needs a per-fragment
        # vlm_supports_span verification pass. Runs whether or not the stitch fired.
        if settings.vlm_backfill_entry and video_path is not None:
            backfill = vlm_teacher.backfill_teacher_entry(
                video_path, dets_by_track, teacher_no
            )
            if backfill:
                for d in backfill:
                    d.track_no = teacher_no
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
                features = roles.compute_features(
                    dets_by_track,
                    meta.duration_ms,
                    board_polygon=board_polygon,
                    raw_ids_by_track=raw_ids_by_track,
                )

    # Every vision-model helper is done with the video by here; drop the shared
    # copy (a temp download on a remote worker) before the analytics pass.
    vlm_teacher.release_media()

    frame_aspect = (meta.width / meta.height) if meta.height else 1.0
    events, analytics = events_mod.derive(
        dets_by_track, roles_map, meta.duration_ms, zones, frame_aspect
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
