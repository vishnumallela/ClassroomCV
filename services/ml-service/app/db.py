"""asyncpg access to TimescaleDB.

- replace_detections: DELETE prior rows for the video, then bulk-insert via
  COPY (copy_records_to_table) in batches of ~5000 — both inside ONE
  transaction so the swap is atomic: a mid-write failure rolls back to the
  previous full set, and concurrent readers (/rederive) see either the old
  complete rows or the new complete rows, never a partial prefix. bbox/meta
  are passed as JSON strings (asyncpg encodes str for jsonb).
- fetch_detections: read rows back as Detection dataclasses for /rederive.
- fetch_video_info: best-effort read of the dashboard's videos row (duration
  etc.) for the /rederive response; returns None when unavailable.

ONLY THE TEACHER IS STORED: her `teacher` boxes, and her `pointing` and
`writing` boxes (her own gestures at the board, at action_conf) since
2026-09-05 so the lesson-start signal survives a re-derive. Every persisted
row is therefore a box on the teacher, so "students are never displayed" is a
property of the data rather than of the renderer — there is no student box in
the database to leak into an overlay by mistake. The board and door live in `zones` (one polygon
each, not one row per frame) and their derived numbers in `video_analytics`, so
nothing is lost.

Since migration 0014 that includes teacher boxes NO TRACK CLAIMED (track_no
NULL). The distinction matters and is not a privacy change: the filter is on
the class, and the class is still teacher-only. What changed is that the
timeline's opinion no longer decides what survives. app/teacher.py follows one
chain and drops everything else; on a lesson with two adults in the room, the
rows it dropped were the only evidence that the chain was blending two people,
and they were dropped before the database ever saw them. Keeping them makes a
changed attribution rule a /rederive over stored rows rather than another paid
detector pass — which is what the rule needs, since it does not exist yet.
"""

from __future__ import annotations

import json
from typing import Optional

import asyncpg

from app.config import get_settings
from app.models import CLASS_POINTING, CLASS_TEACHER, CLASS_WRITING, Detection

COPY_COLUMNS = ["video_ts_ms", "video_id", "track_no", "bbox", "confidence", "meta"]
COPY_BATCH_SIZE = 5_000


class VideoDeletedError(Exception):
    """The videos row vanished (video deleted) while analysis was in flight.

    detection_events has NO foreign key to videos, so a write racing a
    DELETE /api/videos/{id} would otherwise commit permanently orphaned rows.
    jobs.py treats this as a graceful abort (job failed, worker stays alive).
    """


class StaleRunError(Exception):
    """The analysis run that produced these detections has been superseded.

    The dashboard persists a fence token in videos.workflow_run_id (a fresh
    attempt id on every reanalyze, the workflow run id afterwards). A YOLO
    job started by an older run must not rewrite detection_events after a
    newer run/rederive took ownership — that is exactly how a video ends up
    'done' with hundreds of thousands of detections but zero tracks (the
    stale job's rows without the stale job's derived data). jobs.py treats
    this as a graceful abort, like VideoDeletedError.
    """


async def _connect(dsn: Optional[str] = None) -> asyncpg.Connection:
    return await asyncpg.connect(dsn or get_settings().database_url)


async def replace_detections(
    video_id: str,
    detections: list[Detection],
    dsn: Optional[str] = None,
    batch_size: int = COPY_BATCH_SIZE,
    run_tokens: Optional[list[str]] = None,
) -> int:
    """Delete prior detection_events for video_id, COPY the new ones. Returns row count.

    DELETE + all COPY batches run in a single transaction: any mid-write
    failure rolls the whole swap back (previous rows preserved, no partial
    prefix committed for /rederive to trust).

    Orphan-write fence: detection_events has no FK to videos, so before
    writing we verify the videos row still exists — FOR SHARE holds a lock on
    it until commit, so a concurrent DELETE /api/videos/{id} cannot slip
    between the check and the COPY. If the video is already gone, raise
    VideoDeletedError (rolls back, writes nothing) instead of committing rows
    that would be permanently orphaned.

    Stale-run fence: when `run_tokens` is provided (the dashboard workflow
    passes its reanalyze attempt id + workflow run id), the same locked read
    also verifies videos.workflow_run_id still names this run. A NULL stored
    value is accepted (fresh upload before the route persists the run id —
    no competing run can exist then). On mismatch raise StaleRunError inside
    the transaction: a superseded YOLO job rolls back instead of silently
    replacing the current run's detections. Callers without a token (tests,
    direct API use, synchronous /rederive) skip the token check.
    """
    conn = await _connect(dsn)
    try:
        # Every TEACHER box the tracker was allowed to consider, attributed or
        # not. Two details of this filter are load-bearing:
        #
        # It tests the CLASS, not track_no. Before 0014 those selected the same
        # rows, because only teacher boxes were ever stamped — so simply
        # loosening the null check would have silently started persisting
        # screen, door, pointing and writing rows too, quietly ending
        # "teacher-only storage" as a property of the data.
        #
        # It applies teacher_conf, the same threshold app/teacher.py uses to
        # decide what is a candidate at all. The detector emits down to a much
        # lower floor so one pass can serve every consumer; storing that floor
        # would be storing noise no attribution rule would ever consult. Both
        # read the setting rather than a local constant, so there is one
        # definition of "a teacher box worth considering", not two.
        settings = get_settings()
        threshold = settings.teacher_conf
        action_threshold = settings.action_conf
        records = [
            (
                d.video_ts_ms,
                video_id,
                d.track_no,
                json.dumps(d.bbox),
                float(d.conf),
                json.dumps({"cls": int(d.cls), "app": d.app} if d.app is not None else {"cls": int(d.cls)}),
            )
            for d in detections
            if (d.cls == CLASS_TEACHER and d.conf >= threshold)
            or (d.cls in (CLASS_POINTING, CLASS_WRITING) and d.conf >= action_threshold)
        ]
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT workflow_run_id FROM videos WHERE id = $1 FOR SHARE",
                video_id,
            )
            if row is None:
                raise VideoDeletedError(
                    f"video {video_id} deleted during analysis"
                )
            if run_tokens:
                stored = row["workflow_run_id"]
                if stored is not None and stored not in run_tokens:
                    raise StaleRunError(
                        f"analysis run superseded for video {video_id}: "
                        f"videos.workflow_run_id={stored!r} is not one of this "
                        f"run's tokens"
                    )
            await conn.execute(
                "DELETE FROM detection_events WHERE video_id = $1", video_id
            )
            for i in range(0, len(records), batch_size):
                await conn.copy_records_to_table(
                    "detection_events",
                    records=records[i : i + batch_size],
                    columns=COPY_COLUMNS,
                )
        return len(records)
    finally:
        await conn.close()


async def fetch_detections(
    video_id: str, dsn: Optional[str] = None
) -> list[Detection]:
    """Read the stored teacher detections back, for /rederive.

    Rows written before this schema carry no 'cls' in meta; they are read as
    teacher detections, which is what they were — the only rows ever stored.

    A NULL track_no (0014) comes back as None, which is exactly what a fresh
    detector pass hands app/teacher.py. That is what makes /rederive able to
    re-run the attribution question rather than merely replay its old answer:
    the losing candidates are in the input, not just the winners. Note the
    consequence for lessons analysed BEFORE 0014 — their stored rows are the
    winners only, so a rederive of one cannot rediscover a second adult and
    will report no co-presence. Those need a fresh run, not a rederive.
    """
    conn = await _connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT video_ts_ms, track_no, bbox, confidence, meta "
            "FROM detection_events WHERE video_id = $1 ORDER BY video_ts_ms",
            video_id,
        )
    finally:
        await conn.close()

    detections: list[Detection] = []
    for r in rows:
        bbox = r["bbox"]
        bbox = json.loads(bbox) if isinstance(bbox, str) else (bbox or {})
        meta = r["meta"]
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
        track_no = r["track_no"]
        detections.append(
            Detection(
                video_ts_ms=int(r["video_ts_ms"]),
                cls=int(meta.get("cls", CLASS_TEACHER)),
                bbox=bbox,
                conf=float(r["confidence"]),
                track_no=None if track_no is None else int(track_no),
                # Rows written before descriptors existed carry none; attribution then
                # links by continuity alone and its confidence says so.
                app=meta.get("app"),
            )
        )
    return detections


async def fetch_video_info(
    video_id: str, dsn: Optional[str] = None
) -> Optional[dict]:
    """duration_ms/fps/width/height from the dashboard's videos table, or None."""
    try:
        conn = await _connect(dsn)
        try:
            row = await conn.fetchrow(
                "SELECT duration_ms, fps, width, height FROM videos WHERE id = $1",
                video_id,
            )
        finally:
            await conn.close()
    except Exception:
        return None
    return dict(row) if row is not None else None
