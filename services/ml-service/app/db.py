"""asyncpg access to TimescaleDB.

- replace_detections: DELETE prior rows for the video, then bulk-insert via
  COPY (copy_records_to_table) in batches of ~5000 — both inside ONE
  transaction so the swap is atomic: a mid-write failure rolls back to the
  previous full set, and concurrent readers (/rederive) see either the old
  complete rows or the new complete rows, never a partial prefix. bbox/meta
  are passed as JSON strings (asyncpg encodes str for jsonb). track_no is the POST-merge
  identity number; we also stash raw_track_id inside meta so /rederive can
  reconstruct raw id lists without re-running YOLO.
- fetch_detections: read rows back as Detection dataclasses for /rederive.
- fetch_video_info: best-effort read of the dashboard's videos row (duration
  etc.) for the /rederive response; returns None when unavailable.
"""

from __future__ import annotations

import json
from typing import Optional

import asyncpg

from app.config import get_settings
from app.models import Detection

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
    track_hists: Optional[dict[int, list[float]]] = None,
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
        # Appearance persistence: the median torso histogram of each raw track
        # rides in the meta of that track's FIRST row (one ~960-float payload
        # per raw track, not per detection), so /rederive can merge with the
        # same appearance evidence /analyze had instead of degrading to
        # spatial-only scoring.
        hist_pending = dict(track_hists) if track_hists else {}
        records = []
        for d in detections:
            meta: dict = {
                "standing": bool(d.standing),
                "back_to_camera": bool(d.back_to_camera),
                "raw_track_id": int(d.raw_track_id),
            }
            hist = hist_pending.pop(int(d.raw_track_id), None)
            if hist is not None:
                meta["hist"] = hist
            records.append(
                (
                    d.video_ts_ms,
                    video_id,
                    d.track_no,
                    json.dumps(d.bbox),
                    float(d.conf),
                    json.dumps(meta),
                )
            )
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
    """Read stored detections (track_no is already the merged identity)."""
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
        detections.append(
            Detection(
                video_ts_ms=int(r["video_ts_ms"]),
                raw_track_id=int(meta.get("raw_track_id", r["track_no"])),
                bbox=bbox,
                conf=float(r["confidence"]),
                standing=bool(meta.get("standing", False)),
                back_to_camera=bool(meta.get("back_to_camera", False)),
                track_no=int(r["track_no"]),
            )
        )
    return detections


async def fetch_track_hists(
    video_id: str, dsn: Optional[str] = None
) -> dict[int, list[float]]:
    """Median torso histogram per raw_track_id, from rows that carry one."""
    conn = await _connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT meta FROM detection_events "
            "WHERE video_id = $1 AND meta ? 'hist'",
            video_id,
        )
    finally:
        await conn.close()
    out: dict[int, list[float]] = {}
    for r in rows:
        meta = r["meta"]
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
        if "hist" in meta and "raw_track_id" in meta:
            out[int(meta["raw_track_id"])] = [float(v) for v in meta["hist"]]
    return out


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
