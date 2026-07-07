"""Restore a video's detection_events from an offline fixture.

Usage (from services/ml-service):
    uv run python eval/restore.py <video_id>

Reverses experiments that overwrote raw detections: loads
eval/fixtures/<video_id>.dets.jsonl.gz and swaps it back into the database
through the same transactional replace path the analyzer uses.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app import db
from app.models import Detection


async def main(video_id: str) -> int:
    fixture = EVAL_DIR / "fixtures" / f"{video_id}.dets.jsonl.gz"
    if not fixture.exists():
        print(f"no fixture at {fixture}")
        return 1

    detections: list[Detection] = []
    with gzip.open(fixture, "rt") as f:
        header = json.loads(f.readline())
        for line in f:
            row = json.loads(line)
            detections.append(
                Detection(
                    video_ts_ms=row["video_ts_ms"],
                    raw_track_id=row["raw_track_id"],
                    bbox=row["bbox"],
                    conf=row["conf"],
                    standing=row["standing"],
                    back_to_camera=row["back_to_camera"],
                    track_no=row["track_no"],
                )
            )

    conn = await db._connect()
    try:
        fence = await conn.fetchval(
            "select workflow_run_id from videos where id = $1::uuid", video_id
        )
    finally:
        await conn.close()

    written = await db.replace_detections(
        video_id, detections, run_tokens=[fence] if fence else None
    )
    print(f"restored {written} detections for {header['video_id']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
