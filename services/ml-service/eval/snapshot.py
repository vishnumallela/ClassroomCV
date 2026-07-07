"""Export a video's raw detections + info to an offline fixture.

Usage (from services/ml-service):
    uv run python eval/snapshot.py <video_id>

Writes eval/fixtures/<video_id>.dets.jsonl.gz (one detection per line, plus a
leading header line with video info). The fixture makes regression runs
independent of the live database, and preserves the raw state that a
reanalyze would otherwise overwrite.
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


async def main(video_id: str) -> int:
    detections = await db.fetch_detections(video_id)
    if not detections:
        print(f"no detections for {video_id}")
        return 1
    info = await db.fetch_video_info(video_id) or {}
    conn = await db._connect()
    try:
        zone_rows = await conn.fetch(
            "select kind, polygon from zones where video_id = $1::uuid order by created_at",
            video_id,
        )
    finally:
        await conn.close()
    zones = []
    for r in zone_rows:
        polygon = r["polygon"]
        if isinstance(polygon, str):
            polygon = json.loads(polygon)
        zones.append({"kind": r["kind"], "polygon": polygon})

    out_dir = EVAL_DIR / "fixtures"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{video_id}.dets.jsonl.gz"
    with gzip.open(out, "wt") as f:
        f.write(json.dumps({"video_id": video_id, "info": dict(info), "zones": zones}) + "\n")
        for d in detections:
            f.write(
                json.dumps(
                    {
                        "video_ts_ms": d.video_ts_ms,
                        "raw_track_id": d.raw_track_id,
                        "track_no": d.track_no,
                        "bbox": d.bbox,
                        "conf": d.conf,
                        "standing": d.standing,
                        "back_to_camera": d.back_to_camera,
                    }
                )
                + "\n"
            )
    size_kb = out.stat().st_size // 1024
    print(f"wrote {out.name}: {len(detections)} detections, {len(zones)} zones, {size_kb} KB")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
