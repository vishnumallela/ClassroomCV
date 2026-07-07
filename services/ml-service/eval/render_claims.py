"""Render claim-annotated frames for visual truth audits.

For each claimed timestamp this extracts the video frame, overlays every
detection box at the nearest sampled ts (colored by track role), draws the
saved zone polygons, and stamps a banner describing the claim under test so
a reviewer can judge claim-vs-pixels directly.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg
import cv2
import numpy as np

DSN = "postgres://postgres:postgres@localhost:5433/classroom"
OUT_WIDTH = 1568

GREEN = (80, 220, 80)
BLUE = (255, 160, 60)
YELLOW = (0, 220, 255)
EMERALD = (140, 200, 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


async def load(video_id: str):
    conn = await asyncpg.connect(DSN)
    try:
        video = await conn.fetchrow("SELECT file_path, duration_ms FROM videos WHERE id=$1", video_id)
        roles = {
            r["track_no"]: r["role"]
            for r in await conn.fetch("SELECT track_no, role FROM tracks WHERE video_id=$1", video_id)
        }
        zones = [
            {"kind": r["kind"], "polygon": json.loads(r["polygon"])}
            for r in await conn.fetch("SELECT kind, polygon FROM zones WHERE video_id=$1", video_id)
        ]
        dets = defaultdict(list)
        for r in await conn.fetch(
            "SELECT video_ts_ms, track_no, bbox, confidence FROM detection_events WHERE video_id=$1",
            video_id,
        ):
            dets[r["video_ts_ms"]].append(
                (r["track_no"], json.loads(r["bbox"]), r["confidence"])
            )
        return video, roles, zones, dets
    finally:
        await conn.close()


def draw(frame, ts_ms, sampled_ts, dets, roles, zones, banner):
    h, w = frame.shape[:2]
    for z in zones:
        pts = np.array([[int(px * w), int(py * h)] for px, py in z["polygon"]], np.int32)
        color = YELLOW if z["kind"] == "board" else EMERALD
        cv2.polylines(frame, [pts], True, color, 4)
        cv2.putText(frame, z["kind"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    n_teacher = n_student = 0
    for track_no, b, conf in dets:
        role = roles.get(track_no, "unknown")
        x0, y0 = int(b["x"] * w), int(b["y"] * h)
        x1, y1 = int((b["x"] + b["w"]) * w), int((b["y"] + b["h"]) * h)
        if role == "teacher":
            n_teacher += 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 6)
            cv2.putText(frame, f"T{track_no}", (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, GREEN, 4)
        else:
            n_student += 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), BLUE, 2)
            cv2.putText(frame, str(track_no), (x0, max(24, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)

    lines = [
        banner,
        f"frame ts={ts_ms}ms  dets@{sampled_ts}ms  teacher_boxes={n_teacher}  student_boxes={n_student}",
    ]
    for i, line in enumerate(lines):
        y = 46 + i * 46
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, BLACK, 8)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3)
    return frame


def main(video_id: str, out_dir: Path, claims: list[tuple[int, str, str]]):
    video, roles, zones, dets = asyncio.run(load(video_id))
    if video is None:
        sys.exit(f"video {video_id} not found")
    sampled = sorted(dets.keys())
    cap = cv2.VideoCapture(video["file_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for ts_ms, name, banner in claims:
        ts_ms = max(0, min(ts_ms, video["duration_ms"] - 100))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ok, frame = cap.read()
        if not ok:
            print(f"SKIP {name}: no frame at {ts_ms}")
            continue
        i = bisect.bisect_left(sampled, ts_ms)
        candidates = [c for c in (i - 1, i) if 0 <= c < len(sampled)]
        nearest = min((sampled[c] for c in candidates), key=lambda s: abs(s - ts_ms))
        frame = draw(frame, ts_ms, nearest, dets.get(nearest, []), roles, zones, banner)
        scale = OUT_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (OUT_WIDTH, int(frame.shape[0] * scale)))
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"wrote {path}")
    cap.release()


if __name__ == "__main__":
    vid = sys.argv[1]
    out = Path(sys.argv[2])
    claims_json = json.loads(Path(sys.argv[3]).read_text())
    main(vid, out, [(c["ts"], c["name"], c["banner"]) for c in claims_json])
