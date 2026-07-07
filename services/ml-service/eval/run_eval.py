"""Accuracy + performance regression harness.

Re-runs the pure derivation pipeline (remerge_from_raw + derive_result) on the
raw detections stored in TimescaleDB and diffs the resulting analytics against
eval/ground_truth.json. Nothing is written back, so runs are side-effect free
and safe to repeat while tuning merge/roles/events.

Usage (from services/ml-service):
    uv run python eval/run_eval.py [video_id ...]

Exit code 0 only when every gate passes for every evaluated video.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app import db, jobs
from app.models import VideoMeta
GROUND_TRUTH = json.loads((EVAL_DIR / "ground_truth.json").read_text())


async def _fetch_zones(video_id: str) -> list[dict]:
    conn = await db._connect()
    try:
        rows = await conn.fetch(
            "select kind, polygon from zones where video_id = $1::uuid order by created_at",
            video_id,
        )
    finally:
        await conn.close()
    zones: list[dict] = []
    for r in rows:
        polygon = r["polygon"]
        if isinstance(polygon, str):
            polygon = json.loads(polygon)
        zones.append({"kind": r["kind"], "polygon": polygon})
    return zones


def _check(name: str, actual: float, spec: dict) -> tuple[bool, str]:
    ok = abs(actual - spec["value"]) <= spec["tol"]
    marker = "PASS" if ok else "FAIL"
    return ok, f"  [{marker}] {name:<22} actual={actual:<10} expected={spec['value']} +/- {spec['tol']}"


async def evaluate(video_id: str, spec: dict) -> bool:
    print(f"\n=== {spec.get('label', video_id)} ===")
    detections = await db.fetch_detections(video_id)
    if not detections:
        print(f"  [SKIP] no stored detections for {video_id}")
        return True
    info = await db.fetch_video_info(video_id) or {}
    zones = await _fetch_zones(video_id)
    max_ts = max(d.video_ts_ms for d in detections)
    meta = VideoMeta(
        duration_ms=int(info.get("duration_ms") or max_ts),
        fps=float(info.get("fps") or 0.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
    )
    print(f"  detections={len(detections)} zones={[z['kind'] for z in zones]} duration_ms={meta.duration_ms}")

    t0 = time.perf_counter()
    identities = jobs.remerge_from_raw(detections)
    t_merge = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = jobs.derive_result(meta, detections, identities, zones)
    t_derive = time.perf_counter() - t0

    analytics = result["analytics"]
    tracks = result["tracks"]
    teacher_tracks = sum(1 for t in tracks if t["role"] == "teacher")

    actuals = {
        "teacher_present_ms": analytics["teacher_present_ms"],
        "teacher_board_ms": analytics["teacher_board_ms"] or 0,
        "entries": analytics["entries"],
        "exits": analytics["exits"],
        "teacher_tracks": teacher_tracks,
        "max_students": analytics["max_students"] or 0,
        "avg_students": analytics["avg_students"] or 0.0,
    }

    all_ok = True
    for name, gate in spec["gates"].items():
        ok, line = _check(name, actuals[name], gate)
        all_ok = all_ok and ok
        print(line)

    budgets = GROUND_TRUTH["budgets"]
    merge_ok = t_merge <= budgets["remerge_seconds_gate"]
    derive_ok = t_derive <= budgets["derive_seconds_gate"]
    all_ok = all_ok and merge_ok and derive_ok
    print(f"  [{'PASS' if merge_ok else 'FAIL'}] remerge_seconds        actual={t_merge:.1f}s  gate={budgets['remerge_seconds_gate']}s target={budgets['remerge_seconds_target']}s")
    print(f"  [{'PASS' if derive_ok else 'FAIL'}] derive_seconds         actual={t_derive:.1f}s  gate={budgets['derive_seconds_gate']}s")
    print(f"  [info] total_tracks={len(tracks)} (baseline {spec['report_only']['total_tracks']}; fewer with gates green = better re-ID)")
    return all_ok


async def main() -> int:
    requested = sys.argv[1:]
    videos = GROUND_TRUTH["videos"]
    targets = {vid: spec for vid, spec in videos.items() if not requested or vid in requested}
    if not targets:
        print(f"no ground truth for {requested}")
        return 2
    ok = True
    for vid, spec in targets.items():
        ok = (await evaluate(vid, spec)) and ok
    print(f"\n{'ALL GATES PASS' if ok else 'GATE FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
