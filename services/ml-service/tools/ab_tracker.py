"""A/B the teacher tracker against a previous version, on REAL stored rows.

    cd services/ml-service
    PYTHONPATH=. .venv/bin/python tools/ab_tracker.py <git-ref> <video-id>

Phase 1 (migration 0014) is what makes this possible: every teacher-class box
the detector offered is in detection_events, so two versions of app/teacher.py
can be run on the identical candidate set a paid GPU pass produced — locally,
in seconds, for nothing. This is how the containment-vs-IoU dedup defect was
caught in minutes rather than on a $0.72/hr pod, and it is the bar for any
change to the tracker: run it on the 37-minute single-teacher baseline
(b6d19a9c-45c6-4ad7-b8b3-0fe0129c3543) and explain every line that differs.

<git-ref> names the OLD version (any commit, branch or tag); the new one is
whatever is in the working tree. Two comparisons are printed: the tracker's
own primary segment, box by box, and then jobs.derive_result end to end —
tracks with overlay, every analytics field, data_quality, every event.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import asyncpg

from app import jobs
from app import teacher as NEW
from app.models import CLASS_TEACHER, Detection, VideoMeta

ROOT = Path(__file__).resolve().parents[3]
TEACHER_PY = "services/ml-service/app/teacher.py"


def load_old(ref: str):
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:{TEACHER_PY}"], text=True)
    tmp = Path(tempfile.mkdtemp()) / "teacher_old.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("teacher_old", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teacher_old"] = mod  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    if not hasattr(mod.TeacherTrack, "others"):
        # Pre-segment versions have no `others`; jobs.py iterates it.
        build = mod.build_teacher_track

        def shim(*a, **k):
            t = build(*a, **k)
            t.others = []
            return t

        mod.build_teacher_track = shim
    return mod


async def load_video(video_id: str):
    dsn = re.search(r"^DATABASE_URL=(.+)$", Path(".env").read_text(), re.M).group(1).strip()
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "select video_ts_ms, bbox, confidence, meta from detection_events "
            "where video_id = $1 order by video_ts_ms",
            video_id,
        )
        zones = await conn.fetch("select kind, polygon from zones where video_id = $1", video_id)
        video = await conn.fetchrow(
            "select duration_ms, fps, width, height from videos where id = $1", video_id
        )
    finally:
        await conn.close()
    js = lambda v: v if isinstance(v, (dict, list)) else json.loads(v or "{}")
    dets = [
        Detection(
            video_ts_ms=int(r["video_ts_ms"]),
            cls=int(js(r["meta"]).get("cls", CLASS_TEACHER)),
            bbox=js(r["bbox"]),
            conf=float(r["confidence"]),
        )
        for r in rows
    ]
    zs = [{"kind": z["kind"], "polygon": js(z["polygon"])} for z in zones]
    meta = VideoMeta(
        int(video["duration_ms"] or max(d.video_ts_ms for d in dets)),
        float(video["fps"] or 0),
        int(video["width"] or 0),
        int(video["height"] or 0),
    )
    return dets, zs, meta


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list) and k == "teacher":  # heatmap cells
            out[f"{prefix}{k}"] = ("heatmap", sum(v), len(v))
        elif isinstance(v, list):
            out[f"{prefix}{k}"] = json.dumps(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def main(ref: str, video_id: str) -> int:
    OLD = load_old(ref)
    dets, zones, meta = asyncio.run(load_video(video_id))
    key = lambda d: (d.video_ts_ms, d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"], round(d.conf, 6))

    a = OLD.build_teacher_track(copy.deepcopy(dets), meta.duration_ms)
    b = NEW.build_teacher_track(copy.deepcopy(dets), meta.duration_ms)
    print(f"video {video_id}  rows={len(dets)}  old={ref}")
    print(f"\n--- tracker: primary segment ---\n{'':22s}{'OLD':>14s}{'NEW':>14s}")
    differs = 0
    for name, av, bv in (
        ("primary detections", len(a.detections), len(b.detections)),
        ("first_ms", a.first_ms, b.first_ms),
        ("last_ms", a.last_ms, b.last_ms),
        ("coverage", a.coverage, b.coverage),
        ("mean_conf", a.mean_conf, b.mean_conf),
        ("rejected_jumps", a.rejected_jumps, b.rejected_jumps),
        ("co_presence_ms", a.co_presence_ms, b.co_presence_ms),
        ("max_simultaneous", a.max_simultaneous, b.max_simultaneous),
    ):
        flag = "" if av == bv else "   <-- DIFFERS"
        differs += bool(flag)
        print(f"  {name:20s}{str(av):>14s}{str(bv):>14s}{flag}")
    segs = getattr(b, "segments", [])
    print(f"  {'segments':20s}{'-':>14s}{str(len(segs)):>14s}")
    print(f"  {'substantial':20s}{'-':>14s}{str(sum(1 for s in segs if s.substantial)):>14s}")
    print(f"  {'others (numbered)':20s}{'-':>14s}{str(len(getattr(b, 'others', []))):>14s}")
    A = {key(d) for d in a.detections}
    B = {key(d) for d in b.detections}
    ts = sorted({k[0] for k in (A ^ B)})
    print(f"\n  instants where the chosen box differs: {len(ts)}")
    for t in ts[:12]:
        o = [k for k in A - B if k[0] == t]
        n = [k for k in B - A if k[0] == t]
        oc = f"conf={o[0][5]:.3f} x={o[0][1]:.3f}" if o else "(none)"
        nc = f"conf={n[0][5]:.3f} x={n[0][1]:.3f}" if n else "(none)"
        print(f"    t={t / 1000:8.1f}s  old: {oc:24s} new: {nc}")
    if a.notes != b.notes:
        print(f"  notes differ:\n    old: {a.notes}\n    new: {b.notes}")

    def run(mod):
        jobs.teacher_mod = mod
        try:
            return jobs.derive_result(meta, copy.deepcopy(dets), zones, actions_available=False)
        finally:
            jobs.teacher_mod = NEW

    ra, rb = run(OLD), run(NEW)
    fa = flatten({"tracks": {str(i): t for i, t in enumerate(ra["tracks"])}, "analytics": ra["analytics"], "events": ra["events"]})
    fb = flatten({"tracks": {str(i): t for i, t in enumerate(rb["tracks"])}, "analytics": rb["analytics"], "events": rb["events"]})
    diff = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    print(f"\n--- derive_result end to end: {len(fa)} flattened fields ---")
    if diff:
        for k in diff[:30]:
            print(f"  DIFFERS {k}: {str(fa.get(k))[:60]} -> {str(fb.get(k))[:60]}")
    else:
        print("  IDENTICAL (tracks incl. overlay, every analytics field, data_quality, every event)")
    print(f"  tracks now: {[t['role'] for t in rb['tracks']]}")
    return 1 if (differs or diff or ts) else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
