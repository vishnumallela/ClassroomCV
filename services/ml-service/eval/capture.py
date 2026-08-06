"""Record a real video's DETECTION stage once, so derivation is testable offline.

The detector is the only slow, GPU-bound, non-deterministic-ish stage: a 12-min
1080p clip costs ~10 minutes on a laptop. Everything the identity work actually
changes — merge, adult scoring, teacher assignment, events — is a pure function
of the detector's output. This script runs the detector ONCE against a local
video file and writes a fixture that eval/run_eval.py replays in seconds, which
is what makes iterating on re-ID tractable without a database or a GPU.

Usage (from services/ml-service):
    uv run python eval/capture.py <video-file> <fixture-name> [--fps 5] [--zones zones.json]

Writes eval/fixtures/<fixture-name>.dets.jsonl.gz (header line + one detection
per line) and eval/fixtures/<fixture-name>.appearance.npz (per-raw-track torso
histograms and CLIP embeddings, which are float arrays and do not belong in
JSON). The pair is exactly the evidence /rederive gets in production.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app import detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("name")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--zones", default=None, help="JSON file: [{kind, polygon}]")
    args = ap.parse_args()

    video = str(Path(args.video).resolve())
    zones = json.loads(Path(args.zones).read_text()) if args.zones else []

    t0 = time.perf_counter()
    last = [0.0]

    def progress(frac: float) -> None:
        if frac - last[0] >= 0.02:
            last[0] = frac
            el = time.perf_counter() - t0
            eta = el / max(frac, 1e-6) - el
            print(f"\r  {frac * 100:5.1f}%  elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m", end="", flush=True)

    meta, detections, hists, embeds = detector.detect_video(
        video, sample_fps=args.fps, progress_cb=progress
    )
    print()
    took = time.perf_counter() - t0

    out_dir = EVAL_DIR / "fixtures"
    out_dir.mkdir(exist_ok=True)
    dets_path = out_dir / f"{args.name}.dets.jsonl.gz"
    with gzip.open(dets_path, "wt") as f:
        f.write(
            json.dumps(
                {
                    "video_id": args.name,
                    "source": video,
                    "sample_fps": args.fps,
                    "info": {
                        "duration_ms": meta.duration_ms,
                        "fps": meta.fps,
                        "width": meta.width,
                        "height": meta.height,
                    },
                    "zones": zones,
                }
            )
            + "\n"
        )
        for d in detections:
            row = {
                "video_ts_ms": d.video_ts_ms,
                "raw_track_id": d.raw_track_id,
                "track_no": d.track_no,
                "bbox": d.bbox,
                "conf": round(d.conf, 4),
                "standing": d.standing,
                "back_to_camera": d.back_to_camera,
            }
            # Newer detector fields are optional so old fixtures still load.
            for extra in ("occlusion", "body"):
                val = getattr(d, extra, None)
                if val:
                    row[extra] = val
            f.write(json.dumps(row) + "\n")

    npz_path = out_dir / f"{args.name}.appearance.npz"
    payload: dict[str, np.ndarray] = {}
    for rid, samples in hists.items():
        if samples:
            payload[f"hist:{rid}"] = np.median(
                np.stack([np.asarray(s).ravel() for s in samples]), axis=0
            ).astype(np.float32)
    # The CLIP gallery is stored WITH its sample timestamps: a raw id that the
    # tracker handed from one person to another gets split downstream, and each
    # half must only keep the crops taken while it was that body.
    for rid, gallery in embeds.items():
        if not gallery:
            continue
        payload[f"embed:{rid}"] = np.asarray(
            [v for _ts, v in gallery], dtype=np.float32
        )
        payload[f"embedts:{rid}"] = np.asarray(
            [ts for ts, _v in gallery], dtype=np.int64
        )
    np.savez_compressed(npz_path, **payload)

    raw_ids = {d.raw_track_id for d in detections}
    print(
        f"captured {len(detections)} detections over {len(raw_ids)} raw tracks "
        f"in {took / 60:.1f} min ({meta.width}x{meta.height}, {meta.duration_ms / 1000:.0f}s)"
    )
    print(f"  {dets_path.name} {dets_path.stat().st_size // 1024} KB")
    print(f"  {npz_path.name} {npz_path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
