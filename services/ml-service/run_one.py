"""Analyse ONE video end-to-end with no database, no Redis, no API.

The point of this script is to prove the pipeline on a GPU pod without dragging
the rest of the stack there. `write_db=False` means Postgres is never touched,
and passing no zones means the run needs nothing but the video and the
checkpoint — so a pod needs no tunnel back to the Mac, which is the part of the
hybrid setup that is fragile.

Usage (on the pod):
    RFDETR_WEIGHTS=/workspace/ml-service/rfdetr-medium.pth \
    DEVICE=cuda REQUIRE_DEVICE=cuda RFDETR_BATCH=16 \
    uv run python run_one.py /workspace/ml-service/lesson.mp4

Detached, for a long lesson:
    nohup env RFDETR_WEIGHTS=... DEVICE=cuda REQUIRE_DEVICE=cuda RFDETR_BATCH=16 \
      uv run python run_one.py /workspace/ml-service/lesson.mp4 > run.log 2>&1 &
    echo $! > run.pid          # then poll run.log

REQUIRE_DEVICE=cuda is not optional on a pod. Without it a host whose driver is
too old for this torch build silently resolves to CPU and the run still
"succeeds" — hours later, having billed a GPU it never used.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    video = str(Path(sys.argv[1]).resolve())
    sample_fps = float(os.environ.get("SAMPLE_FPS", "5"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from app import detector, jobs

    device = detector.get_device()
    print(f"video       {video}")
    print(f"weights     {detector.resolve_model_name()}")
    print(f"device      {device}")
    print(f"sample_fps  {sample_fps}")

    # Prove CUDA with a real matmul rather than torch.cuda.is_available(), which
    # returns True even when the kernels are for the wrong architecture.
    if device.startswith("cuda"):
        import torch

        x = torch.randn(1024, 1024, device="cuda")
        assert float((x @ x).sum()) != 0.0, "CUDA matmul produced zeros"
        print(f"gpu         {torch.cuda.get_device_name(0)} (matmul OK)")

    last = [0.0]
    t0 = time.perf_counter()

    def progress(stage: str, frac: float) -> None:
        if frac - last[0] >= 0.02 or frac >= 1.0:
            last[0] = frac
            el = time.perf_counter() - t0
            eta = el / max(frac, 1e-6) - el
            print(
                f"  {stage:<10} {frac * 100:5.1f}%  elapsed {el / 60:5.1f}m  eta {eta / 60:5.1f}m",
                flush=True,
            )

    result = jobs.run_pipeline(
        video_id="proof",
        video_path=video,
        sample_fps=sample_fps,
        zones=[],
        progress_cb=progress,
        write_db=False,
    )
    wall = time.perf_counter() - t0

    if device.startswith("cuda"):
        import torch

        print(f"peak VRAM   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"wall clock  {wall / 60:.1f} min")
    print(f"backend     {detector.serving_backend()}")

    a = result["analytics"]
    dq = a.get("data_quality") or {}
    print("\n--- result ---")
    print(f"duration        {result['video']['duration_ms'] / 60000:.1f} min")
    print(f"tracks          {len(result['tracks'])}")
    print(f"events          {len(result['events'])}")
    print(f"teacher present {a['teacher_present_ms'] / 60000:.1f} min")
    print(f"entries/exits   {a['entries']}/{a['exits']}")
    print(f"coverage        {dq.get('coverage')}  breaks {dq.get('breaks')}  "
          f"mean conf {dq.get('mean_confidence')}")
    print(f"quality         {(dq.get('confidence') or {}).get('overall')}")
    for note in dq.get("notes", []):
        print(f"  note: {note}")

    out = Path("run_one_result.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nfull result -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
