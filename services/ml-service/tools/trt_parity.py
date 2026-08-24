"""Prove the TensorRT engine agrees with PyTorch — and measure whether it is worth it.

Run this ON THE POD before trusting RFDETR_TENSORRT=true in production. It
exists because the TensorRT serving path is the one part of this pipeline that
cannot be verified off-GPU: rfdetr ships TensorRT export but not TensorRT
inference, so app/tensorrt_backend.py supplies the preprocessing and batching
around rfdetr's own PostProcess. A mismatch there does not raise — it shifts
every box slightly, and every KPI with it.

Two questions, in the order that matters:

  1. CORRECTNESS. Same frames through both backends; boxes matched by IoU. A
     handful of low-confidence detections may legitimately fall either side of
     the threshold under fp16, so the gate is on the TEACHER class at the
     production threshold, which is what the product actually depends on.

  2. SPEED. Only if (1) passes. This project has adopted TensorRT on a vendor
     claim once before: "~5x" measured 1.05-1.25x, and the real win turned out
     to be an unrelated fp16 bug. fp16 PyTorch is the baseline here precisely
     so that mistake cannot repeat — if the ratio is near 1.0, TensorRT is
     costing you a non-portable artifact and a heavyweight dependency for
     nothing, and RFDETR_TENSORRT=false is the correct answer.

Usage (from services/ml-service, on the GPU pod):
    uv run --extra tensorrt python tools/trt_parity.py <video> [--frames 64]

Exit code 0 only when the teacher-class detections agree.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import detector  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.models import CLASS_NAMES, CLASS_TEACHER  # noqa: E402

# Boxes this close are the same detection; below it the two backends disagree
# about where something is.
MATCH_IOU = 0.9
# Allowed confidence drift on a matched box. fp16 rounding moves scores in the
# third decimal; anything larger means the graphs differ.
MAX_CONF_DELTA = 0.02


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _grab(video: str, n: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.1 + 0.8 * i / max(1, n - 1))))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def _run(frames: list[np.ndarray], batch: int) -> tuple[list, float]:
    """All frames through whatever backend is currently loaded."""
    model = detector._get_model()
    results = []
    # One untimed pass so engine deserialisation / autotune is not counted.
    detector._predict_batch(model, frames[:batch])
    t0 = time.perf_counter()
    for i in range(0, len(frames), batch):
        results.extend(detector._predict_batch(model, frames[i : i + batch]))
    return results, time.perf_counter() - t0


def _reset() -> None:
    detector._model = None
    detector._trt = None
    get_settings.cache_clear()


def _teacher_boxes(det, threshold: float):
    return sorted(
        (
            (float(c), tuple(float(v) for v in b))
            for cid, c, b in zip(det.class_id, det.confidence, det.xyxy)
            if int(cid) == CLASS_TEACHER and float(c) >= threshold
        ),
        key=lambda t: -t[0],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()

    settings = get_settings()
    threshold = settings.teacher_conf
    batch = max(1, int(settings.rfdetr_batch))
    frames = _grab(args.video, args.frames)
    if not frames:
        print(f"could not read frames from {args.video}")
        return 1
    print(f"{len(frames)} frames, batch {batch}, teacher threshold {threshold}")

    import os

    os.environ["RFDETR_TENSORRT"] = "false"
    _reset()
    pt_results, pt_s = _run(frames, batch)
    pt_backend = detector.serving_backend()
    print(f"  pytorch  ({pt_backend}): {pt_s:.2f}s  {len(frames)/pt_s:.1f} fps")

    os.environ["RFDETR_TENSORRT"] = "true"
    _reset()
    trt_results, trt_s = _run(frames, batch)
    trt_backend = detector.serving_backend()
    print(f"  tensorrt ({trt_backend}): {trt_s:.2f}s  {len(frames)/trt_s:.1f} fps")

    if trt_backend != "tensorrt":
        print(
            "\nFAIL: the TensorRT backend did not load, so this compared PyTorch "
            "with itself. Check the log above for the export/load error."
        )
        return 1

    # --- correctness ------------------------------------------------------- #
    mismatches = 0
    conf_deltas = []
    ious = []
    for i, (a, b) in enumerate(zip(pt_results, trt_results)):
        ta, tb = _teacher_boxes(a, threshold), _teacher_boxes(b, threshold)
        if len(ta) != len(tb):
            mismatches += 1
            print(f"  frame {i}: pytorch found {len(ta)} teacher box(es), tensorrt {len(tb)}")
            continue
        for (ca, ba), (cb, bb) in zip(ta, tb):
            iou = _iou(ba, bb)
            ious.append(iou)
            conf_deltas.append(abs(ca - cb))
            if iou < MATCH_IOU or abs(ca - cb) > MAX_CONF_DELTA:
                mismatches += 1
                print(f"  frame {i}: IoU={iou:.3f} conf {ca:.3f} vs {cb:.3f}")

    print()
    if ious:
        print(
            f"teacher boxes compared: {len(ious)}  "
            f"IoU min={min(ious):.4f} mean={np.mean(ious):.4f}  "
            f"max conf delta={max(conf_deltas):.4f}"
        )
    speedup = pt_s / trt_s if trt_s > 0 else 0.0
    print(f"speedup vs fp16 PyTorch: {speedup:.2f}x")

    if mismatches:
        print(f"\nFAIL: {mismatches} disagreement(s). Do NOT enable RFDETR_TENSORRT.")
        return 1
    print("\nPASS: the engine agrees with PyTorch on the teacher class.")
    if speedup < 1.3:
        print(
            f"NOTE: {speedup:.2f}x is not much for a non-portable engine and a "
            "CUDA-only dependency. Consider leaving RFDETR_TENSORRT=false — "
            "this project has been here before (docs/rfdetr-pipeline.md)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
