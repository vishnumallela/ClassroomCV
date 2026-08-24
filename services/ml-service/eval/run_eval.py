"""Accuracy regression harness: does the pipeline still follow the right person?

Runs the REAL production path — detector.detect_video + jobs.derive_result, the
same two calls /analyze makes — over a local video, then scores the teacher
timeline it produced against per-frame ground truth (eval/gt/<name>.teacher.json).

Gates are IDENTITY gates, not KPI totals. A pipeline that follows the teacher
for half a lesson and a look-alike for the other half reports a perfectly
plausible teacher_present_ms; only coverage/purity/switches catch that.

This replaces a harness built on frozen detection fixtures. Those made sense
when the detector was a fixed, expensive, general-purpose model and all the
interesting logic lived downstream of it; now the detector IS the interesting
part, and a fixture of its output could not have caught a regression in it.
The cost is that a run needs the video and a few minutes of inference rather
than milliseconds — which is why the fast, per-rule checks live in
tests/test_teacher.py and tests/test_zones.py and run in CI, while this is the
before-you-ship pass.

Usage (from services/ml-service):
    uv run python eval/run_eval.py                 # every configured video
    uv run python eval/run_eval.py demo_teacher    # just one
    uv run python eval/run_eval.py --fps 2         # faster, coarser sweep

Exit code 0 only when every gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app import detector, jobs  # noqa: E402
from eval import metrics  # noqa: E402

REPO_ROOT = EVAL_DIR.parents[2]
CONFIG = json.loads((EVAL_DIR / "ground_truth.json").read_text())
GT_DIR = EVAL_DIR / "gt"


def _load_truth(name: str) -> dict[int, tuple[float, float, float, float]]:
    path = GT_DIR / f"{name}.teacher.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(ts): tuple(box) for ts, box in data["anchors"].items()}


def _resolve(video: str) -> Path:
    p = Path(video)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _report(rows: list[tuple[str, bool, str]], indent: str = "    ") -> bool:
    ok = True
    for name, passed, detail in rows:
        ok = ok and passed
        print(f"{indent}[{'PASS' if passed else 'FAIL'}] {name:<18} {detail}")
    return ok


def _nearest_truth(
    truth: dict[int, tuple], predicted: dict[int, tuple], tol_ms: int = 120
) -> dict[int, tuple]:
    """Re-key predictions onto the ground truth's own sampled instants.

    Ground truth was annotated at its own cadence; a run at a different
    sample_fps lands on neighbouring milliseconds. Without this every frame
    reads as a miss for a reason that has nothing to do with accuracy.
    """
    if not predicted:
        return {}
    stamps = sorted(predicted)
    out: dict[int, tuple] = {}
    for ts in truth:
        # Nearest predicted stamp within tolerance.
        lo, hi = 0, len(stamps) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if stamps[mid] < ts:
                lo = mid + 1
            else:
                hi = mid - 1
        for idx in (hi, lo):
            if 0 <= idx < len(stamps):
                d = abs(stamps[idx] - ts)
                if d <= tol_ms and (best is None or d < best[0]):
                    best = (d, stamps[idx])
        if best is not None:
            out[ts] = predicted[best[1]]
    return out


def run_video(name: str, spec: dict, sample_fps: float) -> bool:
    video = _resolve(spec["video"])
    print(f"\n=== {name}: {spec.get('label', '')} ===")
    if not video.is_file():
        print(f"    [SKIP] video not found: {video}")
        return True

    t0 = time.perf_counter()
    meta, detections = detector.detect_video(str(video), sample_fps=sample_fps)
    detect_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = jobs.derive_result(meta, detections, spec.get("zones", []))
    derive_s = time.perf_counter() - t0

    frames = len({d.video_ts_ms for d in detections})
    print(
        f"    {len(detections)} detections over {frames} frames, "
        f"{meta.duration_ms / 1000:.0f}s  (detect {detect_s:.0f}s, derive {derive_s:.2f}s)"
    )
    dq = result["analytics"].get("data_quality") or {}
    if dq:
        print(
            f"    teacher coverage {dq['coverage'] * 100:.1f}%  breaks {dq['breaks']}  "
            f"mean conf {dq['mean_confidence']:.2f}  -> {dq['confidence']['overall']}"
        )

    ok = True
    truth = _load_truth(spec.get("gt", name))
    if truth:
        predicted = metrics.teacher_boxes(result["tracks"], detections)
        aligned = _nearest_truth(truth, predicted)
        m = metrics.evaluate_teacher(truth, aligned)
        print("    " + metrics.summarize(name, m))
        ok = _report(metrics.gate(m, spec.get("identity_gates", {}))) and ok
    else:
        print("    [info] no per-frame ground truth; reporting only")

    budget = CONFIG.get("budgets", {}).get("derive_seconds_gate")
    if budget is not None:
        ok = _report(
            [("derive_seconds", derive_s <= budget, f"actual={derive_s:.2f}s gate={budget}s")]
        ) and ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="video names to run (default: all)")
    ap.add_argument("--fps", type=float, default=5.0, help="sample rate (default 5)")
    args = ap.parse_args()

    ok = True
    for name, spec in CONFIG["videos"].items():
        if args.only and name not in args.only:
            continue
        ok = run_video(name, spec, args.fps) and ok

    print("\n" + ("ALL GATES PASS" if ok else "GATES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
