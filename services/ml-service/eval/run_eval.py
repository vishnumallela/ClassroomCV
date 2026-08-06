"""Accuracy + performance regression harness.

Two halves, both offline and both replaying the production derivation
(jobs.remerge_from_raw + jobs.derive_result — what /rederive runs):

REAL     captured fixtures of actual lessons (eval/capture.py). Where a fixture
         has per-frame teacher ground truth (eval/annotate_teacher.py) the gates
         are IDENTITY gates — did we follow the right person, from when, with
         how many switches — because the aggregate KPIs they used to gate on can
         be right for the wrong reasons: a pipeline that follows the teacher for
         half the lesson and a look-alike for the other half reports a perfectly
         plausible teacher_present_ms.

SCENARIO synthetic streams reproducing one real classroom failure each, with
         truth exact by construction (eval/scenarios.py). Milliseconds to run,
         committable, and they cover the cases a given piece of footage happens
         not to contain.

Usage (from services/ml-service):
    uv run python eval/run_eval.py                  # everything available
    uv run python eval/run_eval.py --scenarios      # synthetic only (no fixtures needed)
    uv run python eval/run_eval.py --fixtures khaitan_ch10

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

from app.models import VideoMeta  # noqa: E402
from eval import appearance, fixture as fixture_mod, metrics, replay, scenarios  # noqa: E402

GROUND_TRUTH = json.loads((EVAL_DIR / "ground_truth.json").read_text())
GT_DIR = EVAL_DIR / "gt"


def _load_truth(name: str) -> dict[int, tuple[float, float, float, float]]:
    path = GT_DIR / f"{name}.teacher.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(ts): tuple(box) for ts, box in data["anchors"].items()}


def _report(rows: list[tuple[str, bool, str]], indent: str = "    ") -> bool:
    ok = True
    for name, passed, detail in rows:
        ok = ok and passed
        print(f"{indent}[{'PASS' if passed else 'FAIL'}] {name:<20} {detail}")
    return ok


def run_scenarios(only: list[str]) -> bool:
    print("\n=== scenarios (synthetic, exact truth) ===")
    ok = True
    for scenario in scenarios.build_all():
        if only and scenario.name not in only:
            continue
        fx = fixture_mod.Fixture(
            name=scenario.name,
            meta=scenario.meta,
            zones=scenario.zones,
            detections=scenario.detections,
            hists=scenario.hists,
            embeds=scenario.embeds,
        )
        t0 = time.perf_counter()
        rp = replay.run(fx)
        took = time.perf_counter() - t0
        predicted = metrics.teacher_boxes(rp.tracks, fx.detections)
        m = metrics.evaluate_teacher(scenario.truth, predicted)
        print(f"\n  {scenario.name}: {scenario.description} ({took:.1f}s)")
        if scenario.truth:
            print(f"    {metrics.summarize('', m).strip()}")
        else:
            print(f"    predicted_frames={m['predicted_frames']} (expected none)")
        ok = _report(metrics.gate(m, scenario.gates)) and ok
    return ok


def run_fixture(name: str) -> bool:
    fx = fixture_mod.load(name)
    if fx is None:
        print(f"\n=== {name} ===\n    [SKIP] fixture not present")
        return True
    spec = GROUND_TRUTH.get("videos", {}).get(name, {})
    print(f"\n=== {name}: {spec.get('label', 'captured lesson')} ===")
    print(
        f"    {len(fx.detections)} detections over {fx.raw_track_count} raw tracks, "
        f"{fx.meta.duration_ms / 1000:.0f}s, zones={[z['kind'] for z in fx.zones]}"
    )
    rp = replay.run(fx)
    analytics = rp.analytics
    teacher = rp.teacher_track

    # The evidence going IN, measured directly. Every other number here comes
    # out of the assignment, so a change that destroys the appearance signal
    # shows up only as a mysterious coverage drop several stages later — which
    # is how a histogram change that halved the identity signal (separation AUC
    # 0.625 -> 0.463) reached the working tree unnoticed.
    truth = _load_truth(name)
    app = appearance.measure(fx, truth)
    print(f"    {appearance.summarize(app)}")
    ok = _report(metrics.gate(app, spec.get("appearance_gates", {})))

    if truth:
        m = metrics.evaluate_teacher(truth, metrics.teacher_boxes(rp.tracks, fx.detections))
        print(f"    {metrics.summarize('', m).strip()}")
        ok = _report(metrics.gate(m, spec.get("identity_gates", {}))) and ok
    else:
        print("    [info] no per-frame ground truth; KPI gates only")

    actuals = {
        "teacher_present_ms": analytics["teacher_present_ms"],
        "teacher_board_ms": analytics["teacher_board_ms"] or 0,
        "entries": analytics["entries"],
        "exits": analytics["exits"],
        "teacher_tracks": sum(1 for t in rp.tracks if t["role"] == "teacher"),
        "heatmap_samples": sum(analytics["heatmap"]["teacher"]),
    }
    rows: list[tuple[str, bool, str]] = []
    for metric, gate in (spec.get("gates") or {}).items():
        if metric not in actuals:
            continue
        passed = abs(actuals[metric] - gate["value"]) <= gate["tol"]
        rows.append(
            (metric, passed, f"actual={actuals[metric]} expected={gate['value']} +/- {gate['tol']}")
        )
    ok = _report(rows) and ok

    budgets = GROUND_TRUTH["budgets"]
    rows = [
        (
            "remerge_seconds",
            rp.merge_s <= budgets["remerge_seconds_gate"],
            f"actual={rp.merge_s:.1f}s gate={budgets['remerge_seconds_gate']}s",
        ),
        (
            "derive_seconds",
            rp.derive_s <= budgets["derive_seconds_gate"],
            f"actual={rp.derive_s:.1f}s gate={budgets['derive_seconds_gate']}s",
        ),
    ]
    ok = _report(rows) and ok
    print(
        f"    [info] identities={len(rp.tracks)} "
        f"teacher_raw_ids={teacher['meta']['raw_track_ids'] if teacher else None}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", action="store_true", help="run only the scenario suite")
    ap.add_argument("--fixtures", nargs="*", default=None, help="fixture names to replay")
    ap.add_argument("--only", nargs="*", default=[], help="scenario names to run")
    args = ap.parse_args()

    ok = True
    if not args.fixtures or args.scenarios:
        ok = run_scenarios(args.only) and ok
    if not args.scenarios:
        names = args.fixtures if args.fixtures else fixture_mod.available()
        if not names:
            print("\nno fixtures captured (see eval/capture.py)")
        for name in names:
            ok = run_fixture(name) and ok

    print(f"\n{'ALL GATES PASS' if ok else 'GATE FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
