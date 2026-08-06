"""Replay a captured fixture through the production derivation, offline.

One function, used by every harness entry point, so a number printed by the
regression gates and a number printed by a diagnostic run cannot drift apart:
both come from `jobs.remerge_from_raw` + `jobs.derive_result`, which is exactly
what /rederive executes in production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app import jobs
from eval.fixture import Fixture


@dataclass
class Replay:
    result: dict
    merge_s: float
    derive_s: float
    fixture: Fixture

    @property
    def analytics(self) -> dict:
        return self.result["analytics"]

    @property
    def tracks(self) -> list[dict]:
        return self.result["tracks"]

    @property
    def teacher_track(self) -> Optional[dict]:
        return next((t for t in self.tracks if t["role"] == "teacher"), None)


def run(fx: Fixture) -> Replay:
    """merge + derive a fixture exactly as /rederive would."""
    # Detections are mutated in place by derivation (track_no is rewritten), so
    # a replay must not leak state into the next one.
    hists = {rid: samples for rid, samples in fx.hists.items()}
    t0 = time.perf_counter()
    identities = jobs.remerge_from_raw(fx.detections, hists, fx.embeds)
    merge_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = jobs.derive_result(
        fx.meta,
        fx.detections,
        identities,
        fx.zones,
        track_embeds=fx.embeds,
        track_hists=fx.hists,
    )
    derive_s = time.perf_counter() - t0
    return Replay(result=result, merge_s=merge_s, derive_s=derive_s, fixture=fx)
