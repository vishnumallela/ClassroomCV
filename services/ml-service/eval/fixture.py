"""Load a captured fixture (eval/capture.py) back into pipeline inputs.

Keeping this in one place is what lets every harness entry point — the
regression gates, the scenario suite, the diagnostic replay — run the exact
derivation production runs, with the exact appearance evidence, and no
database or GPU anywhere in the loop.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.models import Detection, VideoMeta

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class Fixture:
    name: str
    meta: VideoMeta
    zones: list[dict]
    detections: list[Detection]
    # raw_track_id -> torso histogram / CLIP gallery (a list of samples; older
    # fixtures carry a single median vector, which is just a gallery of one).
    hists: dict[int, list[list[float]]]
    embeds: dict[int, list[list[float]]]
    source: Optional[str] = None

    @property
    def raw_track_count(self) -> int:
        return len({d.raw_track_id for d in self.detections})


def _load_appearance(path: Path) -> tuple[dict, dict]:
    """(hists, embeds). Embed galleries carry their sample timestamps when the
    fixture has them, in the (ts, vector) shape teacher_track expects."""
    hists: dict[int, list[list[float]]] = {}
    embeds: dict[int, list] = {}
    if not path.exists():
        return hists, embeds
    with np.load(path) as z:
        stamps = {
            int(k.partition(":")[2]): np.asarray(z[k], dtype=np.int64)
            for k in z.files
            if k.startswith("embedts:")
        }
        for key in z.files:
            kind, _, rid = key.partition(":")
            if kind == "embedts":
                continue
            arr = np.atleast_2d(np.asarray(z[key], dtype=np.float64))
            vecs = [list(map(float, v)) for v in arr]
            if kind == "hist":
                hists[int(rid)] = vecs
            elif int(rid) in stamps and len(stamps[int(rid)]) == len(vecs):
                embeds[int(rid)] = list(zip(map(int, stamps[int(rid)]), vecs))
            else:
                embeds[int(rid)] = vecs
    return hists, embeds


def load(name: str, fixture_dir: Optional[Path] = None) -> Optional[Fixture]:
    """Fixture `name`, or None when it is not present (harness auto-skips)."""
    base = fixture_dir or FIXTURE_DIR
    dets_path = base / f"{name}.dets.jsonl.gz"
    if not dets_path.exists():
        return None

    detections: list[Detection] = []
    with gzip.open(dets_path, "rt") as f:
        header = json.loads(f.readline())
        for line in f:
            row = json.loads(line)
            detections.append(
                Detection(
                    video_ts_ms=row["video_ts_ms"],
                    raw_track_id=row["raw_track_id"],
                    bbox=row["bbox"],
                    conf=row["conf"],
                    standing=row["standing"],
                    back_to_camera=row["back_to_camera"],
                    track_no=row.get("track_no"),
                    occlusion=float(row.get("occlusion") or 0.0),
                    body=row.get("body"),
                )
            )
    info = header.get("info") or {}
    max_ts = max((d.video_ts_ms for d in detections), default=0)
    hists, embeds = _load_appearance(base / f"{name}.appearance.npz")
    return Fixture(
        name=name,
        meta=VideoMeta(
            duration_ms=int(info.get("duration_ms") or max_ts),
            fps=float(info.get("fps") or 0.0),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
        ),
        zones=header.get("zones") or [],
        detections=detections,
        hists=hists,
        embeds=embeds,
        source=header.get("source"),
    )


def available(fixture_dir: Optional[Path] = None) -> list[str]:
    base = fixture_dir or FIXTURE_DIR
    if not base.exists():
        return []
    return sorted(p.name[: -len(".dets.jsonl.gz")] for p in base.glob("*.dets.jsonl.gz"))
