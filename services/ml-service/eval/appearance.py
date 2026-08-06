"""Does the stored appearance evidence actually tell people apart?

Every other number in this harness is computed from the assignment's OUTPUT, so
a change that quietly destroys the evidence going IN only shows up as a
mysterious drop in coverage several stages later — which is exactly how a
histogram change that halved the identity signal reached the working tree
before anyone noticed. This module measures the input directly.

Two questions, both answered per modality (torso colour, CLIP/re-ID gallery):

SEPARATION  how well the descriptor ranks "two views of one person" above
            "two different people", as an AUC. 0.5 is a coin flip. Ground
            truth supplies the positives where a lesson has been annotated.

COVERAGE    what fraction of each track's LIFETIME its stored samples span.
            A descriptor can be excellent and still useless if it was all
            collected in the first ten seconds of a nine-minute track: the
            prefix sampler that shipped for months scored a median of 8%, and
            no metric in the harness could see it.

Negatives are free and need no annotation at all: two tracklets alive at the
same instant are certainly different people.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app import adult as adult_mod
from app import teacher_track as tt
from eval.fixture import Fixture

# A tracklet needs this many detections before its appearance summary means
# anything.
MIN_DETS = 20
# Two tracklets overlapping by more than this are certainly two people.
CO_PRESENT_MS = 3_000


def _bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(a * b).sum())


def _label_tracklets(tracklets, truth: dict, match_iou: float = 0.3):
    """Split tracklets into CLEANLY-teacher and CLEANLY-not, using per-frame truth.

    Labelling by raw track id does not work: the ids that matter most are
    exactly the ones the tracker handed from one person to another mid-track,
    so calling such an id "the teacher" poisons the positives with negatives
    and drags any separation measure back to a coin flip. A tracklet counts as
    hers only when nearly all of its detections match her ground-truth box, and
    as somebody else's only when almost none do; the ambiguous middle is
    excluded rather than guessed.
    """
    from eval.metrics import iou as box_iou

    mine, theirs = [], []
    for t in tracklets:
        checked = hit = 0
        for d in t.dets:
            gt = truth.get(d.video_ts_ms)
            if gt is None:
                continue
            checked += 1
            b = d.bbox
            if box_iou(gt, (b["x"], b["y"], b["w"], b["h"])) >= match_iou:
                hit += 1
        if checked < 10:
            continue
        share = hit / checked
        if share >= 0.8:
            mine.append(t)
        elif share <= 0.05:
            theirs.append(t)
    return mine, theirs


def measure(fx: Fixture, truth: Optional[dict] = None) -> dict:
    """Appearance separation + temporal coverage for one fixture."""
    dets_by_raw: dict[int, list] = {}
    for d in fx.detections:
        dets_by_raw.setdefault(d.raw_track_id, []).append(d)

    plane = adult_mod.fit_ground_plane(dets_by_raw)
    galleries = tt.normalize_galleries(fx.embeds)
    tracklets = [
        t for t in tt.build_tracklets(dets_by_raw, plane, galleries)
        if len(t.dets) >= MIN_DETS
    ]

    embeds: dict[int, np.ndarray] = {}
    hists: dict[int, np.ndarray] = {}
    for t in tracklets:
        g = tt._gallery_for(t, galleries)
        if g.size:
            v = g.mean(axis=0)
            embeds[t.tid] = v / max(float(np.linalg.norm(v)), 1e-9)
        h = fx.hists.get(t.raw_id)
        if h is not None:
            arr = np.asarray(h, dtype=np.float64)
            arr = arr.mean(axis=0) if arr.ndim > 1 else arr
            total = float(arr.sum())
            if total > 0:
                hists[t.tid] = arr / total

    # How much of each RAW track's lifetime its samples actually span. This is
    # the number that exposes a sampler which stops early.
    spans: list[float] = []
    for raw_id, dets in dets_by_raw.items():
        life = max(d.video_ts_ms for d in dets) - min(d.video_ts_ms for d in dets)
        entry = galleries.get(raw_id)
        if life < 60_000 or entry is None or entry[0] is None or len(entry[0]) < 2:
            continue
        ts = entry[0]
        spans.append(float(int(ts.max()) - int(ts.min())) / life)

    out: dict = {
        "tracklets": len(tracklets),
        "gallery_span_p50": round(float(np.median(spans)), 3) if spans else None,
        "long_tracks": len(spans),
    }

    if not truth:
        out["separation_hist"] = None
        out["separation_embed"] = None
        return out

    mine, theirs = _label_tracklets(tracklets, truth)
    out["clean_teacher_tracklets"] = len(mine)
    out["clean_other_tracklets"] = len(theirs)

    def auc(vectors: dict[int, np.ndarray], sim) -> Optional[float]:
        pos, neg = [], []
        for i, a in enumerate(mine):
            for b in mine[i + 1 :]:
                if a.tid in vectors and b.tid in vectors and (
                    min(a.last_ms, b.last_ms) - max(a.first_ms, b.first_ms) <= CO_PRESENT_MS
                ):
                    pos.append(sim(vectors[a.tid], vectors[b.tid]))
        for a in mine:
            for b in theirs:
                if a.tid in vectors and b.tid in vectors:
                    neg.append(sim(vectors[a.tid], vectors[b.tid]))
        if not pos or not neg:
            return None
        return round(
            float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 3
        )

    out["separation_hist"] = auc(hists, _bhattacharyya)
    out["separation_embed"] = auc(embeds, lambda a, b: float(np.dot(a, b)))
    return out


def summarize(m: dict) -> str:
    def pct(v):
        return "  n/a" if v is None else f"{v * 100:5.1f}%"

    return (
        f"appearance: separation colour={pct(m['separation_hist'])} "
        f"embed={pct(m['separation_embed'])}  "
        f"gallery spans {pct(m['gallery_span_p50'])} of a track's life "
        f"({m['long_tracks']} long tracks)"
    )
