"""Teacher identification: a HARD-BUDGETED vision-model vote.

Who the teacher is decides every KPI this product ships (her entries/exits,
her board time, her heatmap), so the geometric ranker (roles.assign_roles)
gets one cheap, bounded second opinion: sample a handful of frames, ask a
vision model to POINT at the teacher, map each point to a tracked box, and
majority-vote. At most `vlm_frames` Gemini calls per analysis — about
$0.002 a lesson — and zero calls when no key is configured.

Distilled from the lessons of the earlier (rejected, unbounded) VLM branch;
the heuristics survive, the per-claim call sprawl does not:

- LOCATE-then-map, NOT set-of-marks: send the RAW frame and ask for the
  teacher's normalized centre point. Drawing numbered boxes and asking
  "which number?" fails past ~10 people (models miscount tiny labels).
- Point-in-bbox, NOT nearest-centre: among boxes containing the point,
  prefer the one the point sits most centrally in (distance / box diagonal).
  Nearest-centre snaps onto a closer pupil in a crowd; a standing adult's
  torso is central to her tall box but near the top edge of a foreground
  child's.
- Vote on RAW tracker fragments, then POOL fragments that can be the same
  person (mutually non-co-present). A merged identity can be chimeric, and
  the teacher is often tracked as several ids whose votes must not compete.
  Two fragments sharing >= _CO_PRESENT_MIN sampled frames are two people.
- REJECT needs evidence, not silence: the geometric pick is demoted only
  when the model ANSWERED >= vlm_reject_min_answered frames and never once
  pointed at it. An HTTP failure (answered == 0) keeps the geometric pick —
  the demote-on-failure bug cost a correctly-labelled teacher once.
- Fail closed everywhere: no key / no frames / decode error / inconclusive
  vote returns None and the caller keeps what it had.
- gemini-flash-latest by alias: dated 2.5-flash versions 404 on new API
  keys while still listed. temperature 0; no thinkingConfig (400 on this
  alias); strip ``` fences before parsing; retry 429 with backoff.
- cv2.VideoCapture is not thread-safe: frames are decoded sequentially,
  only the HTTP round-trips fan out.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Optional

import cv2
import httpx

from app.config import get_settings
from app.models import Detection

logger = logging.getLogger(__name__)

_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_PROMPT = (
    "This is a classroom CCTV frame. Exactly one person is the TEACHER (an adult "
    "instructor); everyone else is a child wearing a school uniform. Give the pixel "
    "location of the CENTRE of the teacher's body as fractions of the image size "
    "(x = fraction from the left, 0..1; y = fraction from the top, 0..1). Respond with "
    'ONLY compact JSON: {"x": <0..1>, "y": <0..1>}. Use null for both if no teacher is visible.'
)
# A track's detection must sit within this many ms of the sampled frame to
# count as "present in that frame" for the point->box mapping.
_TS_TOLERANCE_MS = 400
_HTTP_TIMEOUT = 30.0
_CONCURRENCY = 4
# Two fragments sharing at least this many sampled frames are two people (one
# person cannot be two boxes at once); below it, brief overlap is tracker
# noise at an id handoff.
_CO_PRESENT_MIN = 3


class TeacherVote(NamedTuple):
    """Outcome of the vote. track is None when inconclusive; answered is how
    many sampled frames produced a usable point (0 = the model never answered,
    which the caller must treat as no-signal, never as rejection)."""

    track: Optional[int]
    confidence: float
    votes: dict
    answered: int


def _b64_jpeg(frame, max_w: int = 2000) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, round(h * max_w / w)))
    _ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def _parse_point(content: str) -> Optional[tuple[float, float]]:
    """The model's (x, y) in 0..1, or None (absent / garbled)."""
    content = re.sub(r"```(?:json)?\s*|\s*```", "", content)
    found = re.findall(r'\{[^{}]*"x"[^{}]*\}', content, re.S)
    if not found:
        return None
    try:
        j = json.loads(found[-1])  # reasoning echoes may precede the answer
        if j.get("x") is None or j.get("y") is None:
            return None
        return (float(j["x"]), float(j["y"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _ask(client: httpx.Client, key: str, model: str, b64: str) -> Optional[tuple[float, float]]:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2000},
    }
    url = _ENDPOINT_TMPL.format(model=model)
    for attempt in range(3):
        try:
            r = client.post(url, params={"key": key}, json=payload)
        except Exception as exc:
            logger.warning("teacher-id request error: %s", exc)
            return None
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            logger.warning("teacher-id HTTP %s: %s", r.status_code, r.text[:200])
            return None
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return _parse_point("".join(p.get("text", "") for p in parts))
        except (KeyError, IndexError, ValueError):
            return None
    return None


def _point_to_fragment(
    px: float, py: float, boxes: dict[int, tuple[float, float, float, float]]
) -> Optional[int]:
    """Fragment whose box CONTAINS the point, preferring the most central hit.

    Returns None when no box contains the point (model pointed at empty floor
    or between people): an ambiguous frame casts NO vote.
    """
    best: Optional[tuple[float, int]] = None
    for no, (x, y, w, h) in boxes.items():
        if x <= px <= x + w and y <= py <= y + h:
            cx, cy = x + w / 2.0, y + h / 2.0
            diag = math.hypot(w, h) or 1e-6
            central = math.hypot(px - cx, py - cy) / diag
            if best is None or central < best[0]:
                best = (central, no)
    return best[1] if best is not None else None


def pool_fragments(votes: Counter, frames_of_raw: dict[int, set[int]]) -> list[int]:
    """Group vote-getting fragments that can be the SAME person.

    Greedy by vote count: admit a fragment only when it is non-co-present with
    everything already pooled. Only fragments the model actually pointed at are
    eligible, so this never speculates about unseen tracks.
    """
    pool: list[int] = []
    for raw, _n in votes.most_common():
        seen = frames_of_raw.get(raw, set())
        if all(
            len(seen & frames_of_raw.get(other, set())) < _CO_PRESENT_MIN
            for other in pool
        ):
            pool.append(raw)
    return pool


def verify_teacher(
    video_path: str,
    dets_by_track: dict[int, list[Detection]],
    duration_ms: int,
) -> Optional[TeacherVote]:
    """One bounded vote: at most settings.vlm_frames Gemini calls.

    Samples frames evenly across the middle of the lesson (10%..90% — the
    edges are empty or in flux), deduplicated, decodes them sequentially, asks
    concurrently, maps each point onto the raw-fragment boxes visible in that
    frame, pools fragments that can be one person, and crowns the merged track
    holding most of the pooled timeline. None when disabled, unanswerable, or
    inconclusive (fail closed).
    """
    s = get_settings()
    if not s.gemini_api_key or not video_path or not dets_by_track or duration_ms <= 0:
        return None

    all_ts = sorted({d.video_ts_ms for dets in dets_by_track.values() for d in dets})
    if not all_ts:
        return None

    # Raw-fragment index: boxes are matched per raw id, votes pooled across ids.
    track_of_raw: dict[int, int] = {}
    frames_of_raw: dict[int, set[int]] = {}
    for no, dets in dets_by_track.items():
        for d in dets:
            track_of_raw.setdefault(d.raw_track_id, no)
            frames_of_raw.setdefault(d.raw_track_id, set()).add(d.video_ts_ms)

    def boxes_at(ts: int) -> dict[int, tuple[float, float, float, float]]:
        """Nearest box per RAW fragment at ts (chimeric merged tracks hold
        several co-present fragments, so per-merged-track lookup drops boxes)."""
        out: dict[int, tuple[float, float, float, float]] = {}
        best_dt: dict[int, int] = {}
        for dets in dets_by_track.values():
            for d in dets:
                dt = abs(d.video_ts_ms - ts)
                if dt <= _TS_TOLERANCE_MS and dt < best_dt.get(d.raw_track_id, 1 << 60):
                    best_dt[d.raw_track_id] = dt
                    b = d.bbox
                    out[d.raw_track_id] = (b["x"], b["y"], b["w"], b["h"])
        return out

    n = max(1, min(s.vlm_frames, len(all_ts)))
    lo, hi = duration_ms * 0.1, duration_ms * 0.9
    targets = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]

    shots: list[tuple[int, str]] = []
    boxes_by_ts: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    try:
        cap = cv2.VideoCapture(video_path)
        try:
            for tgt in targets:
                stored = min(all_ts, key=lambda m: abs(m - tgt))
                if stored in boxes_by_ts:  # dedupe: one call per distinct frame
                    continue
                boxes = boxes_at(stored)
                if not boxes:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, float(stored))
                ok, frame = cap.read()
                if not ok:
                    continue
                boxes_by_ts[stored] = boxes
                shots.append((stored, _b64_jpeg(frame)))
        finally:
            cap.release()
    except Exception as exc:
        logger.warning("teacher-id frame decode failed: %s", exc)
        return None
    if not shots:
        return None

    votes: Counter = Counter()
    answered = 0
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        with ThreadPoolExecutor(max_workers=min(_CONCURRENCY, len(shots))) as pool:
            futures = {
                pool.submit(_ask, client, s.gemini_api_key, s.vlm_model, b64): ts
                for ts, b64 in shots
            }
            for fut in as_completed(futures):
                ts = futures[fut]
                try:
                    pt = fut.result()
                except Exception:
                    pt = None
                if pt is None:
                    continue
                answered += 1
                frag = _point_to_fragment(pt[0], pt[1], boxes_by_ts[ts])
                if frag is not None:
                    votes[frag] += 1

    # Votes are cast per raw fragment; report them per merged track so the
    # caller's reject check ("did the geometric pick get ANY vote?") sees votes
    # landing on any of that identity's fragments.
    track_votes: Counter = Counter()
    for raw, count in votes.items():
        track_votes[track_of_raw[raw]] += count
    if not votes:
        return TeacherVote(None, 0.0, {}, answered)
    pooled = pool_fragments(votes, frames_of_raw)
    pooled_votes = sum(votes[r] for r in pooled)
    if pooled_votes < s.vlm_min_votes:
        logger.info(
            "teacher-id inconclusive: votes=%s (need >=%d, answered=%d)",
            dict(track_votes), s.vlm_min_votes, answered,
        )
        return TeacherVote(None, 0.0, dict(track_votes), answered)
    # Crown the merged identity already holding most of the pooled timeline.
    weight: Counter = Counter()
    for raw in pooled:
        weight[track_of_raw[raw]] += len(frames_of_raw[raw])
    track_no = weight.most_common(1)[0][0]
    confidence = round(pooled_votes / max(1, answered), 3)
    logger.info(
        "teacher-id: track %d via fragments %s (%d/%d pooled votes, conf %.2f)",
        track_no, sorted(pooled), pooled_votes, answered, confidence,
    )
    return TeacherVote(track_no, confidence, dict(track_votes), answered)


def apply_vote(
    vote: Optional[TeacherVote],
    teacher_no: Optional[int],
    roles_map: dict[int, tuple[str, Optional[float]]],
) -> Optional[int]:
    """Fold the vote into roles_map; returns the (possibly new) teacher_no.

    CONFIRM/OVERRIDE when the vote crowned a track; REJECT the geometric pick
    only on positive evidence (answered enough, zero votes for it); otherwise
    keep the geometric pick untouched.
    """
    if vote is None:
        return teacher_no
    s = get_settings()
    votes_by_track = vote.votes
    if vote.track is not None:
        if teacher_no is not None and teacher_no != vote.track:
            roles_map[teacher_no] = ("student", None)
            logger.info("teacher-id OVERRIDE: geometric %s -> %s", teacher_no, vote.track)
        roles_map[vote.track] = ("teacher", vote.confidence)
        return vote.track
    if (
        teacher_no is not None
        and vote.answered >= s.vlm_reject_min_answered
        and votes_by_track.get(teacher_no, 0) == 0
    ):
        logger.info(
            "teacher-id REJECT: geometric %s got 0/%d votes", teacher_no, vote.answered
        )
        roles_map[teacher_no] = ("student", None)
        return None
    return teacher_no
