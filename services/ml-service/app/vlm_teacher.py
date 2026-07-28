"""Vision-LLM teacher-ID fallback.

When the geometric ranker (roles.assign_roles) returns all-unknown, a vision model
identifies the teacher from the semantic cues geometry/pose/appearance all miss —
an adult, not in the school uniform, at the teacher's desk. Validated 2026-07-11 on
the comes-and-sits video that every geometric/appearance method failed on.

Design (learned the hard way):
- LOCATE-then-map, NOT set-of-marks. We send the RAW frame and ask the model for the
  teacher's normalized centre point, then map that point to the nearest track. Drawing
  numbered boxes and asking "which number?" only works with <~10 people; past that,
  lighter models miscount or misread the tiny labels.
- MAJORITY VOTE over several frames absorbs the occasional point that drifts onto a
  neighbour in a crowded frame.

Provider history (2026-07-20): started on Groq/Llama-4-Scout (deprecated by Groq
soon after); Qwen3.6 on Groq rejected (thinking-mode <think> blocks blew the token
budget and it hit persistent rate limits); OpenRouter's free vision tier mostly
returned 404 "no endpoints support image input" despite being listed, and the one
model that worked (Nemotron) only answered ~50% of calls (Nvidia's shared free-tier
worker pool saturates). Landed on Gemini (`gemini-flash-latest`) after a 6-frame
production-pattern re-test went 6/6 correct with zero failures. `gemini-2.5-flash`
and `-lite` are ALREADY 404 for new API keys despite still being listed in the
model catalog, which is why we pin to the `-latest` alias, not a dated version:
Google keeps it pointed at whatever their current stable Flash model is, so a
future deprecation like this one won't need a code change.

Fail-closed: any missing key / disabled flag / error / inconclusive vote returns None,
and the caller keeps the all-unknown result rather than inventing a teacher.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Optional

import cv2
import httpx

from app import detector
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
# a track's box centre must be within this (normalized) of the queried time to
# count as "present in that frame" for the point->track mapping.
_TS_TOLERANCE_MS = 400


class TeacherVerdict(NamedTuple):
    """Result of locate_teacher — always returned (never None), so the caller can
    tell "the VLM answered but never pointed at X" (reject X) apart from "the VLM
    could not answer" (keep the geometric pick).

    track:      winning track_no if a track cleared vlm_min_votes, else None.
    confidence: winning votes / answered (0.0 when no winner).
    votes:      per-track point-in-bbox tally (may be non-empty even when track is
                None, e.g. votes scattered below the threshold).
    answered:   number of sampled frames the VLM actually returned a point for
                (excludes network/decode failures). 0 means the VLM never answered.
    """

    track: Optional[int]
    confidence: float
    votes: dict
    answered: int
    # Raw tracker fragments pooled as the same person (see pool_fragments). The
    # caller folds these into the teacher identity so the chain starts holding
    # her whole timeline instead of rediscovering it fragment by fragment.
    fragments: list = []


def _b64_jpeg(frame, max_w: int = 2000) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, round(h * max_w / w)))
    _ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def _strip_code_fence(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` — strip the fence before parsing."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", text)


def _parse_answer(content: str) -> tuple[str, Optional[tuple[float, float]]]:
    """Interpret the VLM's reply:
      ("point", (x, y)) - a teacher location;
      ("absent", None)  - the model explicitly reported NO teacher ({"x":null,...});
      ("none",  None)   - nothing parseable (no usable signal).

    Distinguishing 'absent' from 'none' is what makes claim-verify work: an
    explicit "no teacher in this frame" is positive EVIDENCE that a fragment
    present in that frame is not the teacher, whereas 'none' (a failed/garbled
    reply) carries no signal at all.
    """
    content = _strip_code_fence(content)
    m = re.findall(r'\{[^{}]*"x"[^{}]*\}', content, re.S)
    if not m:
        return ("none", None)
    try:
        j = json.loads(m[-1])
    except json.JSONDecodeError:
        return ("none", None)
    if j.get("x") is None or j.get("y") is None:
        return ("absent", None)
    try:
        return ("point", (float(j["x"]), float(j["y"])))
    except (TypeError, ValueError):
        return ("none", None)


def _parse_point(content: str) -> Optional[tuple[float, float]]:
    status, pt = _parse_answer(content)
    return pt if status == "point" else None


# Per-call ceiling. Generous on purpose: a DROPPED answer is not free. The
# teacher's votes split across her own fragments (she is tracked as ~10 ids), so
# the winner often leads on a third of the votes, and losing a couple of answers
# flips which fragment seeds the chain — measured as a swing from 91% to 62%
# teacher coverage. Concurrency, not a short timeout, is what keeps the derive
# inside the API's 120s budget: the worst case is one timeout, not one per frame.
_HTTP_TIMEOUT = 45.0
# Frames are independent questions, so they go out together. Kept modest so a
# burst does not trip provider rate limits, which would cost answers again.
_VLM_CONCURRENCY = 4


# Per-derive call budget. The ml-service runs one analysis at a time, so a
# module-level counter is enough; begin_derive resets it at the top of each
# derive. Exhaustion is not an error — unspent questions simply report "fail",
# which every caller reads as no signal and keeps what it already had.
_calls_used = 0
_calls_budget = 0


# Resolved media for the current derive. Every helper below needs the video as
# a LOCAL file, but on a remote worker video_path is a presigned URL, so calling
# detector.resolve_video_source per helper downloads the whole file per call —
# measured at 886 MB each over an SSH tunnel, which stalls a long lesson for an
# hour. Resolve once, share it, and release at the end of the derive.
_media_cache: dict[str, tuple[str, bool]] = {}


def release_media() -> None:
    """No-op kept for call-site clarity.

    The service-wide cache in detector owns the downloaded file now and evicts
    it when another video needs the slot, so a derive must not delete it — the
    zone endpoints and the next analyze reuse the same copy.
    """
    _media_cache.clear()


def _resolve_cached(video_path: str) -> str:
    """Local path for video_path, downloading at most once per derive."""
    hit = _media_cache.get(video_path)
    if hit is None:
        hit = (detector.resolve_video_cached(video_path), False)
        _media_cache[video_path] = hit
    return hit[0]


def begin_derive(budget: int) -> None:
    """Start a fresh vision-model budget for one derive. 0 disables the cap."""
    global _calls_used, _calls_budget
    release_media()
    _calls_used, _calls_budget = 0, max(0, budget)


def raise_budget(budget: int) -> None:
    """Lift the cap for a later stage without resetting what has been spent.

    Verify/veto/trim run first and are per-claim, so on a heavily fragmented
    lesson they can spend the whole allowance before gap-filling asks its first
    question — measured: the three earliest holes filled, then 140 calls were
    gone and a 447-second hole got 22 probes, all skipped. Holding those stages
    to a share of the budget keeps the most valuable work fundable.
    """
    global _calls_budget
    if budget > _calls_budget:
        _calls_budget = budget


def calls_used() -> int:
    return _calls_used


def _take(n: int) -> int:
    """Reserve up to n calls, returning how many the budget allows."""
    global _calls_used
    if not _calls_budget:
        _calls_used += n
        return n
    allowed = max(0, min(n, _calls_budget - _calls_used))
    _calls_used += allowed
    return allowed


def _ask_batch(
    key: str, model: str, frames: list[tuple[int, str]]
) -> dict[int, tuple[str, Optional[tuple[float, float]]]]:
    """Ask about many frames CONCURRENTLY -> {tag: (status, point)}.

    Frames must already be decoded: cv2.VideoCapture is not thread-safe, so the
    caller reads sequentially and only the HTTP round-trips fan out. A call that
    raises is recorded as "fail", which every caller treats as no signal.
    """
    out: dict[int, tuple[str, Optional[tuple[float, float]]]] = {}
    if not frames:
        return out
    allowed = _take(len(frames))
    if allowed < len(frames):
        logger.info(
            "VLM budget: %d of %d frames skipped (%d calls used)",
            len(frames) - allowed,
            len(frames),
            _calls_used,
        )
        for tag, _b64 in frames[allowed:]:
            out[tag] = ("fail", None)  # unasked reads as no signal
        frames = frames[:allowed]
    if not frames:
        return out
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        with ThreadPoolExecutor(max_workers=min(_VLM_CONCURRENCY, len(frames))) as pool:
            futures = {
                pool.submit(_ask_answer, client, key, model, b64): tag
                for tag, b64 in frames
            }
            for fut in as_completed(futures):
                tag = futures[fut]
                try:
                    out[tag] = fut.result()
                except Exception as exc:  # never let one frame sink the batch
                    logger.warning("VLM batch call failed: %s", exc)
                    out[tag] = ("fail", None)
    return out


def _ask_answer(
    client: httpx.Client, key: str, model: str, b64: str
) -> tuple[str, Optional[tuple[float, float]]]:
    """POST one frame and interpret the reply as ("point", (x,y)) / ("absent",
    None) / ("fail", None). "fail" is a network / HTTP / response-shape error
    (no signal); "absent" is the model explicitly reporting no teacher; "none"
    (unparseable 200) is folded into "fail" for the caller's purposes here — it
    also carries no signal. Retries on rate-limit.
    """
    url = _ENDPOINT_TMPL.format(model=model)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        # NOTE: gemini-flash-latest rejects a generationConfig.thinkingConfig with
        # HTTP 400 INVALID_ARGUMENT, so we do NOT set it here (a thinkingBudget=0
        # would be nice to stop empty MAX_TOKENS answers, but it breaks the call on
        # this alias — revisit if the alias moves to a model that accepts it).
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2000},
    }
    for attempt in range(3):
        try:
            r = client.post(url, params={"key": key}, json=payload)
        except Exception as exc:  # network error — no signal
            logger.warning("VLM request error: %s", exc)
            return ("fail", None)
        if r.status_code == 429:  # rate limited — back off and retry
            time.sleep(4 * (attempt + 1))
            continue
        if r.status_code != 200:
            logger.warning("VLM HTTP %s: %s", r.status_code, r.text[:200])
            return ("fail", None)
        body = r.json()
        try:
            parts = body["candidates"][0]["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            logger.warning("VLM unexpected response shape: %s", json.dumps(body)[:200])
            return ("fail", None)
        return _parse_answer(content)  # "point" / "absent" / "none"
    return ("fail", None)  # exhausted retries (persistent 429)


def _ask_point(
    client: httpx.Client, key: str, model: str, b64: str
) -> Optional[tuple[float, float]]:
    """Return the teacher's (x, y) in 0..1, or None (absent / failed / garbled).
    Thin wrapper over _ask_answer for callers that only need the point."""
    _status, pt = _ask_answer(client, key, model, b64)
    return pt


def identify_teacher(
    video_path: str,
    dets_by_track: dict[int, list[Detection]],
    duration_ms: int,
) -> Optional[tuple[int, float, dict]]:
    """Identify the teacher track via the vision model, or None.

    Samples settings.vlm_frames frames evenly across the middle of the lesson,
    asks the model to point at the teacher on each, maps each point to the nearest
    track's bbox centre at that frame, and majority-votes. Returns
    (track_no, confidence, vote_counts) where confidence = winning_votes /
    frames_that_answered, or None when disabled / no key / no answers / the winner
    falls short of settings.vlm_min_votes / on any error (fail-closed).
    """
    s = get_settings()
    if (
        not s.vlm_teacher_fallback
        or not s.gemini_api_key
        or not video_path
        or not dets_by_track
        or duration_ms <= 0
    ):
        return None

    all_ts = sorted({d.video_ts_ms for dets in dets_by_track.values() for d in dets})
    if not all_ts:
        return None
    n = max(1, min(s.vlm_frames, len(all_ts)))
    # spread the samples across [10%, 90%] of the timeline (skip the pre/post-class
    # edges where the room is empty or in flux).
    lo, hi = duration_ms * 0.1, duration_ms * 0.9
    targets = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]

    def centers_at(ts: int) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for no, dets in dets_by_track.items():
            d = min(dets, key=lambda d: abs(d.video_ts_ms - ts))
            if abs(d.video_ts_ms - ts) <= _TS_TOLERANCE_MS:
                out[no] = (d.bbox["x"] + d.bbox["w"] / 2.0, d.bbox["y"] + d.bbox["h"] / 2.0)
        return out

    local_path = _resolve_cached(video_path)
    votes: Counter = Counter()
    answered = 0
    try:
        cap = cv2.VideoCapture(local_path)
        try:
            with httpx.Client(timeout=60) as client:
                for tgt in targets:
                    stored = min(all_ts, key=lambda m: abs(m - tgt))
                    centers = centers_at(stored)
                    if not centers:
                        continue
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(stored))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    pt = _ask_point(client, s.gemini_api_key, s.vlm_model, _b64_jpeg(frame))
                    if pt is None:
                        continue
                    answered += 1
                    px, py = pt
                    nearest = min(
                        centers,
                        key=lambda t: (centers[t][0] - px) ** 2 + (centers[t][1] - py) ** 2,
                    )
                    votes[nearest] += 1
        finally:
            cap.release()
    except Exception as exc:  # any decode/network failure -> fail closed
        logger.warning("VLM teacher-id failed: %s", exc)
        return None

    if not votes:
        return None
    track_no, win = votes.most_common(1)[0]
    if win < s.vlm_min_votes:
        logger.info(
            "VLM teacher-id inconclusive: votes=%s (need >=%d)",
            dict(votes),
            s.vlm_min_votes,
        )
        return None
    confidence = round(win / max(1, answered), 3)
    logger.info(
        "VLM teacher-id: track %d (%d/%d votes, confidence %.2f)",
        track_no,
        win,
        answered,
        confidence,
    )
    return track_no, confidence, dict(votes)


# Two fragments sharing at least this many sampled frames are two people: one
# person cannot be two boxes at once. Below it, brief overlap is tracker noise
# at a handoff (a dying id and its successor can share a frame or two).
_CO_PRESENT_MIN = 3


def pool_fragments(votes: Counter, frames_of_raw: dict[int, set[int]]) -> list[int]:
    """Group the vote-getting fragments that can be the SAME person.

    The teacher is tracked as many raw ids, so her votes split between them and
    the leader can win on a third of them — which makes the pick a coin flip
    that swung teacher coverage from 91% to 62% when a couple of answers were
    lost. Every one of those votes is the same correct answer ("she is here"),
    so they should be counted together rather than against each other.

    Co-presence is the veto: fragments appearing in the same frames are
    different people. Taking the best-supported fragment first and admitting
    only fragments non-co-present with everything already pooled gives the
    largest defensible group. Only fragments the model actually pointed at are
    eligible, so this never speculates that some unseen track might be her.
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


def _point_to_track(
    px: float, py: float, boxes: dict[int, tuple[float, float, float, float]]
) -> Optional[int]:
    """Map a VLM point to the detection whose bbox CONTAINS it.

    Point-in-bbox, NOT nearest-centre: the model points at the teacher's body, and
    nearest-centre snaps onto a closer neighbour in a crowd (that is how a student
    gets "confirmed"). Among boxes that contain the point, prefer the one the point
    sits most centrally in (distance to box centre / box size) — a standing adult's
    torso is central to her tall box but near the top edge of a foreground pupil's
    box. Returns None when no box contains the point (model pointed at empty floor
    / between people), so an ambiguous frame casts NO vote.
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


def locate_teacher(
    video_path: str,
    dets_by_track: dict[int, list[Detection]],
    duration_ms: int,
    n_frames: Optional[int] = None,
) -> TeacherVerdict:
    """Locate the teacher by POINT-IN-BBOX voting — the authority on *who* she is.

    Like identify_teacher but (a) maps each VLM point to the detection that
    CONTAINS it (see _point_to_track) instead of the nearest centre, and (b)
    samples more frames so an occluded frame doesn't sink the vote. Used to
    CONFIRM / OVERRIDE / REJECT the geometric pick: whoever the model actually
    points at wins, and a track it never points at can't become teacher.

    Always returns a TeacherVerdict (never None) so the caller can distinguish a
    real "not the teacher" (VLM answered N frames, never pointed at the pick) from
    "the VLM couldn't answer" (answered==0). track is None when no track cleared
    vlm_min_votes. Fail-closed: missing key / no boxes / error -> answered 0.
    NOT gated on vlm_teacher_fallback — the caller gates it on vlm_verify_teacher.
    """
    s = get_settings()
    if not s.gemini_api_key or not video_path or not dets_by_track or duration_ms <= 0:
        return TeacherVerdict(None, 0.0, {}, 0)
    all_ts = sorted({d.video_ts_ms for dets in dets_by_track.values() for d in dets})
    if not all_ts:
        return TeacherVerdict(None, 0.0, {}, 0)
    # Sample density follows the lesson: a fixed dozen frames is plenty for a
    # five-minute clip but one every five minutes across an hour, too sparse to
    # locate a teacher who is regularly occluded.
    want = n_frames or min(
        s.vlm_verify_frames_max,
        max(s.vlm_verify_frames, round(duration_ms / 60_000 * 0.4)),
    )
    n = max(1, min(want, len(all_ts)))
    lo, hi = duration_ms * 0.1, duration_ms * 0.9
    targets = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]

    # Vote on RAW tracker fragments, not merged tracks. A merged identity can be
    # chimeric — measured here, the four vote-getting tracks were all co-present
    # with one another, so each held several people and pooling them was
    # impossible. Raw fragments are one person by construction, so hers are
    # mutually non-co-present and can be recognised as the same teacher.
    track_of_raw: dict[int, int] = {}
    frames_of_raw: dict[int, set[int]] = {}
    for no, dets in dets_by_track.items():
        for d in dets:
            track_of_raw.setdefault(d.raw_track_id, no)
            frames_of_raw.setdefault(d.raw_track_id, set()).add(d.video_ts_ms)

    def boxes_at(ts: int) -> dict[int, tuple[float, float, float, float]]:
        """Raw-fragment boxes visible at ts, keyed by raw_track_id."""
        out: dict[int, tuple[float, float, float, float]] = {}
        for dets in dets_by_track.values():
            d = min(dets, key=lambda d: abs(d.video_ts_ms - ts))
            if abs(d.video_ts_ms - ts) <= _TS_TOLERANCE_MS:
                out[d.raw_track_id] = (
                    d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"]
                )
        return out

    local_path = _resolve_cached(video_path)
    votes: Counter = Counter()
    answered = 0
    try:
        # Decode first (cv2 is not thread-safe), then ask about every frame at once.
        shots: list[tuple[int, str]] = []
        boxes_by_ts: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        cap = cv2.VideoCapture(local_path)
        try:
            for tgt in targets:
                stored = min(all_ts, key=lambda m: abs(m - tgt))
                if stored in boxes_by_ts:
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
        for stored, (status, pt) in _ask_batch(
            s.gemini_api_key, s.vlm_model, shots
        ).items():
            if status != "point":
                continue
            answered += 1
            track = _point_to_track(pt[0], pt[1], boxes_by_ts[stored])
            if track is not None:
                votes[track] += 1
    except Exception as exc:  # any decode/network failure -> fail closed
        logger.warning("VLM locate-teacher failed: %s", exc)
        return TeacherVerdict(None, 0.0, dict(votes), answered)

    if not votes:
        logger.info(
            "VLM locate-teacher: no confident point-in-bbox votes (answered=%d)", answered
        )
        return TeacherVerdict(None, 0.0, {}, answered)
    # Pool the fragments that can be one person, so her own ids stop competing.
    pool = pool_fragments(votes, frames_of_raw)
    pooled_votes = sum(votes[r] for r in pool)
    if pooled_votes < s.vlm_min_votes:
        logger.info(
            "VLM locate-teacher inconclusive: votes=%s (need >=%d, answered=%d)",
            dict(votes),
            s.vlm_min_votes,
            answered,
        )
        return TeacherVerdict(None, 0.0, dict(votes), answered)
    # The identity to crown is the one already holding most of her pooled frames.
    weight: Counter = Counter()
    for raw in pool:
        weight[track_of_raw[raw]] += len(frames_of_raw[raw])
    track_no = weight.most_common(1)[0][0]
    confidence = round(pooled_votes / max(1, answered), 3)
    logger.info(
        "VLM locate-teacher: track %d via fragments %s (%d/%d pooled votes, "
        "conf %.2f; unpooled leader had %d)",
        track_no,
        sorted(pool),
        pooled_votes,
        answered,
        confidence,
        votes.most_common(1)[0][1],
    )
    return TeacherVerdict(track_no, confidence, dict(votes), answered, sorted(pool))


# A claim is judged only if the VLM gave at least this many usable answers (a
# point OR an explicit "no teacher") across its span. Below this we couldn't look
# enough to be sure, so we keep the claim (fail-open — never drop a real reclaim
# on thin data).
_CLAIM_VERIFY_MIN_ANSWERED = 3
# Veto the claim when the teacher lands inside the fragment's box in fewer than
# this fraction of judged frames. A genuine reclaim (the fragment IS the teacher)
# scores near 1.0; a student fragment the stitcher chained onto her scores near 0.
# The margin tolerates a few handoff-zone frames where a student's box has already
# drifted onto the entering teacher (that late overlap must not rescue the claim).
_CLAIM_SUPPORT_FRAC = 0.34


def vlm_supports_span(
    video_path: str,
    dets: list[Detection],
    n_frames: Optional[int] = None,
) -> Optional[bool]:
    """Is the teacher actually INSIDE these detections' boxes across their span?

    Verifies one teacher-chain CLAIM before it is applied. The stitcher can fold a
    student fragment that ends where the teacher's tracked chain begins into the
    teacher (a backward claim that passes the height-rise test) — the chimera seen
    as "student labelled teacher until the real teacher comes". We sample frames
    across the fragment's span and, for each, ask the VLM where the teacher is:
      - point INSIDE the fragment's box  -> support (it IS her here);
      - point OUTSIDE, or an explicit "no teacher visible" -> refute (she is
        elsewhere / not in the room, so this fragment is not her here);
      - failed/garbled reply             -> no signal.
    Counting an explicit "no teacher" as refutation is essential: a backward-
    claimed student is often chained across frames where the teacher hasn't
    entered yet, and the VLM correctly returns null there.

    Returns:
      True  - the teacher is inside the box in >= _CLAIM_SUPPORT_FRAC of judged
              frames -> a real reclaim; keep the claim.
      False - judged >= _CLAIM_VERIFY_MIN_ANSWERED frames and support fraction is
              below the floor -> not the teacher; the caller should VETO the claim.
      None  - disabled / no key / too few judged frames / decode error -> keep the
              claim (fail-open, so a failure never drops a genuine reclaim).
    """
    s = get_settings()
    if not s.gemini_api_key or not video_path or not dets:
        return None
    dets = sorted(dets, key=lambda d: d.video_ts_ms)
    t0, t1 = dets[0].video_ts_ms, dets[-1].video_ts_ms
    n = max(1, min(n_frames or s.vlm_claim_verify_frames, len(dets)))
    targets = (
        [t0 + (t1 - t0) * i / max(1, n - 1) for i in range(n)] if t1 > t0 else [t0]
    )

    def box_at(ts: int):
        d = min(dets, key=lambda d: abs(d.video_ts_ms - ts))
        if abs(d.video_ts_ms - ts) <= _TS_TOLERANCE_MS:
            b = d.bbox
            return d.video_ts_ms, (b["x"], b["y"], b["w"], b["h"])
        return None

    local_path = _resolve_cached(video_path)
    support = refute = 0
    try:
        shots: list[tuple[int, str]] = []
        box_by_ts: dict[int, tuple[float, float, float, float]] = {}
        cap = cv2.VideoCapture(local_path)
        try:
            for tgt in targets:
                got = box_at(int(tgt))
                if got is None:
                    continue
                stored, box = got
                if stored in box_by_ts:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, float(stored))
                ok, frame = cap.read()
                if not ok:
                    continue
                box_by_ts[stored] = box
                shots.append((stored, _b64_jpeg(frame)))
        finally:
            cap.release()
        for stored, (status, pt) in _ask_batch(
            s.gemini_api_key, s.vlm_model, shots
        ).items():
            if status == "point":
                x, y, w, h = box_by_ts[stored]
                px, py = pt
                if x <= px <= x + w and y <= py <= y + h:
                    support += 1
                else:
                    refute += 1  # teacher is someone else here
            elif status == "absent":
                refute += 1  # VLM says no teacher in this frame at all
            # "fail"/"none": no usable signal -> ignore
    except Exception as exc:  # any decode/network failure -> keep the claim
        logger.warning("VLM claim-verify failed: %s", exc)
        return None

    judged = support + refute
    if judged < _CLAIM_VERIFY_MIN_ANSWERED:
        return None  # too few looks to be sure -> keep
    frac = support / judged
    if frac < _CLAIM_SUPPORT_FRAC:
        logger.info(
            "VLM claim-verify: span %d-%dms support %.2f (%d in / %d judged) -> VETO",
            t0,
            t1,
            frac,
            support,
            judged,
        )
        return False
    return True


# --- entry backfill ---------------------------------------------------------
_BACKFILL_STEP_MS = 1000  # sample cadence walking backward from the teacher's start
_BACKFILL_MAX_WINDOW_MS = 90_000  # never look back further than this (safety cap)
_BACKFILL_ABSENT_STOP = 2  # consecutive "no teacher" replies => she wasn't in the room yet
_BACKFILL_MAX_MOVE = 0.22  # max normalized centre move between steps (she can't teleport)
_BACKFILL_NEAR_DET = 0.12  # a detection must be within this of the VLM point to be "her"
_BACKFILL_MIN_ANCHORS = 2  # need >= this many continuous anchors to trust the trace


def _bbox_center(b: dict) -> tuple[float, float]:
    return (b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0)


# --- long-hole anchoring ------------------------------------------------------
# How often to ask "where is she?" inside a long hole. Every step costs one call,
# so this trades resolution for budget: 20s over a 7-minute hole is ~22 calls.
LONG_HOLE_STEP_MS = 20_000
# How much timeline a single answer speaks for when we cannot do better (no
# height model). A clock is the wrong unit — its edges fall wherever 10 seconds
# happens to land, which is mid-stride through a body, and the label changes
# there for no physical reason. See _anchor_segment for the unit actually used.
LONG_HOLE_CLAIM_MS = LONG_HOLE_STEP_MS

# --- what one answer speaks for -------------------------------------------
# An answer identifies a BODY at an instant, so it should speak for as much of
# that body's continuous trajectory as remains recognisable — bounded by the
# raw tracker fragment, since that is the only place continuity is even
# defined. It must NOT speak for the whole fragment: BoT-SORT's box migrates,
# ballooning over 1-3 frames off a seated child onto the teacher as she walks
# past, so a fragment can hold two people (raw1967 is a child for 1718
# detections and her for 51). Whole-fragment claiming was measured at ~5700
# student detections handed to the teacher.
#
# The stop rules are calibrated, not guessed. Seeding at all 299 teacher-side
# detections of the six known migrating fragments and asking how many runs
# reach the other person:
#   sample-gap alone .................. 271-276 of 299 cross (up to 935 dets)
#   + height ratio .....................  38 of 299, worst 1 detection
#   + height band ......................   0 of 299
# The gap rule alone is therefore useless as a safety property; the height
# gates are load-bearing. A one-frame occlusion must not cut a run, so a few
# failing samples are bridged, but a SUSTAINED failure ends it.
# That zero-crossing result was measured seeding on detections that are
# SOLIDLY hers (ratio >= 0.80), and it only holds for those. Seeding anywhere
# teacher-ish was measured adding 86 student detections, because a seed part
# way up a migration ramp grows in the wrong direction. So the seed is judged
# strictly and the growth leniently, and a seed that fails simply falls back to
# the old fixed window — never worse than before it, better where it is sure.
ANCHOR_MAX_GAP_MS = 2_000  # a longer silence is not one continuous trajectory
ANCHOR_SEED_RATIO = 0.80  # a seed must be unambiguously teacher-sized
ANCHOR_MIN_RATIO = 0.60  # ...but the run may follow her down to here
ANCHOR_H_BAND = 0.15  # |h - running median| a body may vary within a run
ANCHOR_MISS_TOLERANCE = 2  # consecutive failing samples bridged, not crossed


def _anchor_segment(
    frag: list[Detection],
    seed_idx: int,
    model,
    lo_ms: int,
    hi_ms: int,
) -> list[Detection]:
    """One body's continuous run around frag[seed_idx], inside its own fragment.

    `frag` is a single raw tracker id's detections, ts-sorted; `model` answers
    .ratio(det) = observed height / height expected of the teacher at that
    depth. Returns [] when the seed itself is not teacher-sized, which is the
    correct refusal: at one real anchor point the model pointed into a corner
    child's box, and claiming a clock-window around it would have labelled him.
    """
    if model is None or not frag:
        return []
    if model.ratio(frag[seed_idx]) < ANCHOR_SEED_RATIO:
        return []
    keep = [seed_idx]
    heights = [frag[seed_idx].bbox["h"]]
    for step in (-1, 1):
        pending: list[int] = []
        i = seed_idx + step
        while 0 <= i < len(frag):
            d = frag[i]
            if not lo_ms <= d.video_ts_ms <= hi_ms:
                break
            if abs(d.video_ts_ms - frag[i - step].video_ts_ms) > ANCHOR_MAX_GAP_MS:
                break
            median = sorted(heights)[len(heights) // 2]
            if (
                model.ratio(d) >= ANCHOR_MIN_RATIO
                and abs(d.bbox["h"] - median) <= ANCHOR_H_BAND
            ):
                keep.extend(pending)  # bridge the occlusion we walked over
                keep.append(i)
                heights.append(d.bbox["h"])
                pending = []
            else:
                pending.append(i)
                if len(pending) > ANCHOR_MISS_TOLERANCE:
                    break  # sustained: this is where the body stops being hers
            i += step
    return [frag[i] for i in sorted(keep)]


def anchor_teacher_in_window(
    video_path: str,
    dets_by_track: dict[int, list[Detection]],
    teacher_no: int,
    start_ms: int,
    end_ms: int,
    claimed_ts: Optional[set[int]] = None,
    step_ms: int = LONG_HOLE_STEP_MS,
    height_model=None,
) -> list[Detection]:
    """Find the teacher inside a long hole and return the detections to claim.

    A long hole is not one situation. Measured on a 37-minute lesson: inside a
    single 447-second gap she was out of the room at one point and tracked under
    three DIFFERENT student ids at others. Asking whether one candidate track
    explains the whole window — what gap_fill_candidates does — cannot succeed
    there, and the whole gap stays unlabelled.

    So ask per sub-window instead: step through the hole, ask the model where she
    is, and take her from whichever track holds her at that moment. Each answer
    speaks only for LONG_HOLE_CLAIM_MS around itself. An explicit "no teacher"
    claims nothing, which is the right answer — she really does leave the room,
    and those stretches SHOULD stay unlabelled.
    """
    s = get_settings()
    if not s.gemini_api_key or not video_path or end_ms <= start_ms:
        return []
    taken = set(claimed_ts or ())

    by_ts: dict[int, list[Detection]] = {}
    for no, dets in dets_by_track.items():
        if no == teacher_no:
            continue
        for d in dets:
            if start_ms <= d.video_ts_ms <= end_ms:
                by_ts.setdefault(d.video_ts_ms, []).append(d)
    if not by_ts:
        return []
    stored = sorted(by_ts)
    # Continuity is a property of the RAW tracker fragment, not of the merged
    # identity: a merged track can hold several fragments, so claiming "this
    # track, for 20 seconds" can pull in more than one person at once.
    by_raw: dict[int, list[Detection]] = {}
    for dets in dets_by_track.values():
        for d in dets:
            by_raw.setdefault(d.raw_track_id, []).append(d)
    for lst in by_raw.values():
        lst.sort(key=lambda d: d.video_ts_ms)

    local_path = _resolve_cached(video_path)
    shots: list[tuple[int, str]] = []
    cap = cv2.VideoCapture(local_path)
    try:
        t = start_ms + step_ms // 2
        while t < end_ms:
            ts = min(stored, key=lambda m: abs(m - t))
            if abs(ts - t) <= step_ms // 2 and ts not in dict(shots):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
                ok, frame = cap.read()
                if ok:
                    shots.append((ts, _b64_jpeg(frame)))
            t += step_ms
    finally:
        cap.release()

    answers = _ask_batch(s.gemini_api_key, s.vlm_model, shots)
    out: list[Detection] = []
    absent = anchored = 0
    for ts, (status, pt) in sorted(answers.items()):
        if status == "absent":
            absent += 1
            continue
        if status != "point":
            continue
        here = by_ts.get(ts, [])
        boxes = {
            i: (d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"])
            for i, d in enumerate(here)
        }
        idx = _point_to_track(pt[0], pt[1], boxes)
        if idx is None:
            # She is visible and the model found her, but no box is under the
            # point. Measured on a 37-minute lesson: 16 of 50 probes, 310.8s of
            # timeline. Every time, the nearest boxes were seated children
            # (height ratio 0.23-0.57 against her 0.80+) -- she is simply not
            # DETECTED in those frames, so there is nothing here to claim and
            # loosening the point test would only ever label a child. The lever
            # is detector recall (imgsz, sampling rate), not this mapping.
            continue
        seed = here[idx]
        frag = by_raw.get(seed.raw_track_id, [])
        at = next(
            (i for i, d in enumerate(frag) if d.video_ts_ms == seed.video_ts_ms), None
        )
        span = (
            _anchor_segment(frag, at, height_model, start_ms, end_ms)
            if at is not None
            else []
        )
        if not span:
            # No model, or a seed too doubtful to grow from. Fall back to the
            # old fixed window: its edges are arbitrary, but it is what shipped,
            # so this can only ever place boundaries better, never claim less.
            # Scoped to the fragment, not the merged track: a merged track can
            # hold two fragments alive at once, and claiming by track put two
            # boxes on her at the same instant (probe 130.0s claimed raw290 and
            # raw278 for 130.4s). One body per answer.
            lo, hi = ts - LONG_HOLE_CLAIM_MS // 2, ts + LONG_HOLE_CLAIM_MS // 2
            span = [d for d in frag if lo <= d.video_ts_ms <= hi]
        got = [
            d
            for d in span
            if start_ms <= d.video_ts_ms <= end_ms and d.video_ts_ms not in taken
        ]
        if got:
            anchored += 1
            taken.update(d.video_ts_ms for d in got)
            out.extend(got)
    if shots:
        logger.info(
            "VLM long-hole %.1f-%.1fs: %d probes -> %d anchors, %d absent, %d dets claimed",
            start_ms / 1000,
            end_ms / 1000,
            len(shots),
            anchored,
            absent,
            len(out),
        )
    return out


def backfill_teacher_entry(
    video_path: str,
    dets_by_track: dict[int, list[Detection]],
    teacher_no: int,
    step_ms: int = _BACKFILL_STEP_MS,
) -> list[Detection]:
    """Reclaim the teacher's PRE-CHAIN entry/seated detections into her identity.

    The teacher-chain seeds on her MOBILE (teaching) phase and its backward-claim
    loop skips seated/static fragments, so a teacher who enters and sits before
    teaching stays labelled student ("student until the heuristics kick in"). On
    real footage that entry is catastrophically fragmented (~7 tracks in ~18s, some
    merge-chimeric), so per-track voting can't tell her fragments from noise — but
    the VLM's per-frame points trace ONE smooth path. So we TRACK her backward: seed
    from her first (chain) detection, and at each step ask the VLM where she is;
    accept the point only if it is spatially CONTINUOUS with the previous one (she
    can't teleport — this is the anti-stray guard), then anchor the raw tracker-
    fragment whose detection is nearest that point. Reclaiming at the RAW-fragment
    level (not the merged track) sidesteps the entry chimeras.

    Bounded: stop once the VLM reports no teacher for _BACKFILL_ABSENT_STOP frames
    running (= before she entered). Returns the detections to reassign to teacher_no
    (empty when disabled / no key / no teacher / the trace never gets going).
    """
    s = get_settings()
    if not s.gemini_api_key or not video_path or teacher_no not in dets_by_track:
        return []
    teacher_dets = sorted(dets_by_track[teacher_no], key=lambda d: d.video_ts_ms)
    if not teacher_dets:
        return []
    T = teacher_dets[0].video_ts_ms
    prev = _bbox_center(teacher_dets[0].bbox)  # seed the trajectory from her chain start

    # non-teacher detections, grouped by timestamp (candidates for "where is she")
    by_ts: dict[int, list[Detection]] = {}
    for no, dets in dets_by_track.items():
        if no == teacher_no:
            continue
        for d in dets:
            by_ts.setdefault(d.video_ts_ms, []).append(d)
    stored_ts = sorted(by_ts)
    if not stored_ts:
        return []

    def dets_at(target: int) -> list[Detection]:
        st = min(stored_ts, key=lambda m: abs(m - target))
        return by_ts[st] if abs(st - target) <= _TS_TOLERANCE_MS else []

    local_path = _resolve_cached(video_path)
    anchored: dict[int, int] = {}  # raw_track_id -> earliest ts we placed her in it
    anchors = 0
    absent = 0
    cap = cv2.VideoCapture(local_path)
    try:
        with httpx.Client(timeout=60) as client:
            t = T - step_ms
            while t >= max(0, T - _BACKFILL_MAX_WINDOW_MS):
                here = dets_at(t)
                if here:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(t))
                    ok, frame = cap.read()
                    if ok:
                        status, pt = _ask_answer(
                            client, s.gemini_api_key, s.vlm_model, _b64_jpeg(frame)
                        )
                        if status == "absent":
                            absent += 1
                            if absent >= _BACKFILL_ABSENT_STOP:
                                break  # she hadn't entered yet -> stop walking back
                        elif status == "point":
                            absent = 0
                            if math.hypot(pt[0] - prev[0], pt[1] - prev[1]) <= _BACKFILL_MAX_MOVE:
                                prev = pt  # trust the VLM point as her position
                                d = min(
                                    here,
                                    key=lambda det: math.hypot(
                                        _bbox_center(det.bbox)[0] - pt[0],
                                        _bbox_center(det.bbox)[1] - pt[1],
                                    ),
                                )
                                cd = _bbox_center(d.bbox)
                                if math.hypot(cd[0] - pt[0], cd[1] - pt[1]) <= _BACKFILL_NEAR_DET:
                                    anchored[d.raw_track_id] = t  # walking back -> ends earliest
                                    anchors += 1
                            # discontinuous point -> stray; skip (don't move prev)
                t -= step_ms
    except Exception as exc:  # decode/network failure -> reclaim nothing
        logger.warning("VLM entry-backfill failed: %s", exc)
        return []
    finally:
        cap.release()

    if anchors < _BACKFILL_MIN_ANCHORS or not anchored:
        return []
    earliest = min(anchored.values())
    out = [
        d
        for dets in dets_by_track.values()
        for d in dets
        if d.raw_track_id in anchored and earliest <= d.video_ts_ms < T
    ]
    logger.info(
        "VLM entry-backfill: reclaimed %d dets from raw fragment(s) %s into teacher %d "
        "(window %.0f-%.0fs, %d anchors)",
        len(out),
        sorted(anchored),
        teacher_no,
        earliest / 1000,
        T / 1000,
        anchors,
    )
    return out
