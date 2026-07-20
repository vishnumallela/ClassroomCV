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
import os
import re
import time
from collections import Counter
from typing import Optional

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


def _b64_jpeg(frame, max_w: int = 2000) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, round(h * max_w / w)))
    _ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def _strip_code_fence(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` — strip the fence before parsing."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", text)


def _parse_point(content: str) -> Optional[tuple[float, float]]:
    content = _strip_code_fence(content)
    m = re.findall(r'\{[^{}]*"x"[^{}]*\}', content, re.S)
    if not m:
        return None
    try:
        j = json.loads(m[-1])
    except json.JSONDecodeError:
        return None
    if j.get("x") is None or j.get("y") is None:
        return None
    try:
        return float(j["x"]), float(j["y"])
    except (TypeError, ValueError):
        return None


def _ask_point(
    client: httpx.Client, key: str, model: str, b64: str
) -> Optional[tuple[float, float]]:
    """Return the teacher's (x, y) in 0..1, or None. Retries on rate-limit."""
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
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2000},
    }
    for attempt in range(3):
        try:
            r = client.post(url, params={"key": key}, json=payload)
        except Exception as exc:  # network error — treat as a missing answer
            logger.warning("VLM request error: %s", exc)
            return None
        if r.status_code == 429:  # rate limited — back off and retry
            time.sleep(4 * (attempt + 1))
            continue
        if r.status_code != 200:
            logger.warning("VLM HTTP %s: %s", r.status_code, r.text[:200])
            return None
        body = r.json()
        try:
            parts = body["candidates"][0]["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            logger.warning("VLM unexpected response shape: %s", json.dumps(body)[:200])
            return None
        return _parse_point(content)
    return None  # exhausted retries (persistent 429)


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

    local_path, is_temp = detector.resolve_video_source(video_path)
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
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except OSError:
                pass

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
