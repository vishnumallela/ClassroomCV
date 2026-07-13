"""Vision-LLM teacher-ID fallback.

When the geometric ranker (roles.assign_roles) returns all-unknown, a vision model
identifies the teacher from the semantic cues geometry/pose/appearance all miss —
an adult, not in the school uniform, at the teacher's desk. Validated 2026-07-11 on
the comes-and-sits video that every geometric/appearance method failed on.

Design (learned the hard way):
- LOCATE-then-map, NOT set-of-marks. We send the RAW frame and ask the model for the
  teacher's normalized centre point, then map that point to the nearest track. Drawing
  numbered boxes and asking "which number?" only works with <~10 people; past that,
  lighter models miscount or misread the tiny labels (Scout returned null on a 22-box
  frame but pointed at the right person on the same raw frame).
- MAJORITY VOTE over several frames absorbs the occasional point that drifts onto a
  neighbour in a crowded frame.
- Groq sits behind Cloudflare: httpx with a real User-Agent works; urllib's default UA
  gets an HTTP 403 "error code 1010".

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

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
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


def _ask_point(
    client: httpx.Client, key: str, model: str, b64: str
) -> Optional[tuple[float, float]]:
    """Return the teacher's (x, y) in 0..1, or None. Retries on rate-limit."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "classroomcv-ml/1.0",
    }
    for attempt in range(3):
        try:
            r = client.post(_ENDPOINT, json=payload, headers=headers)
        except Exception as exc:  # network error — treat as a missing answer
            logger.warning("VLM request error: %s", exc)
            return None
        if r.status_code == 429:  # rate limited — back off and retry
            time.sleep(4 * (attempt + 1))
            continue
        if r.status_code != 200:
            logger.warning("VLM HTTP %s: %s", r.status_code, r.text[:200])
            return None
        content = r.json()["choices"][0]["message"]["content"]
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
        or not s.groq_api_key
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
                    pt = _ask_point(client, s.groq_api_key, s.vlm_model, _b64_jpeg(frame))
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
