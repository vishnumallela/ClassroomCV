"""Generate tests/assets/classroom_synth.mp4 — synthetic classroom smoke asset.

Scene (1280x720, per the board-detection feature contract):
- light-gray wall background
- dark-green board rectangle spanning x 0.20-0.75 / y 0.12-0.45 (ground truth)
- brown floor strip at the bottom
- a couple of small dark blobs in the lower half (desks)

A single frame is rendered with numpy/cv2, then ffmpeg loops it into a 6 s
720p yuv420p mp4. Deterministic (seeded noise) so the asset is reproducible:

    uv run python tests/assets/make_classroom_synth.py

Ground-truth board bbox (normalized): x0=0.20 y0=0.12 x1=0.75 y1=0.45.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
DURATION_S = 6
FPS = 30

# Normalized ground truth used by the smoke test (x0, y0, x1, y1).
BOARD_GT = (0.20, 0.12, 0.75, 0.45)


def build_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    frame = np.full((HEIGHT, WIDTH, 3), (205, 205, 200), dtype=np.uint8)  # wall

    # Board: dark green rectangle at x 0.2-0.75 / y 0.12-0.45.
    x0, y0, x1, y1 = BOARD_GT
    cv2.rectangle(
        frame,
        (int(x0 * WIDTH), int(y0 * HEIGHT)),
        (int(x1 * WIDTH), int(y1 * HEIGHT)),
        (45, 90, 35),  # BGR dark green
        thickness=-1,
    )

    # Floor: brown strip at the bottom (~y 0.78-1.0).
    cv2.rectangle(
        frame,
        (0, int(0.78 * HEIGHT)),
        (WIDTH, HEIGHT),
        (60, 95, 140),  # BGR brown
        thickness=-1,
    )

    # Desks: a couple of small dark blobs in the lower half.
    for cx, cy, w, h in ((0.30, 0.66, 0.10, 0.06), (0.62, 0.70, 0.11, 0.06)):
        cv2.ellipse(
            frame,
            (int(cx * WIDTH), int(cy * HEIGHT)),
            (int(w * WIDTH / 2), int(h * HEIGHT / 2)),
            0,
            0,
            360,
            (40, 45, 55),  # dark blob
            thickness=-1,
        )

    # Light deterministic sensor noise so the encoder keeps natural gradients.
    noise = rng.normal(0.0, 2.0, frame.shape)
    return np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    out = Path(__file__).resolve().parent / "classroom_synth.mp4"
    frame = build_frame()
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "classroom_frame.png"
        cv2.imwrite(str(png), frame)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(png),
                "-t",
                str(DURATION_S),
                "-r",
                str(FPS),
                "-vf",
                "format=yuv420p",
                str(out),
            ],
            check=True,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
