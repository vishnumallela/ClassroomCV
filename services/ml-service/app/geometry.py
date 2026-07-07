"""Small pure-python geometry helpers for normalized (0-1) coordinates."""

from __future__ import annotations

import math


def polygon_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Axis-aligned bbox (x0, y0, x1, y1) of a polygon [[x, y], ...]."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(
    bbox: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return x0 - margin, y0 - margin, x1 + margin, y1 + margin


def bboxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def rdp_indices(points: list[tuple[float, float]], epsilon: float) -> list[int]:
    """Ramer-Douglas-Peucker simplification; returns kept indices, ascending.

    Iterative (explicit stack) so multi-thousand-point tracks cannot hit the
    recursion limit. Endpoints are always kept; an interior point survives
    only if it deviates more than epsilon from the chord of its segment.
    """
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        chord = math.hypot(dx, dy)
        max_dist = -1.0
        max_idx = start
        for i in range(start + 1, end):
            px, py = points[i]
            if chord == 0.0:
                dist = math.hypot(px - ax, py - ay)
            else:
                dist = abs(dx * (ay - py) - (ax - px) * dy) / chord
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        if max_dist > epsilon:
            keep[max_idx] = True
            stack.append((start, max_idx))
            stack.append((max_idx, end))
    return [i for i, k in enumerate(keep) if k]


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x_cross > x:
                inside = not inside
    return inside
