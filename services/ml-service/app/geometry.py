"""Small pure-python geometry helpers for normalized (0-1) coordinates."""

from __future__ import annotations


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
