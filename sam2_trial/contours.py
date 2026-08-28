"""Export unsmoothed, JSON-ready contours from one binary mask.

Coordinates are [x, y] pixel-center coordinates in the mask's original
coordinate system. Rings are implicitly closed and are deliberately not
smoothed, so downstream code can reproduce the segmentation faithfully.
"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_area_px2(points: list[list[int]]) -> float:
    """Return unsigned shoelace area for an implicitly closed ring."""
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.shape == (0,):
        return 0.0
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("Contour points must have shape (N, 2).")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Contour coordinates must be finite.")
    if len(vertices) < 3:
        return 0.0
    vertices = vertices - vertices[0]
    x, y = vertices[:, 0], vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _spatial_key(points: list[list[int]]) -> tuple[int, int]:
    return min(point[1] for point in points), min(point[0] for point in points)


def mask_to_contours(mask: np.ndarray) -> list[dict]:
    """Return components, holes and pixel-center contour coordinates.

    Input must be a two-dimensional bool or numeric binary mask. Nested
    foreground islands remain independent components, and holes are retained.
    polygon_area_px2 is geometry between center points, not pixel count.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional (height, width).")
    if array.dtype.kind not in "buif":
        raise TypeError("mask must contain bool or numeric 0/1 values.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must contain only bool or numeric 0/1 values.")
    if array.size == 0 or not np.any(array):
        return []

    binary = np.ascontiguousarray(array != 0, dtype=np.uint8)
    found, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []

    links = hierarchy[0]
    components: list[dict] = []
    for index, contour in enumerate(found):
        if links[index, 3] != -1:
            continue
        outer = contour.reshape(-1, 2).tolist()
        holes: list[list[list[int]]] = []
        child = int(links[index, 2])
        while child != -1:
            holes.append(found[child].reshape(-1, 2).tolist())
            child = int(links[child, 0])
        holes.sort(key=_spatial_key)
        components.append(
            {
                "outer": outer,
                "holes": holes,
                "polygon_area_px2": polygon_area_px2(outer)
                - sum(polygon_area_px2(hole) for hole in holes),
            }
        )

    components.sort(key=lambda component: _spatial_key(component["outer"]))
    return components
