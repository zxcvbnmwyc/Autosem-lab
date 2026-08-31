"""Pure image-compositing helpers for AutoSEM's editable-mask workflow.

The browser sends only small, validated editing instructions.  These helpers
always start from a server-owned source image and SAM2 mask, so a displayed
canvas can never accidentally become the export resolution.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_mask_strokes(mask: np.ndarray, strokes: list[dict[str, Any]]) -> np.ndarray:
    """Apply validated add/erase brush strokes to a binary mask."""
    output = np.ascontiguousarray(mask.astype(np.uint8))
    for stroke in strokes:
        points = stroke["points"]
        radius = int(stroke["radius"])
        value = 1 if stroke["mode"] == "add" else 0
        coordinates = np.asarray(
            [[round(float(point["x"])), round(float(point["y"]))] for point in points],
            dtype=np.int32,
        )
        if coordinates.size == 0:
            continue
        if len(coordinates) == 1:
            cv2.circle(output, tuple(coordinates[0]), radius, value, thickness=-1, lineType=cv2.LINE_8)
            continue
        cv2.polylines(
            output,
            [coordinates.reshape((-1, 1, 2))],
            isClosed=False,
            color=value,
            thickness=max(1, radius * 2),
            lineType=cv2.LINE_8,
        )
        cv2.circle(output, tuple(coordinates[0]), radius, value, thickness=-1, lineType=cv2.LINE_8)
        cv2.circle(output, tuple(coordinates[-1]), radius, value, thickness=-1, lineType=cv2.LINE_8)
    return output.astype(bool)


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove isolated foreground flecks and fill tiny enclosed holes.

    The largest foreground component is retained even when it is smaller than
    the threshold, which keeps intentionally small selected objects usable.
    """
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = np.flatnonzero(areas >= minimum_area) + 1
    if keep_labels.size == 0:
        keep_labels = np.asarray([int(np.argmax(areas)) + 1], dtype=np.int32)
    cleaned = np.isin(labels, keep_labels)

    inverted = np.ascontiguousarray((~cleaned).astype(np.uint8))
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if hole_count <= 1:
        return cleaned
    border_labels = np.unique(
        np.concatenate((hole_labels[0, :], hole_labels[-1, :], hole_labels[:, 0], hole_labels[:, -1]))
    )
    hole_areas = hole_stats[1:, cv2.CC_STAT_AREA]
    candidate_labels = np.flatnonzero(hole_areas <= minimum_area) + 1
    fill_labels = np.setdiff1d(candidate_labels, border_labels, assume_unique=False)
    if fill_labels.size:
        cleaned |= np.isin(hole_labels, fill_labels)
    return cleaned


def refine_mask(mask: np.ndarray, *, edge_offset: int, cleanup: bool) -> np.ndarray:
    """Apply light deterministic cleanup and a pixel-space edge offset."""
    refined = mask.astype(bool)
    if cleanup:
        minimum_area = max(16, round(refined.size * 0.00002))
        refined = _remove_small_components(refined, minimum_area)
    if edge_offset:
        kernel_size = max(1, abs(int(edge_offset)) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        operation = cv2.dilate if edge_offset > 0 else cv2.erode
        refined = operation(np.ascontiguousarray(refined.astype(np.uint8)), kernel, iterations=1).astype(bool)
    return refined


def _alpha_from_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    if feather_px > 0:
        sigma = max(0.1, float(feather_px) / 2.0)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return alpha


def _adjust_subject(
    image_rgb: np.ndarray,
    *,
    brightness: int,
    saturation: int,
    blur_px: int,
) -> np.ndarray:
    adjusted = image_rgb
    if blur_px > 0:
        sigma = max(0.1, float(blur_px) / 2.0)
        adjusted = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    if brightness == 0 and saturation == 0:
        return adjusted
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation / 100.0), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + brightness / 100.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def compose_edit(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    background_mode: str,
    background_color: tuple[int, int, int],
    background_blur_px: int,
    subject_brightness: int,
    subject_saturation: int,
    subject_blur_px: int,
    feather_px: int,
) -> np.ndarray:
    """Render a full-resolution RGB or RGBA edit from a source image and mask."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB source image.")
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("Mask dimensions do not match the source image.")

    subject = _adjust_subject(
        image_rgb,
        brightness=subject_brightness,
        saturation=subject_saturation,
        blur_px=subject_blur_px,
    )
    alpha = _alpha_from_mask(mask, feather_px)
    if background_mode == "transparent":
        return np.dstack((subject, alpha))

    if background_mode == "color":
        background = np.empty_like(image_rgb)
        background[:, :] = background_color
    elif background_mode == "blur":
        sigma = max(0.1, float(background_blur_px) / 2.0)
        background = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        background = image_rgb

    alpha_float = alpha.astype(np.float32)[:, :, None] / 255.0
    return np.clip(subject.astype(np.float32) * alpha_float + background.astype(np.float32) * (1.0 - alpha_float), 0, 255).astype(np.uint8)
