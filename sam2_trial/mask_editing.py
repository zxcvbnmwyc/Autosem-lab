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


def _apply_tone(
    image_rgb: np.ndarray,
    *,
    brightness: int,
    saturation: int,
    hue_degrees: int,
    grayscale: bool,
) -> np.ndarray:
    if brightness == 0 and saturation == 0 and hue_degrees == 0 and not grayscale:
        return image_rgb
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + float(hue_degrees) / 2.0, 180.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation / 100.0), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + brightness / 100.0), 0, 255)
    adjusted = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if grayscale:
        gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        adjusted = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return adjusted


def _apply_contrast(image_rgb: np.ndarray, contrast: int) -> np.ndarray:
    if contrast == 0:
        return image_rgb
    factor = 1.0 + contrast / 100.0
    adjusted = (image_rgb.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_temperature(image_rgb: np.ndarray, temperature: int) -> np.ndarray:
    if temperature == 0:
        return image_rgb
    adjusted = image_rgb.astype(np.float32)
    delta = float(temperature) * 0.6
    adjusted[:, :, 0] += delta
    adjusted[:, :, 2] -= delta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _apply_sharpen(image_rgb: np.ndarray, sharpen: int) -> np.ndarray:
    if sharpen <= 0:
        return image_rgb
    sigma = 1.0
    strength = min(2.0, float(sharpen) / 20.0)
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(image_rgb, 1.0 + strength, blurred, -strength, 0)


def _adjust_region(
    image_rgb: np.ndarray,
    *,
    brightness: int,
    saturation: int,
    contrast: int,
    hue_degrees: int,
    temperature: int,
    blur_px: int,
    sharpen: int,
    grayscale: bool = False,
) -> np.ndarray:
    adjusted = image_rgb
    if blur_px > 0:
        sigma = max(0.1, float(blur_px) / 2.0)
        adjusted = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    adjusted = _apply_tone(
        adjusted,
        brightness=brightness,
        saturation=saturation,
        hue_degrees=hue_degrees,
        grayscale=grayscale,
    )
    adjusted = _apply_contrast(adjusted, contrast)
    adjusted = _apply_temperature(adjusted, temperature)
    return _apply_sharpen(adjusted, sharpen)


def _over(
    base_rgb: np.ndarray,
    base_alpha: np.ndarray,
    layer_rgb: np.ndarray,
    layer_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite one straight-alpha RGB layer over another."""
    layer_alpha_3d = layer_alpha[:, :, None]
    base_alpha_3d = base_alpha[:, :, None]
    output_alpha = layer_alpha + base_alpha * (1.0 - layer_alpha)
    premultiplied = (
        layer_rgb * layer_alpha_3d
        + base_rgb * base_alpha_3d * (1.0 - layer_alpha_3d)
    )
    denominator = np.maximum(output_alpha[:, :, None], 1e-6)
    output_rgb = np.where(output_alpha[:, :, None] > 0, premultiplied / denominator, 0)
    return output_rgb, output_alpha


def _solid_layer(shape: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    layer = np.empty((*shape, 3), dtype=np.float32)
    layer[:, :] = color
    return layer


def _shift_alpha(alpha: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    height, width = alpha.shape
    matrix = np.asarray([[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)]], dtype=np.float32)
    return cv2.warpAffine(
        alpha,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def crop_to_subject(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    padding_px: int,
    aspect_ratio: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image and its mask around the selected subject."""
    if image.shape[:2] != mask.shape:
        raise ValueError("Mask dimensions do not match the rendered image.")
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot crop an empty subject mask.")
    height, width = mask.shape
    subject_y0, subject_x0 = coordinates.min(axis=0)
    subject_y1, subject_x1 = coordinates.max(axis=0) + 1
    padding = max(0, int(padding_px))
    padded_x0 = max(0, int(subject_x0) - padding)
    padded_y0 = max(0, int(subject_y0) - padding)
    padded_x1 = min(width, int(subject_x1) + padding)
    padded_y1 = min(height, int(subject_y1) + padding)

    ratios = {"free": None, "1:1": (1, 1), "4:5": (4, 5), "16:9": (16, 9)}
    if aspect_ratio not in ratios:
        raise ValueError("Unsupported crop aspect ratio.")
    ratio_parts = ratios[aspect_ratio]
    if ratio_parts is None:
        return (
            image[padded_y0:padded_y1, padded_x0:padded_x1].copy(),
            mask[padded_y0:padded_y1, padded_x0:padded_x1].copy(),
        )

    ratio_width, ratio_height = ratio_parts
    required_scale = max(
        int(np.ceil((padded_x1 - padded_x0) / ratio_width)),
        int(np.ceil((padded_y1 - padded_y0) / ratio_height)),
    )
    maximum_scale = min(width // ratio_width, height // ratio_height)
    crop_scale = min(required_scale, maximum_scale)
    crop_width = ratio_width * crop_scale
    crop_height = ratio_height * crop_scale
    subject_width = int(subject_x1 - subject_x0)
    subject_height = int(subject_y1 - subject_y0)
    if crop_scale < 1 or crop_width < subject_width or crop_height < subject_height:
        raise ValueError("当前图片无法在不裁掉主体的情况下生成所选比例。")

    # Preserve all requested padding when it fits.  If the source boundary makes
    # that impossible, keep the complete subject and use the largest exact-ratio
    # rectangle available instead of silently returning the wrong ratio.
    if crop_width >= padded_x1 - padded_x0 and crop_height >= padded_y1 - padded_y0:
        constraint_x0, constraint_y0 = padded_x0, padded_y0
        constraint_x1, constraint_y1 = padded_x1, padded_y1
    else:
        constraint_x0, constraint_y0 = int(subject_x0), int(subject_y0)
        constraint_x1, constraint_y1 = int(subject_x1), int(subject_y1)

    center_x = (padded_x0 + padded_x1) / 2.0
    center_y = (padded_y0 + padded_y1) / 2.0
    minimum_x0 = max(0, constraint_x1 - crop_width)
    maximum_x0 = min(constraint_x0, width - crop_width)
    minimum_y0 = max(0, constraint_y1 - crop_height)
    maximum_y0 = min(constraint_y0, height - crop_height)
    x0 = int(np.clip(round(center_x - crop_width / 2.0), minimum_x0, maximum_x0))
    y0 = int(np.clip(round(center_y - crop_height / 2.0), minimum_y0, maximum_y0))
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def _fit_background_cover(
    background_rgb: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Centre-crop to the target ratio, then resize without a large intermediate."""
    if background_rgb.ndim != 3 or background_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB background image.")
    source_height, source_width = background_rgb.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError("Background image cannot be empty.")
    if target_height < 1 or target_width < 1:
        raise ValueError("Target background dimensions must be positive.")

    # A resize-first cover fit can turn an extreme panorama into a huge, mostly
    # discarded intermediate image.  Crop the excess dimension while the image
    # is still at source resolution, then allocate only the final target size.
    if source_width * target_height > source_height * target_width:
        crop_height = source_height
        crop_width = min(
            source_width,
            max(1, int(round(source_height * target_width / target_height))),
        )
    else:
        crop_width = source_width
        crop_height = min(
            source_height,
            max(1, int(round(source_width * target_height / target_width))),
        )
    x0 = (source_width - crop_width) // 2
    y0 = (source_height - crop_height) // 2
    cropped = np.ascontiguousarray(
        background_rgb[y0 : y0 + crop_height, x0 : x0 + crop_width]
    )
    interpolation = (
        cv2.INTER_AREA
        if target_width < crop_width or target_height < crop_height
        else cv2.INTER_CUBIC
    )
    return cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=interpolation,
    )


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
    background_brightness: int = 0,
    background_saturation: int = 0,
    background_grayscale: bool = False,
    subject_contrast: int = 0,
    subject_hue_degrees: int = 0,
    subject_temperature: int = 0,
    subject_sharpen: int = 0,
    subject_opacity: int = 100,
    outline_width_px: int = 0,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    outline_opacity: int = 0,
    shadow_offset_x: int = 0,
    shadow_offset_y: int = 0,
    shadow_blur_px: int = 0,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    shadow_opacity: int = 0,
    background_image_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Render a full-resolution RGB or RGBA edit from a source image and mask."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB source image.")
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("Mask dimensions do not match the source image.")

    subject = _adjust_region(
        image_rgb,
        brightness=subject_brightness,
        saturation=subject_saturation,
        contrast=subject_contrast,
        hue_degrees=subject_hue_degrees,
        temperature=subject_temperature,
        blur_px=subject_blur_px,
        sharpen=subject_sharpen,
    )
    alpha = _alpha_from_mask(mask, feather_px).astype(np.float32) / 255.0

    if background_mode == "color":
        background = np.empty_like(image_rgb)
        background[:, :] = background_color
    elif background_mode == "image":
        if background_image_rgb is None:
            raise ValueError("Custom background mode requires an image.")
        background = _fit_background_cover(
            background_image_rgb, image_rgb.shape[0], image_rgb.shape[1]
        )
    elif background_mode == "blur":
        sigma = max(0.1, float(background_blur_px) / 2.0)
        background = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        background = image_rgb
    background = _adjust_region(
        background,
        brightness=background_brightness,
        saturation=background_saturation,
        contrast=0,
        hue_degrees=0,
        temperature=0,
        blur_px=0,
        sharpen=0,
        grayscale=background_grayscale,
    )
    if background_mode == "transparent":
        base_rgb = np.zeros_like(image_rgb, dtype=np.float32)
        base_alpha = np.zeros(mask.shape, dtype=np.float32)
    else:
        base_rgb = background.astype(np.float32)
        base_alpha = np.ones(mask.shape, dtype=np.float32)

    if shadow_opacity > 0:
        shadow_alpha = _shift_alpha(alpha, shadow_offset_x, shadow_offset_y)
        if shadow_blur_px > 0:
            sigma = max(0.1, float(shadow_blur_px) / 2.0)
            shadow_alpha = cv2.GaussianBlur(
                shadow_alpha,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_CONSTANT,
            )
        shadow_alpha = np.clip(shadow_alpha * (shadow_opacity / 100.0), 0.0, 1.0)
        base_rgb, base_alpha = _over(
            base_rgb,
            base_alpha,
            _solid_layer(mask.shape, shadow_color),
            shadow_alpha,
        )

    if outline_width_px > 0 and outline_opacity > 0:
        kernel_size = outline_width_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        outline_alpha = np.clip(dilated - alpha, 0.0, 1.0) * (outline_opacity / 100.0)
        base_rgb, base_alpha = _over(
            base_rgb,
            base_alpha,
            _solid_layer(mask.shape, outline_color),
            outline_alpha,
        )

    subject_alpha = np.clip(alpha * (subject_opacity / 100.0), 0.0, 1.0)
    base_rgb, base_alpha = _over(
        base_rgb,
        base_alpha,
        subject.astype(np.float32),
        subject_alpha,
    )
    output_rgb = np.clip(np.rint(base_rgb), 0, 255).astype(np.uint8)
    if background_mode == "transparent":
        output_alpha = np.clip(np.rint(base_alpha * 255.0), 0, 255).astype(np.uint8)
        return np.dstack((output_rgb, output_alpha))
    return output_rgb
