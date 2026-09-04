import unittest
from unittest.mock import patch

import numpy as np

from mask_editing import (
    _fit_background_cover,
    apply_mask_strokes,
    compose_edit,
    crop_to_subject,
    refine_mask,
)


class MaskEditingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((30, 40, 3), dtype=np.uint8)
        self.image[:, :] = (24, 58, 92)
        self.image[8:22, 12:29] = (190, 75, 42)
        self.mask = np.zeros((30, 40), dtype=bool)
        self.mask[8:22, 12:29] = True

    def test_strokes_can_add_and_erase_pixels(self) -> None:
        updated = apply_mask_strokes(
            self.mask,
            [
                {"mode": "add", "radius": 2, "points": [{"x": 6, "y": 6}]},
                {"mode": "erase", "radius": 2, "points": [{"x": 18, "y": 15}]},
            ],
        )
        self.assertTrue(updated[6, 6])
        self.assertFalse(updated[15, 18])
        self.assertTrue(updated[9, 13])

    def test_refine_mask_expands_and_cleans_small_flecks(self) -> None:
        noisy = self.mask.copy()
        noisy[1, 1] = True
        cleaned = refine_mask(noisy, edge_offset=0, cleanup=True)
        self.assertFalse(cleaned[1, 1])
        expanded = refine_mask(self.mask, edge_offset=2, cleanup=False)
        self.assertGreater(expanded.sum(), self.mask.sum())

    def test_transparent_compose_preserves_subject_and_alpha(self) -> None:
        rendered = compose_edit(
            self.image,
            self.mask,
            background_mode="transparent",
            background_color=(255, 255, 255),
            background_blur_px=18,
            subject_brightness=0,
            subject_saturation=0,
            subject_blur_px=0,
            feather_px=0,
        )
        self.assertEqual(rendered.shape, (30, 40, 4))
        self.assertEqual(int(rendered[15, 18, 3]), 255)
        self.assertEqual(int(rendered[1, 1, 3]), 0)
        self.assertEqual(tuple(rendered[15, 18, :3]), (190, 75, 42))

    def test_color_background_only_replaces_unselected_area(self) -> None:
        rendered = compose_edit(
            self.image,
            self.mask,
            background_mode="color",
            background_color=(255, 255, 255),
            background_blur_px=18,
            subject_brightness=0,
            subject_saturation=0,
            subject_blur_px=0,
            feather_px=0,
        )
        self.assertEqual(rendered.shape, (30, 40, 3))
        self.assertEqual(tuple(rendered[1, 1]), (255, 255, 255))
        self.assertEqual(tuple(rendered[15, 18]), (190, 75, 42))

    def test_custom_background_is_cover_fitted_behind_the_subject(self) -> None:
        background = np.zeros((12, 12, 3), dtype=np.uint8)
        background[:, :] = (38, 170, 92)
        rendered = compose_edit(
            self.image,
            self.mask,
            background_mode="image",
            background_color=(255, 255, 255),
            background_blur_px=0,
            subject_brightness=0,
            subject_saturation=0,
            subject_blur_px=0,
            feather_px=0,
            background_image_rgb=background,
        )
        self.assertEqual(rendered.shape, (30, 40, 3))
        self.assertEqual(tuple(rendered[1, 1]), (38, 170, 92))
        self.assertEqual(tuple(rendered[15, 18]), (190, 75, 42))

    def test_extreme_wide_background_is_cropped_before_vertical_resize(self) -> None:
        background = np.zeros((24, 12_000, 3), dtype=np.uint8)
        resize_calls: list[tuple[tuple[int, ...], tuple[int, int]]] = []

        def guarded_resize(source, size, *, interpolation):
            resize_calls.append((source.shape, size))
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        with patch("mask_editing.cv2.resize", side_effect=guarded_resize):
            rendered = _fit_background_cover(background, 480, 120)

        self.assertEqual(rendered.shape, (480, 120, 3))
        self.assertEqual(resize_calls, [((24, 6, 3), (120, 480))])

    def test_extreme_tall_background_is_cropped_before_horizontal_resize(self) -> None:
        background = np.zeros((12_000, 24, 3), dtype=np.uint8)
        resize_calls: list[tuple[tuple[int, ...], tuple[int, int]]] = []

        def guarded_resize(source, size, *, interpolation):
            resize_calls.append((source.shape, size))
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        with patch("mask_editing.cv2.resize", side_effect=guarded_resize):
            rendered = _fit_background_cover(background, 120, 480)

        self.assertEqual(rendered.shape, (120, 480, 3))
        self.assertEqual(resize_calls, [((6, 24, 3), (480, 120))])

    def test_cover_fit_uses_the_center_two_color_crop_without_stretching(self) -> None:
        background = np.zeros((60, 180, 3), dtype=np.uint8)
        background[:, :60] = (20, 180, 40)
        background[:, 60:90] = (230, 35, 55)
        background[:, 90:120] = (40, 70, 225)
        background[:, 120:] = (235, 200, 30)

        rendered = _fit_background_cover(background, 60, 60)

        self.assertTrue(np.array_equal(rendered, background[:, 60:120]))
        self.assertEqual(tuple(rendered[30, 10]), (230, 35, 55))
        self.assertEqual(tuple(rendered[30, 50]), (40, 70, 225))

    def test_cover_fit_keeps_checker_cells_square_after_center_crop(self) -> None:
        background = np.zeros((80, 240, 3), dtype=np.uint8)
        for row in range(4):
            for column in range(12):
                value = 255 if (row + column) % 2 else 0
                background[row * 20 : (row + 1) * 20, column * 20 : (column + 1) * 20] = value

        rendered = _fit_background_cover(background, 160, 160)
        binary = rendered[:, :, 0] >= 128
        horizontal_transitions = int(np.count_nonzero(binary[20, 1:] != binary[20, :-1]))
        vertical_transitions = int(np.count_nonzero(binary[1:, 20] != binary[:-1, 20]))

        self.assertEqual(rendered.shape, (160, 160, 3))
        self.assertEqual(horizontal_transitions, 3)
        self.assertEqual(vertical_transitions, 3)

    def test_custom_background_mode_requires_an_rgb_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an image"):
            compose_edit(
                self.image,
                self.mask,
                background_mode="image",
                background_color=(255, 255, 255),
                background_blur_px=0,
                subject_brightness=0,
                subject_saturation=0,
                subject_blur_px=0,
                feather_px=0,
            )

    def test_compose_edit_supports_background_subject_effects_and_opacity(self) -> None:
        rendered = compose_edit(
            self.image,
            self.mask,
            background_mode="original",
            background_color=(255, 255, 255),
            background_blur_px=0,
            background_brightness=-12,
            background_saturation=-40,
            background_grayscale=True,
            subject_brightness=12,
            subject_saturation=15,
            subject_contrast=20,
            subject_hue_degrees=45,
            subject_temperature=18,
            subject_blur_px=0,
            subject_sharpen=10,
            subject_opacity=70,
            outline_width_px=2,
            outline_color=(255, 255, 255),
            outline_opacity=100,
            shadow_offset_x=3,
            shadow_offset_y=2,
            shadow_blur_px=0,
            shadow_color=(0, 0, 0),
            shadow_opacity=65,
            feather_px=0,
        )
        self.assertEqual(rendered.shape, (30, 40, 3))
        self.assertEqual(len({int(value) for value in rendered[1, 1]}), 1)
        self.assertNotEqual(tuple(rendered[1, 1]), (24, 58, 92))
        self.assertNotEqual(tuple(rendered[15, 18]), (190, 75, 42))
        self.assertNotEqual(tuple(rendered[15, 18]), tuple(rendered[1, 1]))
        self.assertNotEqual(tuple(rendered[7, 18]), tuple(rendered[1, 1]))
        self.assertNotEqual(tuple(rendered[23, 30]), tuple(rendered[1, 1]))

    def test_crop_to_subject_respects_padding_and_aspect_ratio(self) -> None:
        cropped_image, cropped_mask = crop_to_subject(
            self.image,
            self.mask,
            padding_px=3,
            aspect_ratio="1:1",
        )
        self.assertEqual(cropped_image.shape[:2], (23, 23))
        self.assertEqual(cropped_mask.shape, (23, 23))
        self.assertEqual(int(cropped_mask.sum()), int(self.mask.sum()))
        self.assertTrue(cropped_mask[4, 3])
        self.assertTrue(cropped_mask[17, 19])

    def test_crop_keeps_exact_ratio_when_padding_hits_source_boundary(self) -> None:
        boundary_mask = np.zeros((30, 40), dtype=bool)
        boundary_mask[6:24, 10:30] = True
        cropped_image, cropped_mask = crop_to_subject(
            self.image,
            boundary_mask,
            padding_px=12,
            aspect_ratio="1:1",
        )
        self.assertEqual(cropped_image.shape[:2], (30, 30))
        self.assertEqual(cropped_mask.shape, (30, 30))
        self.assertEqual(int(cropped_mask.sum()), int(boundary_mask.sum()))

    def test_crop_rejects_ratio_that_would_cut_the_subject(self) -> None:
        full_width_mask = np.zeros((30, 40), dtype=bool)
        full_width_mask[10:20, :] = True
        with self.assertRaisesRegex(ValueError, "无法在不裁掉主体"):
            crop_to_subject(
                self.image,
                full_width_mask,
                padding_px=0,
                aspect_ratio="1:1",
            )


if __name__ == "__main__":
    unittest.main()
