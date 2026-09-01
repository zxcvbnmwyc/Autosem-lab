import unittest

import numpy as np

from mask_editing import apply_mask_strokes, compose_edit, crop_to_subject, refine_mask


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
