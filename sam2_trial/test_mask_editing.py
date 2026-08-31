import unittest

import numpy as np

from mask_editing import apply_mask_strokes, compose_edit, refine_mask


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


if __name__ == "__main__":
    unittest.main()
