import unittest

import numpy as np

from contours import mask_to_contours, polygon_area_px2


class ContourExportTests(unittest.TestCase):
    def test_empty_mask_has_no_components(self) -> None:
        self.assertEqual(mask_to_contours(np.zeros((5, 7), dtype=bool)), [])

    def test_rectangle_has_one_outer_ring(self) -> None:
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[2:6, 3:8] = 1
        result = mask_to_contours(mask)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["holes"], [])
        self.assertEqual(result[0]["polygon_area_px2"], 12.0)

    def test_hole_is_retained(self) -> None:
        mask = np.ones((9, 9), dtype=bool)
        mask[3:6, 3:6] = False
        result = mask_to_contours(mask)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["holes"]), 1)

    def test_disconnected_components_are_retained(self) -> None:
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:4, 1:4] = True
        mask[7:10, 8:11] = True
        result = mask_to_contours(mask)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["outer"][0], [1, 1])

    def test_non_binary_mask_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mask_to_contours(np.array([[0, 2]], dtype=np.uint8))

    def test_degenerate_area_is_zero(self) -> None:
        self.assertEqual(polygon_area_px2([[1, 1]]), 0.0)


if __name__ == "__main__":
    unittest.main()
