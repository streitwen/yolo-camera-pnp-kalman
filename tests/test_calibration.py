from __future__ import annotations

import unittest

import cv2
import numpy as np

from calibrate import calibrate_from_correspondences, make_object_points


class CalibrationTests(unittest.TestCase):
    def test_synthetic_chessboard_calibration_recovers_intrinsics(self):
        image_size = (1280, 720)
        expected_k = np.array(
            [
                [920.0, 0.0, 642.0],
                [0.0, 915.0, 358.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.zeros(5, dtype=np.float64)
        template = make_object_points((9, 6), 25.0)
        object_points = []
        image_points = []
        poses = [
            ((-0.25, -0.10, 0.02), (-110.0, -40.0, 620.0)),
            ((0.20, -0.18, -0.04), (-90.0, -20.0, 650.0)),
            ((-0.20, 0.24, 0.05), (-100.0, 10.0, 610.0)),
            ((0.28, 0.16, -0.03), (-80.0, -30.0, 640.0)),
            ((0.12, -0.30, 0.08), (-120.0, 0.0, 670.0)),
            ((-0.30, 0.08, -0.07), (-70.0, -50.0, 630.0)),
            ((0.33, -0.05, 0.04), (-105.0, -15.0, 690.0)),
            ((-0.08, 0.31, -0.05), (-95.0, -25.0, 660.0)),
        ]
        for rvec, tvec in poses:
            projected, _ = cv2.projectPoints(
                template,
                np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                expected_k,
                dist,
            )
            object_points.append(template.copy())
            image_points.append(projected)

        result = calibrate_from_correspondences(
            object_points,
            image_points,
            image_size,
        )

        np.testing.assert_allclose(result.camera_matrix, expected_k, atol=0.25)
        self.assertLess(result.rms_reprojection_error_px, 1e-4)
        self.assertEqual(len(result.per_view_errors_px), len(poses))

    def test_object_points_have_expected_grid(self):
        points = make_object_points((9, 6), 20.0)
        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[8], [160.0, 0.0, 0.0])
        np.testing.assert_allclose(points[9], [0.0, 20.0, 0.0])


if __name__ == "__main__":
    unittest.main()
