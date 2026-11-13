from __future__ import annotations

import unittest

import cv2
import numpy as np

from pose_math import (
    CameraCalibration,
    object_points_for_order,
    parse_corner_order,
    select_primary_detection,
    solve_armor_pose,
)


class PoseMathTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            image_size=(1280, 720),
            camera_matrix=np.array(
                [
                    [900.0, 0.0, 640.0],
                    [0.0, 910.0, 360.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            dist_coeffs=np.zeros(5, dtype=np.float64),
            rms_reprojection_error_px=0.2,
        )

    def test_corner_order_parser_and_object_points(self):
        order = parse_corner_order("左上,tr,bottom-right,BL")
        self.assertEqual(order, ("TL", "TR", "BR", "BL"))
        points = object_points_for_order(140.0, 55.0, order)
        np.testing.assert_allclose(points[0], [-70.0, -27.5, 0.0])
        np.testing.assert_allclose(points[2], [70.0, 27.5, 0.0])

        with self.assertRaises(ValueError):
            parse_corner_order("TL,TR,TL,BL")

    def test_synthetic_pnp_recovers_camera_translation(self):
        order = ("TL", "TR", "BR", "BL")
        object_points = object_points_for_order(140.0, 55.0, order)
        expected_rvec = np.array([0.08, -0.12, 0.03], dtype=np.float64)
        expected_tvec = np.array([25.0, -18.0, 820.0], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
        )

        pose = solve_armor_pose(
            projected.reshape(4, 2),
            self.calibration,
            armor_width_mm=140.0,
            armor_height_mm=55.0,
            corner_order=order,
            method="ippe",
        )

        np.testing.assert_allclose(pose.xyz_mm, expected_tvec, atol=1.0)
        self.assertLess(pose.reprojection_error_px, 0.1)
        self.assertAlmostEqual(pose.distance_mm, np.linalg.norm(expected_tvec), delta=1.0)

    def test_pose_rejects_degenerate_keypoints(self):
        points = np.array(
            [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]
        )
        with self.assertRaises(ValueError):
            solve_armor_pose(
                points,
                self.calibration,
                armor_width_mm=140.0,
                armor_height_mm=55.0,
                corner_order=("TL", "TR", "BR", "BL"),
            )

    def test_primary_target_prefers_previous_position(self):
        detections = [
            {
                "box": [300.0, 300.0, 400.0, 380.0],
                "conf": 0.95,
                "kxy": [[300, 380], [300, 300], [400, 300], [400, 380]],
            },
            {
                "box": [100.0, 100.0, 180.0, 160.0],
                "conf": 0.55,
                "kxy": [[100, 160], [100, 100], [180, 100], [180, 160]],
            },
        ]
        selected, reason = select_primary_detection(
            detections,
            (640, 480),
            previous_box=[90.0, 90.0, 170.0, 150.0],
        )
        self.assertIsNone(reason)
        assert selected is not None
        self.assertAlmostEqual(selected["conf"], 0.55)


if __name__ == "__main__":
    unittest.main()
