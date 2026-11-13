from __future__ import annotations

import unittest

import numpy as np

from kalman_tracker import ConstantVelocityKalman


class KalmanTrackerTests(unittest.TestCase):
    def make_tracker(self):
        return ConstantVelocityKalman(
            process_noise_accel=300.0,
            measurement_noise_mm=10.0,
            initial_position_std_mm=100.0,
            initial_velocity_std_mm_s=1000.0,
            gate_threshold=50.0,
            jump_reset_mm=800.0,
            reset_after_rejections=3,
        )

    def test_constant_velocity_estimation_and_future_prediction(self):
        tracker = self.make_tracker()
        result = None
        for index in range(16):
            timestamp = index * 0.1
            measurement = np.array([100.0 + 1000.0 * timestamp, -50.0, 1200.0])
            result = tracker.update(measurement, timestamp, prediction_seconds=0.2)

        assert result is not None
        self.assertTrue(result.initialized)
        self.assertAlmostEqual(result.velocity[0], 1000.0, delta=35.0)
        self.assertAlmostEqual(result.position[0], 1600.0, delta=25.0)
        self.assertAlmostEqual(result.predicted_future[0], 1800.0, delta=50.0)

    def test_missing_measurement_only_predicts(self):
        tracker = self.make_tracker()
        for index in range(6):
            timestamp = index * 0.1
            tracker.update([500.0 + 400.0 * timestamp, 0.0, 1500.0], timestamp)

        before = tracker.state.copy()
        predicted_only = tracker.predict_to(0.9)

        self.assertFalse(predicted_only.updated)
        self.assertTrue(predicted_only.initialized)
        self.assertAlmostEqual(
            predicted_only.position[0],
            before[0] + before[3] * 0.4,
            delta=8.0,
        )
        self.assertGreater(predicted_only.miss_duration_s, 0.25)

    def test_large_jump_resets_tracker(self):
        tracker = self.make_tracker()
        tracker.update([0.0, 0.0, 1000.0], 0.0)
        tracker.update([10.0, 0.0, 1000.0], 0.1)

        result = tracker.update([5000.0, 0.0, 1000.0], 0.2)

        self.assertTrue(result.reset)
        self.assertTrue(result.updated)
        np.testing.assert_allclose(result.position, [5000.0, 0.0, 1000.0])
        np.testing.assert_allclose(result.velocity, [0.0, 0.0, 0.0])

    def test_clear_removes_track(self):
        tracker = self.make_tracker()
        tracker.update([1.0, 2.0, 3.0], 0.0)
        tracker.clear()
        result = tracker.predict_to(1.0)
        self.assertFalse(result.initialized)
        self.assertTrue(np.all(np.isnan(result.position)))


if __name__ == "__main__":
    unittest.main()
