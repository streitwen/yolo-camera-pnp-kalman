#!/usr/bin/env python3
"""六状态匀速卡尔曼滤波器：位置 [x,y,z] 与速度 [vx,vy,vz]。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KalmanResult:
    state: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    predicted_future: np.ndarray
    initialized: bool
    updated: bool
    rejected: bool
    reset: bool
    miss_duration_s: float
    mahalanobis_squared: float | None


class ConstantVelocityKalman:
    """使用真实时间差传播的恒速模型；丢测时仅预测，不伪造测量。"""

    def __init__(
        self,
        *,
        process_noise_accel: float = 800.0,
        measurement_noise_mm: float = 25.0,
        initial_position_std_mm: float = 300.0,
        initial_velocity_std_mm_s: float = 1500.0,
        gate_threshold: float = 25.0,
        jump_reset_mm: float = 1000.0,
        reset_after_rejections: int = 3,
        max_dt_s: float = 1.0,
    ):
        if process_noise_accel <= 0:
            raise ValueError("process_noise_accel 必须大于 0")
        if measurement_noise_mm <= 0:
            raise ValueError("measurement_noise_mm 必须大于 0")
        self.process_noise_accel = float(process_noise_accel)
        self.measurement_noise_mm = float(measurement_noise_mm)
        self.initial_position_std_mm = float(initial_position_std_mm)
        self.initial_velocity_std_mm_s = float(initial_velocity_std_mm_s)
        self.gate_threshold = float(gate_threshold)
        self.jump_reset_mm = float(jump_reset_mm)
        self.reset_after_rejections = max(1, int(reset_after_rejections))
        self.max_dt_s = float(max_dt_s)

        self.initialized = False
        self.last_time: float | None = None
        self.last_measurement_time: float | None = None
        self.consecutive_rejections = 0
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)
        self.R = np.eye(3, dtype=np.float64) * self.measurement_noise_mm**2

    def _transition(self, dt: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        return transition

    def _process_covariance(self, dt: float) -> np.ndarray:
        # 每个轴使用连续白噪声加速度模型；位置/速度单位分别为 mm 和 mm/s。
        q = self.process_noise_accel**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q1 = np.array(
            [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]],
            dtype=np.float64,
        )
        process = np.zeros((6, 6), dtype=np.float64)
        for position_index, velocity_index in ((0, 3), (1, 4), (2, 5)):
            process[
                np.ix_([position_index, velocity_index], [position_index, velocity_index])
            ] = q1 * q
        return process

    def elapsed_since_measurement(self, timestamp_s: float) -> float:
        if self.last_measurement_time is None:
            return 0.0
        return max(0.0, float(timestamp_s) - self.last_measurement_time)

    def reset(self, measurement_mm: Sequence[float], timestamp_s: float) -> None:
        measurement = np.asarray(measurement_mm, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(measurement)):
            raise ValueError("卡尔曼复位测量包含非有限值")
        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = measurement
        self.covariance = np.diag(
            [
                self.initial_position_std_mm**2,
                self.initial_position_std_mm**2,
                self.initial_position_std_mm**2,
                self.initial_velocity_std_mm_s**2,
                self.initial_velocity_std_mm_s**2,
                self.initial_velocity_std_mm_s**2,
            ]
        )
        self.last_time = float(timestamp_s)
        self.last_measurement_time = float(timestamp_s)
        self.consecutive_rejections = 0
        self.initialized = True

    def clear(self) -> None:
        """清除当前航迹；下一次有效测量会重新初始化，不伪造数据。"""
        self.initialized = False
        self.last_time = None
        self.last_measurement_time = None
        self.consecutive_rejections = 0
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)

    def predict_to(self, timestamp_s: float) -> KalmanResult:
        if not self.initialized or self.last_time is None:
            return self._empty_result(float(timestamp_s))

        dt = max(0.0, float(timestamp_s) - self.last_time)
        if dt > 0:
            dt = min(dt, self.max_dt_s)
            transition = self._transition(dt)
            self.state = transition @ self.state
            self.covariance = (
                transition @ self.covariance @ transition.T
                + self._process_covariance(dt)
            )
            self.last_time = float(timestamp_s)
        return self._result(float(timestamp_s), updated=False)

    def update(
        self,
        measurement_mm: Sequence[float],
        timestamp_s: float,
        *,
        prediction_seconds: float = 0.2,
    ) -> KalmanResult:
        measurement = np.asarray(measurement_mm, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(measurement)):
            raise ValueError("卡尔曼测量包含非有限值")

        now = float(timestamp_s)
        if not self.initialized:
            self.reset(measurement, now)
            return self._result(now, updated=True, reset=True, prediction_seconds=prediction_seconds)

        if self.last_time is None:
            self.reset(measurement, now)
            return self._result(now, updated=True, reset=True, prediction_seconds=prediction_seconds)

        self.predict_to(now)
        innovation = measurement - self.H @ self.state
        jump = float(np.linalg.norm(innovation))
        if jump > self.jump_reset_mm:
            self.reset(measurement, now)
            return self._result(
                now,
                updated=True,
                reset=True,
                prediction_seconds=prediction_seconds,
            )

        innovation_covariance = self.H @ self.covariance @ self.H.T + self.R
        try:
            mahalanobis_squared = float(
                innovation.T @ np.linalg.solve(innovation_covariance, innovation)
            )
        except np.linalg.LinAlgError:
            self.reset(measurement, now)
            return self._result(
                now,
                updated=True,
                reset=True,
                prediction_seconds=prediction_seconds,
            )

        if mahalanobis_squared > self.gate_threshold:
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= self.reset_after_rejections:
                old_mahalanobis = mahalanobis_squared
                self.reset(measurement, now)
                result = self._result(
                    now,
                    updated=True,
                    reset=True,
                    prediction_seconds=prediction_seconds,
                )
                return KalmanResult(
                    state=result.state,
                    position=result.position,
                    velocity=result.velocity,
                    predicted_future=result.predicted_future,
                    initialized=result.initialized,
                    updated=result.updated,
                    rejected=False,
                    reset=True,
                    miss_duration_s=result.miss_duration_s,
                    mahalanobis_squared=old_mahalanobis,
                )
            return self._result(
                now,
                updated=False,
                rejected=True,
                prediction_seconds=prediction_seconds,
                mahalanobis_squared=mahalanobis_squared,
            )

        gain = (
            self.covariance
            @ self.H.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        # Joseph 形式在浮点误差下仍保持协方差对称、半正定。
        correction = identity - gain @ self.H
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ self.R @ gain.T
        )
        self.last_measurement_time = now
        self.consecutive_rejections = 0
        return self._result(
            now,
            updated=True,
            prediction_seconds=prediction_seconds,
            mahalanobis_squared=mahalanobis_squared,
        )

    def predict_future(self, seconds: float) -> np.ndarray:
        if not self.initialized:
            return np.full(3, np.nan, dtype=np.float64)
        seconds = max(0.0, float(seconds))
        transition = self._transition(seconds)
        return (transition @ self.state)[:3]

    def _result(
        self,
        timestamp_s: float,
        *,
        updated: bool,
        rejected: bool = False,
        reset: bool = False,
        prediction_seconds: float = 0.2,
        mahalanobis_squared: float | None = None,
    ) -> KalmanResult:
        return KalmanResult(
            state=self.state.copy(),
            position=self.state[:3].copy(),
            velocity=self.state[3:].copy(),
            predicted_future=self.predict_future(prediction_seconds),
            initialized=self.initialized,
            updated=updated,
            rejected=rejected,
            reset=reset,
            miss_duration_s=self.elapsed_since_measurement(timestamp_s),
            mahalanobis_squared=mahalanobis_squared,
        )

    def _empty_result(self, timestamp_s: float) -> KalmanResult:
        return KalmanResult(
            state=np.full(6, np.nan),
            position=np.full(3, np.nan),
            velocity=np.full(3, np.nan),
            predicted_future=np.full(3, np.nan),
            initialized=False,
            updated=False,
            rejected=False,
            reset=False,
            miss_duration_s=self.elapsed_since_measurement(timestamp_s),
            mahalanobis_squared=None,
        )
