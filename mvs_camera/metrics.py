"""Image-quality, frame-rate, and CPU measurements for camera experiments."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def image_metrics(image_bgr: np.ndarray, saturation_threshold: int = 250) -> dict[str, Any]:
    """Measure brightness, clipping, dark noise, and a sharpness proxy."""
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    pixels = gray.reshape(-1)
    if pixels.size == 0:
        raise ValueError("empty image")

    dark = pixels[pixels < 64]
    scale = min(1.0, 960.0 / max(gray.shape[1], 1))
    if scale < 1.0:
        sharp_gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        sharp_gray = gray
    return {
        "pixel_count": int(pixels.size),
        "mean_gray": round(float(pixels.mean()), 3),
        "std_gray": round(float(pixels.std()), 3),
        "saturated_pixel_ratio": round(
            float(np.count_nonzero(pixels >= saturation_threshold) / pixels.size), 6
        ),
        "dark_pixel_ratio": round(float(dark.size / pixels.size), 6),
        "dark_noise_std": round(float(dark.std()), 3) if dark.size >= 16 else None,
        "laplacian_variance": round(
            float(cv2.Laplacian(sharp_gray, cv2.CV_64F).var()), 3
        ),
        "saturation_threshold": saturation_threshold,
    }


def target_metrics(dets: list[dict], image_shape: tuple[int, ...]) -> dict[str, Any]:
    h, w = image_shape[:2]
    centers = []
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        centers.append(
            {
                "class": det["name"],
                "confidence": round(float(det["conf"]), 6),
                "center_x": round(float((x1 + x2) / 2.0), 2),
                "center_y": round(float((y1 + y2) / 2.0), 2),
            }
        )
    return {
        "image_width": int(w),
        "image_height": int(h),
        "target_count": len(dets),
        "class_counts": {
            name: sum(1 for d in dets if d["name"] == name)
            for name in sorted({d["name"] for d in dets})
        },
        "average_confidence": round(
            float(np.mean([d["conf"] for d in dets])), 6
        )
        if dets
        else None,
        "centers": centers,
    }


class FpsMeter:
    def __init__(self, window: int = 60):
        self.timestamps: deque[float] = deque(maxlen=max(2, window))

    def tick(self, timestamp: float | None = None) -> float:
        self.timestamps.append(time.perf_counter() if timestamp is None else timestamp)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def value(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class CpuMonitor:
    """Sample process and system CPU percentages using ``psutil`` when available."""

    def __init__(self):
        try:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(None)
            psutil.cpu_percent(None)
        except ImportError:
            self.psutil = None
            self.process = None

    def sample(self) -> dict[str, float | int | None]:
        if self.psutil is None:
            return {
                "process_cpu_percent": None,
                "system_cpu_percent": None,
                "logical_cpu_count": os.cpu_count(),
            }
        return {
            # psutil process CPU can exceed 100% when multiple cores are used.
            "process_cpu_percent": round(float(self.process.cpu_percent(None)), 2),
            "system_cpu_percent": round(float(self.psutil.cpu_percent(None)), 2),
            "logical_cpu_count": int(self.psutil.cpu_count(logical=True) or 1),
        }


@dataclass
class RuntimeStats:
    frame_count: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    fps_samples: list[float] = field(default_factory=list)
    conversion_ms: list[float] = field(default_factory=list)
    cpu_samples: list[dict[str, Any]] = field(default_factory=list)
    conversion: str = "unknown"

    def add(
        self,
        *,
        fps: float,
        conversion_ms: float,
        cpu: dict[str, Any],
        conversion: str = "unknown",
    ) -> None:
        self.frame_count += 1
        if fps > 0:
            self.fps_samples.append(float(fps))
        self.conversion_ms.append(float(conversion_ms))
        self.cpu_samples.append(cpu)
        self.conversion = conversion

    def summary(self, actual_pixel_format: str, requested_pixel_format: str) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        process_cpu = [
            float(s["process_cpu_percent"])
            for s in self.cpu_samples
            if s.get("process_cpu_percent") is not None
        ]
        system_cpu = [
            float(s["system_cpu_percent"])
            for s in self.cpu_samples
            if s.get("system_cpu_percent") is not None
        ]
        return {
            "frames": self.frame_count,
            "elapsed_s": round(float(elapsed), 4),
            "processed_fps": round(float(self.frame_count / elapsed), 3),
            "interval_fps_mean": round(float(np.mean(self.fps_samples)), 3)
            if self.fps_samples
            else None,
            "interval_fps_p95": round(float(np.percentile(self.fps_samples, 95)), 3)
            if self.fps_samples
            else None,
            "conversion_ms_mean": round(float(np.mean(self.conversion_ms)), 4)
            if self.conversion_ms
            else 0.0,
            "conversion_ms_p95": round(float(np.percentile(self.conversion_ms, 95)), 4)
            if self.conversion_ms
            else 0.0,
            "process_cpu_percent_mean": round(float(np.mean(process_cpu)), 2)
            if process_cpu
            else None,
            "system_cpu_percent_mean": round(float(np.mean(system_cpu)), 2)
            if system_cpu
            else None,
            "requested_pixel_format": requested_pixel_format,
            "actual_camera_pixel_format": actual_pixel_format,
            "conversion": self.conversion,
        }
